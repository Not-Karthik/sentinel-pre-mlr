# Video analysis in the Sentinel ADK agent

Video review runs inside the existing Sentinel ADK agent (`app/`), on the same Agent Runtime engine as the image/text MLR flow. It makes one model call, returns timestamped findings, and attaches a bounding box to anything visible on screen. It reuses the prompt and category list of the FastAPI video path (`api/`) without importing from it.

Evidence labels: **[Measured]** run on the live stack on 2026-11-04; **[Code]** read from the repo or the installed ADK 2.12.0; **[Not verified]** not tested.

## How it works

```
START
  -> route_media            routes by input; emits a route only (no output)
       |- "video"      -> video_reviewer          one LlmAgent call, media_resolution HIGH
       |                    -> finalize_video_report   validates boxes and time ranges   [terminal]
       `- "mlr_review" -> _load_custom_rules -> intake -> ... -> synthesizer   (unchanged image/text flow)
```

- **`MediaInputPlugin`** (`app/app_utils/media.py`) runs before the workflow. It turns YouTube URLs and `gs://` URIs found in the user message into `file_data` parts and sets `video_metadata.fps` on every video part. A client-supplied fps is kept and clamped to 0.5 / 1 / 2.
- **`route_media`** (`app/agent.py`) sends the message to `video_reviewer` when any part is a video, otherwise to the existing flow. It emits no output, so the media is not sent a second time. ADK includes the root user message in every LLM node, so the video branch must have one LLM node [Code].
- **`video_reviewer`** is one `LlmAgent`: prompt `app/prompts/video_reviewer.md`, `output_schema=VideoReport`, `media_resolution` from `VIDEO_MEDIA_RESOLUTION`.
- **`finalize_video_report`** (`app/app_utils/video_report.py`) checks every bbox (length 4, 0..1, ordered, not degenerate), rescales a 0..1000 box, drops unusable ones, swaps an inverted time range, and recomputes `counts_by_severity`.
- **`UsageLoggingPlugin`** (`app/app_utils/usage.py`) logs `usage_metadata` per model call (prompt, video/audio/text details, candidates, thoughts).

## Input

| How | Example |
|---|---|
| Text message with a `gs://` URI (must be inside the allow-listed folder) | `Review gs://pre-mlr-bucket-001/dev/9_0.mp4` |
| Text message with a public YouTube URL | `Review https://www.youtube.com/watch?v=ly9epln4fbE` |
| Structured message (stream / async stream query) | `{"role":"user","parts":[{"fileData":{"fileUri":"gs://...","mimeType":"video/mp4"},"videoMetadata":{"fps":1}},{"text":"Review this video"}]}` |
| Attached file in `adk web` | inline base64; small clips only (about 20 MB request limit per the Gemini docs) |

- The ADK web UI cannot attach a `gs://` file; type the URI as message text [Code].
- `gs://` URIs are checked against `gs://{GCS_BUCKET_NAME}/{GCS_MEDIA_FOLDER}/`; other buckets, other folders, `..`, backslashes and NUL are rejected [Measured: three bad URIs rejected].
- YouTube URLs must be public (Gemini docs).
- `image/gif` stays on the image flow, which most likely analyses a single frame; GIF animation is not analysed.

## Output

The final event is the JSON of `VideoReport` (also in the event `output`):

```json
{
  "content_summary": "...",
  "findings": [
    {
      "finding_id": "V-1",
      "category": "misleading_claim",
      "severity": "medium",
      "observation": "...",
      "quoted_content": "...",
      "start_timestamp": "00:02.8",
      "end_timestamp": "00:04",
      "location": {"section": "screen top right parameter", "bbox": [0.58, 0.26, 0.71, 0.42]}
    }
  ],
  "counts_by_severity": {"medium": 1}
}
```

- `start_timestamp` is the first moment the issue is clearly visible or audible, as `MM:SS` with an optional decimal (`00:02.5`) or `HH:MM:SS`.
- `location.bbox` is `[x_min, y_min, x_max, y_max]`, fractions of the frame width and height, origin top-left, for the frame at `start_timestamp`. `location` is null for findings with no place in the frame (narration, audio, captions, pacing).
- Categories: `medical_accuracy, citation_missing, misleading_claim, outdated_information, unverified_statement, contraindication, dosage_concern, presentation_style, wording_concern, visual_quality, audio_quality, accessibility, professionalism, other`. Severity: `informational, low, medium, high, critical`.
- Drawing the box and seeking the player are the frontend's job; the ADK web UI shows the JSON only [Code].

## Configuration

| Variable | Values | Default | Note |
|---|---|---|---|
| `VIDEO_FPS` | 0.5, 1, 2 | 1 | invalid value falls back to the default with a warning |
| `VIDEO_MEDIA_RESOLUTION` | HIGH, DEFAULT | HIGH | DEFAULT leaves the model's own setting |
| `GCS_BUCKET_NAME` | bucket | none | same variable the API uses |
| `GCS_MEDIA_FOLDER` | folder | `dev` | same variable the API uses |

Local: add them to the `.env` at the sentinel root; ADK walks up from the agent folder to find it, and shell variables win over `.env` [Code]. Deployed: pass them as environment variables (Cloud Build `_ENV_VARS`, semicolon-separated). How to pass env vars with `agents-cli deploy` directly is **[Not verified]**.

## Run locally

```
uv run adk web app --session_service_uri="agentengine://projects/pre-mlr/locations/us-central1/reasoningEngines/6707271892962639872"
```

Open a new session and send a `gs://` URI or YouTube URL as text. The identity making the model call needs read access to the object (locally your `gcloud auth application-default login` user) [Measured for the local user].

## Measured results (gemini-3.8-flash, one run each, so treat timings as ranges)

| Input | fps / res | Video tok | Prompt tok | Output tok (answer + thinking) | Time | Findings (with bbox) |
|---|---|---|---|---|---|---|
| Local MP4, 8 s, attached inline | 1 / HIGH | 2,112 | 4,167 | 4,161 | 41.2 s | 21 (17) |
| Local MP4, 8 s, attached inline | 2 / HIGH | 4,224 | 6,279 | 3,652 | 46.9 s | 18 (14) |
| `gs://pre-mlr-bucket-001/dev/9_0.mp4` (8 s) | 1 / HIGH | 2,112 | 4,186 | 9,879 | 64.3 s | 22 (16) |
| YouTube `ly9epln4fbE` (about 61 s) | 1 / HIGH | 16,104 | 19,503 | 6,415 | 199.3 s | 25 (14) |
| Same ad as a local file, `gs://pre-mlr-bucket-001/dev/ozempic-commercial-2023-test.mp4` (60.8 s, 1920x1076) | 1 / HIGH | 16,104 | 19,508 | 6,339 | 50.5 s | 22 (13) |

- The video is sent once: video tokens equal frames x 264 at HIGH (8 s = 2,112; 2 fps = 4,224; 61 s = 16,104).
- At $0.75 in / $3.75 out per 1M tokens (through 2026-12-31; $1.50 / $7.50 after): the local 1 fps run is $0.019, the `gs://` run $0.040, the YouTube run $0.039. Output (mostly thinking) is the larger part; thinking varied from 0.8k to 5.9k tokens for the same 8 s clip.
- Latency is volatile: the same YouTube video took 51 s in an earlier direct-SDK run and 199 s here, with similar token counts. The 1-2 minute target is not guaranteed for a 61 s video.
- An earlier schema that added a `timestamp` inside `location` made the model write its reasoning into that field and no bbox at all; it was removed and `ContentLocation` is unchanged.

## Why bbox are approximate

Checked by drawing every box on the actual frame:

1. **Video frames carry far less visual detail than still images** [Measured]. A video frame at HIGH is 264 tokens; the same 1280x720 frame sent as a still image is about 1.3k tokens. Small text and UI elements are boxed coarsely.
2. **The box is tied to one moment, which the model may pick loosely.** Before the prompt was tightened, boxes fitted the frame about 0.5 s later than `start_timestamp`; the camera in the test clip moves and zooms fast, so half a second changes everything. The prompt now defines `start_timestamp` as the first clearly visible moment and allows tenths of a second. After that, boxes such as the "96" readout and the ECG waveform were exact, others were still loose.
3. **Some boxes point at things that are not in the frame** (for example a subtitle that does not exist), because findings come from the narration or the model's expectation.
4. **Not the causes:** the axis order (swapping x and y was clearly wrong), and fps (2 fps was no better than 1).

A second pass that boxes each finding on the extracted still frame was measured on the same clip: it roughly doubled the cost (+11.5k input, +14.0k output tokens, +$0.019), added about 69 s with limited concurrency (eight parallel calls hit `429 RESOURCE_EXHAUSTED`), and the accuracy was mixed (tighter on small text, worse on one large box, one box lost). It was not adopted.

## Visual check of the boxes

Frames of `gs://pre-mlr-bucket-001/dev/9_0.mp4` (same file as the local test MP4; MD5 matches) with the boxes from the final `gs://` run, drawn at each finding's `start_timestamp`:

- `docs/images/bbox-gcs-9_0-a.png`: 0.5 s, 1.8 s, 2.0 s, 3.2 s
- `docs/images/bbox-gcs-9_0-b.png`: 5.2 s, 5.5 s, 5.6 s

My reading of those 16 boxes (not a measured score):

| | Findings |
|---|---|
| On target | V-20 (ghost "VITALYNX" title), V-21 and V-22 (disclaimer text, just above it), V-15 (the screen, coarse) |
| Near but loose | V-4, V-11, V-18, V-19 |
| Miss | V-7, V-8, V-9, V-10, V-12, V-13, V-14, V-17 |

An earlier run on the same file hit other items exactly (the "96" readout, the ECG waveform) and missed others, so accuracy varies from run to run. Treat boxes as approximate. The YouTube video could not be drawn because the agent never receives its frames (Gemini reads the URL server-side); drawing it needs a local copy of the file.

### Visual check on the Ozempic ad (60.8 s, 1080p, `gs://` copy)

Boxes from the `gs://` run drawn on the frames of the downloaded file at each `start_timestamp` (images are not stored in the repo because they contain third-party ad frames). My reading of the 13 boxes:

| | Findings |
|---|---|
| On target | V-2 (disclaimer text), V-8 (study footnote lines), V-17 (side-effect text), V-22 (the "$25" callout) |
| Partly right | V-3, V-6, V-11, V-15 |
| Miss | V-1 (box on a black frame), V-4 (dosage text appears later than 4 s), V-10, V-12, V-21 |

Boxes on text blocks and callouts were mostly good; boxes for abstract graphics, and findings whose item is visible at a different moment than `start_timestamp`, tended to miss.

## Schema alignment with the image flow

| | Image `Finding` / `FinalReport` | Video `VideoFinding` / `VideoReport` |
|---|---|---|
| `finding_id`, `observation`, `quoted_content` | yes | yes (same names) |
| `severity` | `Severity` | same enum |
| `location` | `ContentLocation` (`page`, `section`, `bbox`, `quote`) | same class, same bbox convention |
| `category` | `IssueCategory` (28 MLR values) | `VideoIssueCategory` (14 API values, adds `audio_quality`) |
| Time range | none | `start_timestamp`, `end_timestamp` |
| `review_lens`, `confidence`, `confidence_band`, `evidence_depth`, `mlr_principle`, `discussion`, `suggested_*`, `related_item_ids`, `citations` | yes | **no** (kept out for short output) |
| Report wrapper | `content_summary`, `intended_audience`, `promotional_intent`, `executive_summary`, `themes`, `findings`, `open_questions_for_reviewers`, `recommended_discussion_topics`, `counts_by_lens`, `counts_by_severity` | `content_summary`, `findings`, `counts_by_severity` |

The overlay-relevant parts (`finding_id`, `severity`, `observation`, `location.bbox`) are the same, so one bbox overlay component can serve both. The frontend's current viewer types (`frontend/src/viewer/types.ts`) mark `review_lens`, `category`, `confidence`, `confidence_band`, `evidence_depth`, `mlr_principle` and `discussion` as required, so a strictly typed renderer needs them optional or a separate video type. The report has no `media_type` field yet; adding one would let a client choose the renderer.

## Tests

`tests/unit/test_media_input.py` (URL conversion, allow-list, fps clamping, routing, plugin + router through an in-memory runner), `tests/unit/test_video_report.py` (schemas, bbox/time-range validation, final node, usage formatting), plus the updated terminal-node assertion in `tests/unit/test_resume.py`.
`tests/unit/test_storage_security.py` currently fails at collection for an unrelated reason: the root `.env` has `GOOGLE_CLOUD_AGENT_ENGINE_ID` and `GOOGLE_CLOUD_AGENT_ENGINE_LOCATION`, which the API's `Settings` rejects as extra inputs.

## Open items

- Deployed engine: not tested. Check that `App(plugins=...)` is honoured by the Agent Runtime wrapper, that `file_data` survives `stream_query`, and that the Agent Runtime identity can read the media bucket.
- Agent Runtime request duration and payload limits for long videos.
- A video longer than a few minutes has not been run. At 1 fps HIGH, input is about 289 tokens per second of video (about 53k for 3 minutes, extrapolated); the model's input limit is 1,048,576 tokens.
- Prompt asks for 20-40 findings, which is a lot for an 8 s clip; tune after the vendor reviews real output.
- Several boxes per finding over time, GIF animation, custom rules in the video branch: out of scope for now.
