# Video analysis in the Sentinel ADK agent

Video review runs inside the existing Sentinel ADK agent (`app/`), on the same Agent Runtime engine as the image/text MLR flow. It makes one model call, returns timestamped findings, and attaches a bounding box to anything visible on screen. It reuses the prompt and category list of the FastAPI video path (`api/`) without importing from it.

Evidence labels: **[Measured]** run on the live stack on 2026-11-04; **[Code]** read from the repo or the installed ADK 2.12.0; **[Not verified]** not tested.

## How it works

```
START
  -> route_media            routes by input; emits a route only (no output)
       |- "video"      -> video_reviewer          pass 1: one LlmAgent call on the video, media_resolution HIGH
       |                    -> refine_video_boxes     pass 2: ffmpeg stills + direct model calls (not agents)
       |                    -> finalize_video_report   validates boxes and time ranges   [terminal]
       `- "mlr_review" -> _load_custom_rules -> intake -> ... -> synthesizer   (unchanged image/text flow)
```

- **`MediaInputPlugin`** (`app/app_utils/media.py`) runs before the workflow. It turns YouTube URLs and `gs://` URIs found in the user message into `file_data` parts and sets `video_metadata.fps` on every video part. A client-supplied fps is kept and clamped to the allowed set (0.5, 1, 2, 4, 8, 16).
- **`route_media`** (`app/agent.py`) sends the message to `video_reviewer` when any part is a video, otherwise to the existing flow. It emits no output, so the media is not sent a second time. ADK includes the root user message in every LLM node, so the video branch must have one LLM node [Code].
- **`video_reviewer`** is one `LlmAgent`: prompt `app/prompts/video_reviewer.md`, `output_schema=VideoReport`, `media_resolution` from `VIDEO_MEDIA_RESOLUTION`. Its instruction is a function (`_video_instruction` in `app/agent.py`) that appends a short note built per request from the fps actually set on the video part, so a client-supplied fps is respected (for example "the frames are 0.5 seconds apart; report timestamps that match a sampled frame"). The prompt also tells the model to leave `bbox` null, or `location` null, when it cannot clearly see the item at `start_timestamp`, and never to box black or empty areas.
- **`refine_video_boxes`** (`app/agent.py`, logic in `app/app_utils/bbox_refine.py` and `frames.py`) is pass 2 (on by default; `VIDEO_BBOX_REFINE=false` turns it off). It is a plain function node, not an agent: for each distinct finding start time it cuts the frames at -0.5, 0, +0.5 and +1.0 s with ffmpeg (bundled via `imageio-ffmpeg`, or one on `PATH`), sends those still images to the model in one call, and gets back the earliest frame where each item is visible plus a tight box (`[ymin, xmin, ymax, xmax]` 0-1000, converted to the usual normalised x/y). The finding's `start_timestamp` moves to that frame. It needs a video the agent can read: a `gs://` object (downloaded to a temp folder, deleted afterwards) or an attached file. YouTube URLs are skipped, as is any failure; the pass-1 boxes are then kept. It is not the image MLR flow, which is unrelated.
- **`finalize_video_report`** (`app/app_utils/video_report.py`) checks every bbox (length 4, 0..1, ordered, not degenerate), rescales a 0..1000 box, drops unusable ones, swaps an inverted time range, and recomputes `counts_by_severity`.
- **`UsageLoggingPlugin`** (`app/app_utils/usage.py`) logs `usage_metadata` per model call (prompt, video/audio/text details, candidates, thoughts).

## Flow from input to final report

```
user message (text with a gs:// or YouTube URL, or an attached video)
  -> MediaInputPlugin     before the workflow starts:
                          - turns the URL into a file_data part (mime from the extension)
                          - checks gs:// against gs://<bucket>/<folder>/
                          - sets video_metadata.fps = 4 on every video part
  -> route_media          video part present -> video branch, else the image/text flow
  -> PASS 1: video_reviewer (one LlmAgent call)
        sees: the whole video at 4 fps, media_resolution HIGH, plus its audio
        prompt: video_reviewer.md plus a note saying how far apart the frames are
        returns VideoReport: 20-40 findings, each with
          category, severity, observation, quoted_content, start/end timestamp,
          and optionally location {section, bbox} (a first-guess box)
  -> PASS 2: refine_video_boxes (a function node; runs if VIDEO_BBOX_REFINE=true)
        1. find the video in the session (gs:// object or attached bytes)
           and download it to a temp folder (cap VIDEO_REFINE_MAX_MB, default 500)
        2. take findings that have a box and group them by start time (to 0.1 s)
        3. for each group, cut stills at start -0.5, 0, +0.5, +1.0 s with ffmpeg
        4. one model call per group, sending up to 4 stills plus the findings' text;
           the model returns the earliest frame where each item is clearly visible
           and a tight box on that frame
        5. replace the box and move start_timestamp to that frame
        6. delete the temp folder
  -> finalize_video_report   validates boxes and times, recounts severities
  -> output, with the events stored in the session
```

If pass 2 is off, or the video cannot be read, pass 1's report goes straight to `finalize_video_report`. YouTube links always take that path.

### How many LLM calls

- **Pass 1:** 1 call (`video_reviewer`). The node retries up to 3 times on an error; no retry occurred in the runs below.
- **Pass 2:** one call per distinct start time (to 0.1 s) among findings that have a box. That is not one call per finding, and no still is sent twice. They are direct model calls, not agents, with thinking level low and at most 3 running at once (`VIDEO_REFINE_CONCURRENCY`).
- **Measured locally:** 8 s clip, 9 calls for 16 boxed findings (24 stills); 61 s ad, 14 calls for 15 boxed findings (52 stills).
- **Deployed runs (inferred):** 12 calls for the 8 s clip (15 boxed findings) and 20 for the 61 s ad (20 boxed findings), counted from the distinct start times in the pass-1 report. The engine's INFO log line was not visible, so these were not confirmed.
- **Total:** about 1 + N calls per run, where N is at most the number of boxed findings (at most 40).

## Why two passes, and how they work

**The problem with one pass.** Pass 1 sends the whole video to the model once. At 4 fps and HIGH resolution each sampled frame costs about 264 tokens, which is enough to decide what is said and shown and when, but not enough to place a box precisely: small text and UI elements are boxed coarsely, and the model tends to report a start time slightly before the item is clearly visible, so the box can fit a frame a moment later. Single-pass boxes were wrong or loose in many findings (see "Why bbox are approximate").

**What each pass does.**

| | Pass 1: `video_reviewer` | Pass 2: `refine_video_boxes` |
|---|---|---|
| Input | the whole video at 4 fps (video, audio and prompt) | still frames cut with ffmpeg around each finding's start time (width capped at 1280 px) |
| Model call | one `LlmAgent` call | direct model calls from a function node, up to 3 at a time, thinking level low |
| Decides | what the issues are: category, severity, observation, quoted content, time range, first-guess box | where the item is: the earliest frame in which it is clearly visible, and a tight box on that frame |
| Writes | the whole `VideoReport` | only `location.bbox` and `start_timestamp` (and `end_timestamp` if it would end before the new start) |

**Steps in pass 2** (`app/app_utils/bbox_refine.py`):

1. Take every finding that has a box and group them by start time (to 0.1 s).
2. For each group, cut stills at start -0.5, 0, +0.5 and +1.0 s (skipping negative times). The window reaches forward because pass 1 usually reports an item slightly early.
3. Send the stills for one group, in one call, with the findings' region, quoted text and observation, and ask for each finding the index of the earliest frame in which the item is clearly visible plus a box on that frame (`[ymin, xmin, ymax, xmax]`, 0-1000).
4. Convert to the usual normalised `[x_min, y_min, x_max, y_max]`, replace the box, and move `start_timestamp` to the chosen frame's time.
5. `finalize_video_report` then validates the boxes and time ranges.

**What it improves** (single runs, my reading of the screenshots; see "Two-pass results with the new defaults" for tokens, time and cost):

| Video | Boxes on target, pass 1 only (4 fps) | Boxes on target, pass 1 + pass 2 | Start times moved by pass 2 |
|---|---|---|---|
| 8 s clip | 13 of 17 | 15 of 16 | 13 of 16 |
| 61 s ad | 10 of 20 | 13 of 15 (plus 2 boxes covering the whole frame) | 13 of 15 |

- Boxes are drawn on a still image, which carries more detail than a sampled video frame, and the model picks the frame in which the item is visible, so the box and `start_timestamp` agree. Clicking a finding in the player should therefore land on a frame where the box fits.
- Pass 2 does not change what is reported: findings, categories, severities and observations stay as pass 1 wrote them.

**What it does not do.**

- It does not add findings or boxes: a finding with `bbox` null stays null, and a finding is re-boxed only if pass 1 gave it a box.
- It does not search the whole video: if the item is more than 1 s after the start time pass 1 gave, or before it by more than 0.5 s, it is not in the window (the model then reports it as not visible and the pass-1 box is kept).
- It cannot run on YouTube links, which the agent cannot download.

**Failure behaviour.** The pass-1 box and time are kept when the stills cannot be cut, a model call fails, the item is reported as not visible, or the video cannot be read. Pass 2 is switched off with `VIDEO_BBOX_REFINE=false`, and `VIDEO_REFINE_CONCURRENCY` (default 3; 8 parallel calls hit 429) and `VIDEO_REFINE_MAX_MB` (default 500) tune it.

**Cost of the second pass.** Pass 2 roughly doubles the model cost and adds 47 s (8 s clip) to 121 s (61 s ad) in the runs above. It scales with the number of distinct start times, not with video length: 24 stills in 9 calls for the 8 s clip, 52 stills in 14 calls for the 61 s ad. Longer videos than 61 s were not run.

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

## Routing: video flow or image/text flow

The flow is chosen by the MIME type of the media in the message, not by the file name alone. `route_media` sends the message to the video flow when any part (`file_data` or `inline_data`) has a MIME type starting with `video/`. Anything else (plain text, images, PDFs, GIFs, no media) runs the existing image/text flow [Code: `media.has_video`]. A message with both a video and an image goes to the video flow.

Where the MIME type comes from:

| Input | MIME type used |
|---|---|
| `gs://` URI typed as text | From the file extension: `mp4` video/mp4, `mpeg` video/mpeg, `mpg` video/mpg, `mov` video/quicktime, `avi` video/avi, `wmv` video/wmv, `webm` video/webm; `jpg`/`jpeg`/`png`/`webp`/`gif` become `image/*`. Any other extension is left as plain text and no media is attached |
| YouTube URL typed as text | Always assumed `video/mp4` |
| Structured message with `fileData` / `inlineData` | The `mimeType` the client sends |
| File attached in `adk web` | The type the browser reports for the file |

The code never opens the file to check its content, so a video stored with an image extension would be treated as an image (inferred from the code, not tested).

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
| `VIDEO_FPS` | 0.5, 1, 2, 4, 8, 16 | 4 | invalid value falls back to the default with a warning |
| `VIDEO_MEDIA_RESOLUTION` | HIGH, DEFAULT | HIGH | DEFAULT leaves the model's own setting |
| `VIDEO_BBOX_REFINE` | true, false | **true** | pass 2 (see below); set to false for the single video call only |
| `VIDEO_REFINE_CONCURRENCY` | integer | 3 | parallel model calls in pass 2 (eight at once hit 429s) |
| `VIDEO_REFINE_MAX_MB` | integer | 500 | videos above this skip pass 2 |
| `GCS_BUCKET_NAME` | bucket | none | same variable the API uses |
| `GCS_MEDIA_FOLDER` | folder | `dev` | same variable the API uses |

All of them are listed in `.env.example`; an unset variable uses the default above. Local: add them to the `.env` at the sentinel root; ADK walks up from the agent folder to find it, and shell variables win over `.env` [Code]. Deployed: pass them as environment variables (Cloud Build `_ENV_VARS`, semicolon-separated). How to pass env vars with `agents-cli deploy` directly is **[Not verified]**.

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

## How Gemini counts a video (tokens)

- The video is split into frames on Google's side. `VIDEO_FPS` (sent as `video_metadata.fps`) sets how many frames per second are sampled, and `VIDEO_MEDIA_RESOLUTION` sets how much detail each frame gets. The Gemini docs give a default of 1 fps and about 66 tokens per frame at default detail and 258 at high [External]; I measured 66 and 264.
- The API reports the tokens by type in `usage_metadata.prompt_tokens_details`: **VIDEO** (the sampled frames, reported as video, not as image), **AUDIO** and **TEXT** (our prompt, about 1.9k). How a frame is encoded internally is not exposed.
- **Audio** is added automatically when the file has an audio track and is counted separately: about 25 tokens per second here (1,526 for the 60.8 s ad, 200 for the 8 s clip; the docs say 32) [Measured]. A video with no audio track (a GIF converted to MP4) showed no audio tokens [Measured]. Audio does not change with fps or resolution.
- Both settings take effect. Measured on the 61 s ad (direct call):

| Setting | VIDEO tokens | AUDIO tokens |
|---|---|---|
| 0.5 fps, default detail | 1,980 | 1,526 |
| 1 fps, default detail | 4,026 | 1,526 |
| 2 fps, default detail | 8,052 | 1,526 |
| 1 fps, HIGH | 16,104 | 1,526 |
| 2 fps, HIGH | 32,208 | 1,526 |

- So 2 fps HIGH is about 528 video tokens per second plus about 25 audio tokens per second (35,168 input tokens for the 61 s ad including the text prompt), against about 289 per second at 1 fps HIGH. It doubles the video tokens and cost; in my tests it did not make the boxes better.
- Billing: the Gemini API pricing page lists one input rate for text, image, video and audio tokens, and a separate output rate that includes thinking tokens [External]. The Vertex page was not read directly.

## Why bbox are approximate

Checked by drawing every box on the actual frame:

1. **Video frames carry far less visual detail than still images** [Measured]. A video frame at HIGH is 264 tokens; the same 1280x720 frame sent as a still image is about 1.3k tokens. Small text and UI elements are boxed coarsely.
2. **The box is tied to one moment, which the model may pick loosely.** Before the prompt was tightened, boxes fitted the frame about 0.5 s later than `start_timestamp`; the camera in the test clip moves and zooms fast, so half a second changes everything. The prompt now defines `start_timestamp` as the first clearly visible moment and allows tenths of a second. After that, boxes such as the "96" readout and the ECG waveform were exact, others were still loose.
3. **Some boxes point at things that are not in the frame** (for example a subtitle that does not exist), because findings come from the narration or the model's expectation.
4. **Not the causes:** the axis order (swapping x and y was clearly wrong), and fps (2 fps was no better than 1).

### Pass 2: boxes on still frames (`refine_video_boxes`, on by default)

Pass 2 is on by default together with 4 fps. It roughly doubles the cost and adds tens of seconds to a couple of minutes; `VIDEO_BBOX_REFINE=false` gives the single video call only. It needs a `gs://` or attached video, so a YouTube link only gets pass 1.

First tried as a throwaway test with one frame per finding (about double the cost, mixed accuracy, one box lost), then built into the agent with three changes: a window of several frames per finding so the model can pick the one where the item is visible, a low thinking level for these calls, and falling back to the pass-1 box when the item is not found. Result on the 8 s clip (`gs://.../9_0.mp4`, frames drawn at each finding's final `start_timestamp`):

| Run | Window | Findings with box | Frames / calls | Pass-2 tokens (in / out) | Whole run | On target (my reading) |
|---|---|---|---|---|---|---|
| A | -0.5, 0, +0.5 s | 15 | 18 / 8 | 30,913 / 1,340 | 99.6 s | 10 of 15 |
| B | -0.5, 0, +0.5, +1.0 s | 19 | 30 / 8 | 40,112 / 1,422 | 77.5 s | 17 of 19 |

- In run B the small readouts (all eight at 2.5 s), the Apple logo, the title, the tagline and the disclaimer were boxed tightly; the two misses were an item not yet visible at its start time and one box on the frame edge. Run-to-run variance is large, so these are single runs.
- Most start times moved by +0.5 s (10 of 15 in run A): pass 1 tends to report an item slightly before it is clearly visible, which is why the window reaches forward.
- Cost and time: pass 2 is roughly as expensive as pass 1 (run B: pass 1 about $0.035, pass 2 about $0.035) and adds tens of seconds. Pass 2 was not timed separately in these runs (the log line now reports `elapsed_s`).
- Both are single runs of an 8 s clip. The 61 s Ozempic ad was run with pass 2 later, at the new defaults (see "Two-pass results with the new defaults").

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

### Sampling note and null-box rule (latest single-pass runs)

Added to the prompt: the sampling note above and the rule to leave `bbox` null when unsure. Single runs, second pass off, 1 fps / HIGH unless stated:

| Clip | fps | Time | Prompt tok | Output tok | Findings (with box) | `location` null | Start times |
|---|---|---|---|---|---|---|---|
| 8 s (`gs://.../9_0.mp4`) | 1 | 53.5 s | 4,343 | 7,141 | 22 (18) | 4 | whole seconds |
| 8 s | 2 | 32.5 s | 6,459 | 5,245 | 20 (16) | 4 | 0.5 s steps |
| 61 s Ozempic ad (`gs://` copy) | 1 | 76.8 s | 19,665 | 9,977 | 23 (17) | 6 | whole seconds |

- **The note is dynamic and the model follows it:** start times are multiples of the frame spacing (1 s at 1 fps, 0.5 s at 2 fps). The note adds about 160 text tokens.
- **Boxes, my reading of the screenshots:** 61 s ad: 9 of 17 on target, 3 partly right, 5 wrong (the earlier run without these prompt changes had 2 on target out of 11). The 5 wrong ones are mostly timing: the dosage text appears about 2 s after the reported 4 s, the "lost up to 14 lbs" headline at 24 s not 23 s, the "$25" callout at 59 s not 58 s. 8 s clip at 1 fps: about 11 of 18 acceptable (several coarse), the rest on tiny screen labels or on items not yet visible. At 2 fps the disclaimer boxes were exact.
- **The null rule did not visibly reduce the number of boxes** (17 of 23 findings still have one), and the model still boxes items before they appear. No box sat entirely on black in these runs, but this is one run each, and the earlier black-frame box came from a different run.
- Screenshots: `prompt-note-*` files in the screenshots folder. The JSON the agent returned for each run (the `VideoReport`, with findings tagged V-1, V-17, ...) is saved unmodified next to the screenshots with the same prefix, for example `prompt-note-61s-fps1.json`.

### Frame rate comparison (single runs, HIGH resolution, second pass off)

The agent now accepts 0.5, 1, 2, 4, 8 and 16 fps (default 4). The runs below were made before the default changed, with the limit lifted in a test script for 4, 8 and 16 fps. The API itself allows up to 24 fps. "On target" is my reading of the screenshots, not a measured score.

8 s clip (`gs://.../9_0.mp4`):

| fps | Video tokens | Time | Cost | Boxes on target |
|---|---|---|---|---|
| 1 | 2,112 | 53.5 s, 71.1 s | $0.030, $0.047 | about 7 of 21 |
| 2 | 4,224 | 32.5 s | $0.025 | 4 of 10 reviewed |
| 4 | 8,448 | 53.6 s | $0.026 | 13 of 17 |
| 8 | 16,896 | 57.0 s | $0.040 | 9 of 15 |
| 16 | 33,792 | 53.3 s | $0.058 | about 7 of 17 |

61 s Ozempic ad (`gs://` copy):

| fps | Video tokens | Time | Cost | Boxes on target |
|---|---|---|---|---|
| 1 | 16,104 | 50.5 s to 449.9 s (three runs) | $0.034 to $0.052 | 9 of 17 |
| 4 | 64,152 | 98.0 s | $0.085 | 10 of 20 (4 partly right, 2 whole-frame, 4 wrong) |
| 8 | 128,568 | 473.7 s, including a failed first attempt | $0.127 for the successful call, about $0.385 as billed | 11 of 18 (4 whole-frame, 3 wrong) |

- **Cost:** video tokens are fps x 264 per second, so cost grows with fps. Time does not follow fps, because the thinking tokens vary far more.
- **Accuracy:** gains are clear from 1 to 4 fps. 4 fps matched 8 fps on both videos, and 16 fps showed no further gain. Most of the remaining misses are items that appear a little after the reported time.
- **Limits:** at HIGH resolution the 1,048,576-token window fits about 31 min at 2 fps, 16 min at 4 fps and 8 min at 8 fps.
- **Runaway output:** one 8 fps run had a first attempt whose output ran away (about 62k tokens) and produced invalid JSON; ADK retried it and the run was billed for both attempts. No other run, out of about 30, did this. `max_output_tokens` is 65,535; a lower cap would limit the cost of such a failure.
- **Default:** 4 fps, which suits videos up to about 5 minutes; use 2 fps for longer ones by setting `VIDEO_FPS` (the agent does not pick the fps from the video length). This is from single runs and has not been tried on other videos.

### Two-pass results with the new defaults (4 fps, pass 1 + pass 2)

Single runs on the `gs://` copies, no retries or rate-limit errors. "On target" is my reading of the screenshots.

| Video | Total time | Pass 1 | Pass 2 | Cost (both passes) | Boxes on target |
|---|---|---|---|---|---|
| 8 s clip | 140.3 s | 10,685 in / 4,603 out, about $0.025 | 24 frames, 9 calls, 44,517 in / 806 out, 46.6 s, about $0.036 | about $0.062 | 15 of 16 |
| 61 s ad | 191.5 s | 67,717 in / 5,961 out, about $0.073 (about 71 s) | 52 frames, 14 calls, 69,564 in / 1,261 out, 120.9 s, about $0.057 | about $0.130 | 13 of 15, plus 2 boxes covering the whole frame |

- Pass 1 alone at 4 fps gave 13 of 17 (8 s) and 10 of 20 (61 s), against 15 of 16 and 13 of 15 with pass 2 (my reading of single runs).
- Pass 2 moved the start time of 13 findings in each run to the frame where the item is visible; no finding fell back to the pass 1 box.
- Pass 2 is roughly as expensive as pass 1. The same videos took 53.6 s (8 s clip) and 98.0 s (61 s ad) with pass 1 only, so with pass 2 the run took about 2.6 and 2.0 times as long (single runs, timings vary a lot); the 61 s ad took 3.2 minutes.
- Not tested: videos longer than 61 s, and videos over a few minutes would also need windowed pass 1 (not built).

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

## Deployed engine test (Agent Runtime, 2026-11-05)

Deployed with `agents-cli deploy --update-only --service-name="Sentinel Agent"` to the existing engine 6707271892962639872 (project `pre-mlr`, `us-central1`), passing `GCS_BUCKET_NAME`, `GCS_MEDIA_FOLDER`, `VIDEO_FPS=4`, `VIDEO_MEDIA_RESOLUTION=HIGH`, `VIDEO_BBOX_REFINE=true`, `VIDEO_REFINE_CONCURRENCY=3` and `VIDEO_REFINE_MAX_MB=500`. Checked afterwards: the engine kept its 14 existing variables and gained these 7.

Deployment notes [Measured]:
- `agents-cli deploy` merges the project-root `.env` into the engine's environment, and the engine rejects the reserved `GOOGLE_CLOUD_AGENT_ENGINE_*` names with a 400. Deploy from a copy of the project without `.env` so secrets such as `GEMINI_API_KEY` are not sent.
- The engine is found by its display name, so `--service-name="Sentinel Agent"` is required, and a `Dockerfile` must be in the project root.
- Pass 2 downloads the video with the agent identity, which first got a 403 (`storage.objects.get` denied) on the bucket. Granting that identity `roles/storage.objectViewer` on the media bucket fixed it. Pass 1 needed no grant: the model read the `gs://` video with the same input tokens as locally.

Results (4 fps, `gs://` video, one run each, no errors). Boxes on target is my reading of the screenshots in `deployed-4fps-*.png`:

| | 8 s clip | 61 s ad |
|---|---|---|
| Total time (same video locally) | 78.4 s (140.3 s) | 147.9 s (191.5 s) |
| Pass 1 tokens, in / out | 10,685 / 3,155 | 67,717 / 5,235 |
| Findings, with a box | 21, 15 | 26, 20 |
| Pass 2 effect | all 15 boxes replaced, 4 start times moved | 19 of 20 boxes replaced, 13 start times moved |
| Boxes on target | 14 of 15 | 17 of 20 |

Sessions [Measured]:
- `async_create_session` works. After a query the session holds 5 events: the user message (the `gs://` link converted to a `file_data` part), `route_media`, `video_reviewer`, `refine_video_boxes` and `finalize_video_report`. `async_get_session` and `async_list_sessions` return it.
- `resume_stream_query` on a finished query returns an empty stream.
- After cutting the client at 30 s (pass 1 already saved), `resume_stream_query` returned only the pass 2 and finalize events; `video_reviewer` ran once, so pass 1 was not repeated.
- The server keeps running a query after the client disconnects. In that test the session ended with pass 2 and finalize recorded twice, 2 min 47 s apart. Engine logs show two pass 2 executions starting 5 s apart: the original run (still going after the client was cut) and the resume, which was called 33 s in. `find_unfinished_invocation_id` (`app/app_utils/resume.py`) treats any invocation without a terminal output as interrupted and does not check whether it is still running, so it resumed the same invocation a second time. The duplicate is therefore caused by resuming while the original run was alive, not by a crash. A resume after the original has really died (container crash) has not been tested. A client that resumes after a dropped stream (`resilient_client.py` waited 2 s first) could hit the same case, and a duplicate costs pass 2 twice.

Fix (deployed to the engine on 2026-11-05 and tested there on the video flow; it applies to the video and image flows because it reads the terminal nodes from the graph, but the image flow was only covered by unit tests): `resume_stream_query` now answers a single `{"status": "invocation_in_progress", "retry_after_seconds": ...}` event instead of resuming when the invocation's newest event is younger than `RESUME_MIN_IDLE_SECONDS` (default 600 s). `"force": true` skips the check; the server's own automatic resume and a retry after a `resumable` failure event use it. `resilient_client.py` polls on that status instead of starting a second run, and reads the stored events of the invocation from the session if it turns out to have finished. This is a time threshold, not a lease: a silent step longer than the threshold can still be resumed twice, and a real crash is only resumed after the threshold unless the caller forces it.

Retest on the deployed engine (8 s clip, fresh session, client cut at 41 s after pass 1 was saved) [Measured]: five `resume_stream_query` calls without `force`, 20 s apart. The first four each returned one `invocation_in_progress` event (the reported idle time grew from 44.5 s to 112.5 s) and ran nothing. The fifth, at 137 s, returned nothing because the original run had finished. The session ended with 5 events and three outputs containing findings (pass 1, pass 2, finalize), against 7 events and pass 2 run twice in the earlier test. The idle time is counted from the newest event's timestamp, and that timestamp looks like the node's start rather than its end (idle was 44.5 s at 47 s into a run whose pass 1 ended at about 41 s), so the idle figure overstates how long a step has been silent.

Seen in the engine logs of that test: `ffmpeg failed at 8.000s` and `8.500s` (candidate frames past the end of the 8 s clip; harmless, the frame is skipped), and one `Box refinement failed for frames at 4.0s: 429 RESOURCE_EXHAUSTED` while the two pass 2 executions overlapped, so that group kept its pass-1 boxes.

Not measured: pass 2 tokens and cost on the deployed engine, a container crash with automatic resume, and the 401 warnings about the `pre-mlr-sentinel-agent-logs` bucket seen in the engine logs (not checked whether they predate this deploy).

## Tests

`tests/unit/test_media_input.py` (URL conversion, allow-list, fps clamping, routing, plugin + router through an in-memory runner), `tests/unit/test_video_report.py` (schemas, bbox/time-range validation, final node, usage formatting), plus the updated terminal-node assertion in `tests/unit/test_resume.py`.
`tests/unit/test_storage_security.py` currently fails at collection for an unrelated reason: the root `.env` has `GOOGLE_CLOUD_AGENT_ENGINE_ID` and `GOOGLE_CLOUD_AGENT_ENGINE_LOCATION`, which the API's `Settings` rejects as extra inputs.

## Open items

- Deployed engine: tested on 8 s and 61 s (see "Deployed engine test"). Still open: a resume after a real container crash (the guard retest above used a cut client, not a dead container), pass 2 token and cost on the engine, and the same bucket read grant for any other environment.
- Agent Runtime request duration and payload limits for long videos.
- A video longer than a few minutes has not been run. At 1 fps HIGH, input is about 289 tokens per second of video (about 53k for 3 minutes, extrapolated); the model's input limit is 1,048,576 tokens.
- Prompt asks for 20-40 findings, which is a lot for an 8 s clip; tune after the vendor reviews real output.
- Several boxes per finding over time, GIF animation, custom rules in the video branch: out of scope for now.
