# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for frame extraction and the still-frame box refinement."""

import subprocess
from types import SimpleNamespace

import pytest
from google.genai import types

from app.agent import refine_video_boxes
from app.app_utils import frames
from app.app_utils.bbox_refine import (
    FrameBox,
    FrameBoxes,
    build_prompt,
    candidate_times,
    format_timestamp,
    group_findings,
    refine_report,
)
from app.schemas import (
    ContentLocation,
    Severity,
    VideoFinding,
    VideoIssueCategory,
    VideoReport,
)


def _finding(
    finding_id: str,
    start: str,
    end: str = "00:09",
    bbox: list[float] | None = (0.1, 0.1, 0.2, 0.2),
) -> VideoFinding:
    return VideoFinding(
        finding_id=finding_id,
        category=VideoIssueCategory.ACCESSIBILITY,
        severity=Severity.LOW,
        observation="The on-screen disclaimer is too small to read.",
        quoted_content="disclaimer",
        start_timestamp=start,
        end_timestamp=end,
        location=ContentLocation(section="disclaimer", bbox=list(bbox)) if bbox else None,
    )


def _report(*findings: VideoFinding) -> VideoReport:
    return VideoReport(content_summary="A video.", findings=list(findings))


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(2.0) == "00:02"
    assert format_timestamp(2.5) == "00:02.5"
    assert format_timestamp(61.26) == "01:01.3"
    assert format_timestamp(3725) == "01:02:05"
    assert format_timestamp(-1) == "00:00"


def test_candidate_times_skip_negative() -> None:
    assert candidate_times(2.0) == [1.5, 2.0, 2.5, 3.0]
    assert candidate_times(0.2) == [0.2, 0.7, 1.2]


def test_group_findings_only_boxed_and_parsable() -> None:
    report = _report(
        _finding("V-1", "00:02"),
        _finding("V-2", "00:02.0"),
        _finding("V-3", "00:05", bbox=None),
        _finding("V-4", "bad"),
        _finding("V-5", "00:03.5"),
    )
    groups = group_findings(report)
    assert sorted(groups) == [2.0, 3.5]
    assert [f.finding_id for f in groups[2.0]] == ["V-1", "V-2"]


def test_build_prompt_lists_frames_and_findings() -> None:
    finding = _finding("V-1", "00:02")
    prompt = build_prompt([1.5, 2.0], [finding])
    assert "frame 0 at 00:01.5" in prompt
    assert "frame 1 at 00:02" in prompt
    assert "V-1" in prompt
    assert "[ymin, xmin, ymax, xmax]" in prompt


async def _extract(seconds: float) -> bytes | None:
    return b"jpeg-%.1f" % seconds


def _generator(by_id: dict[str, FrameBox], calls: list):
    async def generate(images, prompt):
        calls.append((list(images), prompt))
        return FrameBoxes(boxes=list(by_id.values())), 100, 20

    return generate


@pytest.mark.asyncio
async def test_refine_replaces_box_and_converts_axes() -> None:
    report = _report(_finding("V-1", "00:02"))
    calls: list = []
    gen = _generator(
        {"V-1": FrameBox(finding_id="V-1", frame=1, box_2d=[200, 100, 400, 300])}, calls
    )
    refined, stats = await refine_report(report, extract_frame=_extract, generate_boxes=gen)
    finding = refined.findings[0]
    assert finding.location.bbox == [0.1, 0.2, 0.3, 0.4]
    assert finding.start_timestamp == "00:02"
    assert (stats.refined, stats.shifted, stats.kept_original) == (1, 0, 0)
    assert (stats.calls, stats.frames) == (1, 4)
    assert (stats.prompt_tokens, stats.output_tokens) == (100, 20)
    assert len(calls[0][0]) == 4


@pytest.mark.asyncio
async def test_refine_moves_start_to_the_frame_where_item_is_visible() -> None:
    report = _report(_finding("V-1", "00:02", end="00:03"))
    gen = _generator(
        {"V-1": FrameBox(finding_id="V-1", frame=2, box_2d=[0, 0, 500, 500])}, []
    )
    refined, stats = await refine_report(report, extract_frame=_extract, generate_boxes=gen)
    assert refined.findings[0].start_timestamp == "00:02.5"
    assert refined.findings[0].end_timestamp == "00:03"
    assert stats.shifted == 1


@pytest.mark.asyncio
async def test_refine_extends_end_when_start_moves_past_it() -> None:
    report = _report(_finding("V-1", "00:02", end="00:02"))
    gen = _generator(
        {"V-1": FrameBox(finding_id="V-1", frame=2, box_2d=[0, 0, 500, 500])}, []
    )
    refined, _ = await refine_report(report, extract_frame=_extract, generate_boxes=gen)
    assert refined.findings[0].start_timestamp == "00:02.5"
    assert refined.findings[0].end_timestamp == "00:02.5"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        FrameBox(finding_id="V-1", frame=None, box_2d=None),
        FrameBox(finding_id="V-1", frame=1, box_2d=None),
        FrameBox(finding_id="V-1", frame=1, box_2d=[1, 2, 3]),
        FrameBox(finding_id="V-1", frame=1, box_2d=[500, 500, 500, 500]),
        FrameBox(finding_id="V-1", frame=7, box_2d=[0, 0, 500, 500]),
    ],
)
async def test_refine_keeps_original_box_when_result_is_unusable(result) -> None:
    report = _report(_finding("V-1", "00:02"))
    gen = _generator({"V-1": result}, [])
    refined, stats = await refine_report(report, extract_frame=_extract, generate_boxes=gen)
    assert refined.findings[0].location.bbox == [0.1, 0.1, 0.2, 0.2]
    assert stats.kept_original == 1
    assert stats.refined == 0


@pytest.mark.asyncio
async def test_refine_keeps_original_when_model_call_fails() -> None:
    async def boom(images, prompt):
        raise RuntimeError("429")

    report = _report(_finding("V-1", "00:02"))
    refined, stats = await refine_report(report, extract_frame=_extract, generate_boxes=boom)
    assert refined.findings[0].location.bbox == [0.1, 0.1, 0.2, 0.2]
    assert (stats.calls, stats.kept_original) == (0, 1)


@pytest.mark.asyncio
async def test_refine_skips_group_when_center_frame_is_missing() -> None:
    async def only_neighbours(seconds: float) -> bytes | None:
        return None if seconds == 2.0 else b"x"

    calls: list = []
    report = _report(_finding("V-1", "00:02"))
    refined, stats = await refine_report(
        report, extract_frame=only_neighbours, generate_boxes=_generator({}, calls)
    )
    assert calls == []
    assert refined.findings[0].location.bbox == [0.1, 0.1, 0.2, 0.2]
    assert stats.kept_original == 1


@pytest.mark.asyncio
async def test_refine_one_call_per_start_time() -> None:
    calls: list = []
    report = _report(
        _finding("V-1", "00:02"), _finding("V-2", "00:02"), _finding("V-3", "00:05")
    )
    gen = _generator(
        {
            "V-1": FrameBox(finding_id="V-1", frame=1, box_2d=[0, 0, 500, 500]),
            "V-2": FrameBox(finding_id="V-2", frame=1, box_2d=[100, 100, 600, 600]),
            "V-3": FrameBox(finding_id="V-3", frame=1, box_2d=[0, 0, 300, 300]),
        },
        calls,
    )
    refined, stats = await refine_report(report, extract_frame=_extract, generate_boxes=gen)
    assert stats.calls == 2
    assert stats.refined == 3
    assert [f.location.bbox for f in refined.findings][1] == [0.1, 0.1, 0.6, 0.6]


@pytest.mark.asyncio
async def test_refine_leaves_findings_without_boxes_alone() -> None:
    report = _report(_finding("V-1", "00:02", bbox=None))
    calls: list = []
    refined, stats = await refine_report(
        report, extract_frame=_extract, generate_boxes=_generator({}, calls)
    )
    assert refined == report
    assert calls == [] and stats.calls == 0


def test_find_video_source_prefers_newest_readable_video() -> None:
    def event(author: str, *parts: types.Part):
        return SimpleNamespace(author=author, content=types.Content(parts=list(parts)))

    youtube = types.Part(
        file_data=types.FileData(file_uri="https://youtu.be/abc", mime_type="video/mp4")
    )
    gcs = types.Part(
        file_data=types.FileData(file_uri="gs://b/dev/a.mp4", mime_type="video/mp4")
    )
    inline = types.Part(inline_data=types.Blob(data=b"abc", mime_type="video/mp4"))

    assert frames.find_video_source([event("user", youtube)]) is None
    assert frames.find_video_source([event("user", gcs)]).gcs_uri == "gs://b/dev/a.mp4"
    assert frames.find_video_source([event("user", inline)]).data == b"abc"
    assert frames.find_video_source([event("model", gcs)]) is None
    newest = frames.find_video_source(
        [event("user", gcs), event("model"), event("user", inline)]
    )
    assert newest.data == b"abc"


@pytest.mark.asyncio
async def test_materialize_rejects_uri_outside_allow_list(tmp_path) -> None:
    source = frames.VideoSource(gcs_uri="gs://other/dev/a.mp4")
    with pytest.raises(ValueError):
        await frames.materialize(source, tmp_path, bucket="mine", folder="dev")


@pytest.mark.asyncio
async def test_materialize_writes_attached_video_and_enforces_limit(tmp_path) -> None:
    path = await frames.materialize(
        frames.VideoSource(data=b"123"), tmp_path, bucket=None, folder="dev"
    )
    assert path.read_bytes() == b"123"
    with pytest.raises(ValueError):
        await frames.materialize(
            frames.VideoSource(data=b"123"), tmp_path, bucket=None, folder="dev", max_bytes=2
        )


@pytest.mark.asyncio
async def test_extract_frame_with_real_ffmpeg(tmp_path) -> None:
    ffmpeg = frames.ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error", "-f", "lavfi", "-i",
            "testsrc=duration=2:size=320x240:rate=10", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    frame = await frames.extract_frame(ffmpeg, video, 1.0)
    assert frame is not None and frame[:2] == b"\xff\xd8"
    assert await frames.extract_frame(ffmpeg, video, 30.0) is None
    assert await frames.extract_frame(ffmpeg, video, -1.0) is None


class _State(dict):
    pass


class _Ctx:
    def __init__(self, state: dict, events: list | None = None) -> None:
        self.state = _State(state)
        self.session = SimpleNamespace(events=events or [])


@pytest.mark.asyncio
async def test_refine_node_is_a_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("VIDEO_BBOX_REFINE", "false")
    raw = _report(_finding("V-1", "00:02")).model_dump(mode="json")
    event = await refine_video_boxes(_Ctx({"video_report": raw}), None)
    assert event.output == raw


@pytest.mark.asyncio
async def test_refine_node_is_on_by_default_but_skips_without_a_readable_video(monkeypatch) -> None:
    monkeypatch.delenv("VIDEO_BBOX_REFINE", raising=False)
    raw = _report(_finding("V-1", "00:02")).model_dump(mode="json")
    ctx = _Ctx({"video_report": raw})
    event = await refine_video_boxes(ctx, None)
    assert event.output == raw
    assert ctx.state["video_report"] == raw


@pytest.mark.asyncio
async def test_refine_node_passes_through_when_video_is_not_readable(monkeypatch) -> None:
    monkeypatch.setenv("VIDEO_BBOX_REFINE", "true")
    raw = _report(_finding("V-1", "00:02")).model_dump(mode="json")
    youtube = SimpleNamespace(
        author="user",
        content=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri="https://youtu.be/abc", mime_type="video/mp4"
                    )
                )
            ]
        ),
    )
    ctx = _Ctx({"video_report": raw}, [youtube])
    event = await refine_video_boxes(ctx, None)
    assert event.output == raw
    assert ctx.state["video_report"] == raw


@pytest.mark.asyncio
async def test_refine_node_survives_garbage_input(monkeypatch) -> None:
    monkeypatch.setenv("VIDEO_BBOX_REFINE", "true")
    event = await refine_video_boxes(_Ctx({"video_report": {"findings": "nope"}}), None)
    assert event.output == {"findings": "nope"}
