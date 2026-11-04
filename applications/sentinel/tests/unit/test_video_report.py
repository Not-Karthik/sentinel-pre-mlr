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

"""Unit tests for video report schemas, bbox/timestamp validation and the node."""

import json

import pytest

from app.agent import finalize_video_report
from app.app_utils.usage import format_usage
from app.app_utils.video_report import (
    finalize_report,
    repair_bbox,
    timestamp_to_seconds,
)
from app.schemas import (
    ContentLocation,
    Severity,
    VideoFinding,
    VideoIssueCategory,
    VideoReport,
)


class MockContext:
    def __init__(self, state: dict | None = None) -> None:
        self.state = state if state is not None else {}


def _finding(
    start: str = "00:03",
    end: str = "00:06",
    severity: Severity = Severity.HIGH,
    location: ContentLocation | None = None,
    finding_id: str = "V-1",
) -> VideoFinding:
    return VideoFinding(
        finding_id=finding_id,
        category=VideoIssueCategory.MISLEADING_CLAIM,
        severity=severity,
        observation="The narration makes an unsubstantiated claim.",
        quoted_content="the future of patient monitoring",
        start_timestamp=start,
        end_timestamp=end,
        location=location,
    )


def test_timestamp_to_seconds() -> None:
    assert timestamp_to_seconds("00:03") == 3
    assert timestamp_to_seconds("01:30") == 90
    assert timestamp_to_seconds("1:02:03") == 3723
    assert timestamp_to_seconds("12") is None
    assert timestamp_to_seconds("aa:bb") is None
    assert timestamp_to_seconds("") is None
    assert timestamp_to_seconds(None) is None


@pytest.mark.parametrize(
    ("bbox", "expected", "rescaled"),
    [
        ([0.1, 0.2, 0.5, 0.6], [0.1, 0.2, 0.5, 0.6], False),
        ([0.5, 0.6, 0.1, 0.2], [0.1, 0.2, 0.5, 0.6], False),
        ([100, 200, 500, 600], [0.1, 0.2, 0.5, 0.6], True),
        ([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], False),
    ],
)
def test_repair_bbox_valid_and_repairable(bbox, expected, rescaled) -> None:
    assert repair_bbox(bbox) == (expected, rescaled)


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        [],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [-0.1, 0.2, 0.5, 0.6],
        [0.1, 0.2, 5000, 0.6],
        [0.2, 0.2, 0.2, 0.6],
        [0.1, 0.2, 0.1001, 0.6],
        [float("nan"), 0.2, 0.5, 0.6],
        ["a", 0.2, 0.5, 0.6],
    ],
)
def test_repair_bbox_unusable(bbox) -> None:
    assert repair_bbox(bbox) == (None, False)


def test_finalize_keeps_valid_location() -> None:
    location = ContentLocation(section="logo", bbox=[0.1, 0.2, 0.5, 0.6])
    report, stats = finalize_report(
        VideoReport(content_summary="A video.", findings=[_finding(location=location)])
    )
    assert report.findings[0].location.bbox == [0.1, 0.2, 0.5, 0.6]
    assert report.findings[0].location.section == "logo"
    assert (stats.bboxes_dropped, stats.bboxes_rescaled, stats.time_ranges_swapped) == (
        0,
        0,
        0,
    )


def test_finalize_drops_bad_bbox_but_keeps_section() -> None:
    location = ContentLocation(section="logo", bbox=[0.3, 0.3, 0.3, 0.3])
    report, stats = finalize_report(
        VideoReport(content_summary="A video.", findings=[_finding(location=location)])
    )
    kept = report.findings[0].location
    assert kept.bbox is None
    assert kept.section == "logo"
    assert stats.bboxes_dropped == 1


def test_finalize_drops_location_with_nothing_left() -> None:
    location = ContentLocation(bbox=[0.3, 0.3, 0.3, 0.3])
    report, _ = finalize_report(
        VideoReport(content_summary="A video.", findings=[_finding(location=location)])
    )
    assert report.findings[0].location is None


def test_finalize_rescales_0_to_1000_boxes() -> None:
    location = ContentLocation(bbox=[100, 200, 500, 600])
    report, stats = finalize_report(
        VideoReport(content_summary="A video.", findings=[_finding(location=location)])
    )
    assert report.findings[0].location.bbox == [0.1, 0.2, 0.5, 0.6]
    assert stats.bboxes_rescaled == 1


def test_finalize_swaps_inverted_time_range() -> None:
    report, stats = finalize_report(
        VideoReport(
            content_summary="A video.", findings=[_finding(start="00:09", end="00:02")]
        )
    )
    finding = report.findings[0]
    assert (finding.start_timestamp, finding.end_timestamp) == ("00:02", "00:09")
    assert stats.time_ranges_swapped == 1


def test_finalize_recomputes_counts_and_keeps_order() -> None:
    report, _ = finalize_report(
        VideoReport(
            content_summary="A video.",
            counts_by_severity={"critical": 99},
            findings=[
                _finding(finding_id="V-1", severity=Severity.HIGH),
                _finding(finding_id="V-2", severity=Severity.LOW),
                _finding(finding_id="V-3", severity=Severity.HIGH),
            ],
        )
    )
    assert report.counts_by_severity == {"high": 2, "low": 1}
    assert [f.finding_id for f in report.findings] == ["V-1", "V-2", "V-3"]


def test_video_finding_requires_a_real_observation() -> None:
    with pytest.raises(ValueError):
        VideoFinding(
            finding_id="V-1",
            category=VideoIssueCategory.OTHER,
            severity=Severity.LOW,
            observation="short",
            start_timestamp="00:00",
            end_timestamp="00:01",
        )


def test_finalize_node_emits_json_and_output() -> None:
    raw = VideoReport(
        content_summary="A video.",
        findings=[
            _finding(
                location=ContentLocation(section="caption", bbox=[0.1, 0.2, 0.5, 0.6])
            )
        ],
    ).model_dump(mode="json")
    event = finalize_video_report(MockContext({"video_report": raw}), None)
    assert event.output["counts_by_severity"] == {"high": 1}
    text = event.content.parts[0].text
    assert json.loads(text) == event.output
    assert event.output["findings"][0]["location"]["bbox"] == [0.1, 0.2, 0.5, 0.6]
    assert "quote" not in event.output["findings"][0]["location"]


def test_finalize_node_reads_node_input_when_state_is_empty() -> None:
    report = VideoReport(content_summary="A video.", findings=[_finding()])
    event = finalize_video_report(MockContext(), report)
    assert event.output["findings"][0]["finding_id"] == "V-1"


def test_finalize_node_reports_invalid_input() -> None:
    event = finalize_video_report(MockContext({"video_report": {"findings": "nope"}}), None)
    assert "error" in event.output
    assert json.loads(event.content.parts[0].text) == event.output


def test_format_usage_includes_modality_details() -> None:
    from google.genai import types

    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=19064,
        candidates_token_count=3006,
        thoughts_token_count=2276,
        total_token_count=24346,
        prompt_tokens_details=[
            types.ModalityTokenCount(modality=types.MediaModality.VIDEO, token_count=16104),
            types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=1526),
        ],
    )
    line = format_usage(usage)
    assert "prompt=19064" in line
    assert "thoughts=2276" in line
    assert "video=16104" in line
    assert "audio=1526" in line
