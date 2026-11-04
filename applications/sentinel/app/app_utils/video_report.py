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

"""Post-processing of the video review: bbox and time-range validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.schemas import VideoFinding, VideoReport

MIN_BOX_SIDE = 0.005


@dataclass
class FinalizeStats:
    """What ``finalize_report`` changed, for logging."""

    bboxes_dropped: int = 0
    bboxes_rescaled: int = 0
    time_ranges_swapped: int = 0


def timestamp_to_seconds(value: str | None) -> float | None:
    """Parse ``MM:SS`` or ``HH:MM:SS`` into seconds; None when malformed."""
    if not value:
        return None
    pieces = value.strip().split(":")
    if not 2 <= len(pieces) <= 3:
        return None
    try:
        numbers = [float(piece) for piece in pieces]
    except ValueError:
        return None
    if any(number < 0 or math.isnan(number) or math.isinf(number) for number in numbers):
        return None
    total = 0.0
    for number in numbers:
        total = total * 60 + number
    return total


def repair_bbox(bbox: Sequence[float] | None) -> tuple[list[float] | None, bool]:
    """Return a valid normalized ``[x_min, y_min, x_max, y_max]`` and whether it
    was rescaled from a 0..1000 scale, or ``(None, False)`` if unusable.

    Values above 1 but within 1000 are treated as a 0..1000 scale and divided by
    1000; inverted coordinates are reordered; boxes thinner than
    ``MIN_BOX_SIDE`` are dropped.
    """
    if bbox is None or len(bbox) != 4:
        return None, False
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None, False
    if any(math.isnan(value) or math.isinf(value) or value < 0 for value in values):
        return None, False

    rescaled = False
    if max(values) > 1:
        if max(values) > 1000:
            return None, False
        values = [value / 1000 for value in values]
        rescaled = True

    x_min, x_max = sorted((values[0], values[2]))
    y_min, y_max = sorted((values[1], values[3]))
    if x_max - x_min < MIN_BOX_SIDE or y_max - y_min < MIN_BOX_SIDE:
        return None, False
    return [round(value, 4) for value in (x_min, y_min, x_max, y_max)], rescaled


def _finalize_finding(finding: VideoFinding, stats: FinalizeStats) -> VideoFinding:
    update: dict = {}

    start = timestamp_to_seconds(finding.start_timestamp)
    end = timestamp_to_seconds(finding.end_timestamp)
    if start is not None and end is not None and end < start:
        update["start_timestamp"] = finding.end_timestamp
        update["end_timestamp"] = finding.start_timestamp
        stats.time_ranges_swapped += 1

    location = finding.location
    if location is not None and location.bbox is not None:
        bbox, rescaled = repair_bbox(location.bbox)
        if bbox is None:
            stats.bboxes_dropped += 1
        if rescaled:
            stats.bboxes_rescaled += 1
        location = location.model_copy(update={"bbox": bbox})
        if not any((location.bbox, location.section, location.quote, location.page)):
            location = None
        update["location"] = location

    return finding.model_copy(update=update) if update else finding


def finalize_report(report: VideoReport) -> tuple[VideoReport, FinalizeStats]:
    """Validate bboxes and time ranges and recompute ``counts_by_severity``."""
    stats = FinalizeStats()
    findings = [_finalize_finding(finding, stats) for finding in report.findings]
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
    return (
        report.model_copy(update={"findings": findings, "counts_by_severity": counts}),
        stats,
    )
