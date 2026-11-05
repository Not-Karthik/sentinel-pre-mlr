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

"""Second pass that re-boxes video findings on still frames.

The video review boxes items from sampled video frames, which carry far less
detail than a still image. This pass extracts the frames around each finding's
``start_timestamp`` with ffmpeg and asks the model to box the item on those
stills, which is how the image flow works.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from google.genai import types
from pydantic import BaseModel, Field

from app.app_utils.video_report import repair_bbox, timestamp_to_seconds
from app.schemas import VideoFinding, VideoReport

logger = logging.getLogger(__name__)

CANDIDATE_OFFSETS = (-0.5, 0.0, 0.5, 1.0)
MAX_OUTPUT_TOKENS = 8192

ExtractFrame = Callable[[float], Awaitable[bytes | None]]
GenerateBoxes = Callable[
    [list[bytes], str], Awaitable[tuple["FrameBoxes", int, int]]
]


class FrameBox(BaseModel):
    """The model's box for one finding on the provided frames."""

    finding_id: str
    frame: int | None = Field(
        None,
        description=(
            "0-based index of the earliest provided frame in which the item is "
            "clearly visible, or null if it is not visible in any frame."
        ),
    )
    box_2d: list[int] | None = Field(
        None,
        description=(
            "Tight box of the item in that frame as [ymin, xmin, ymax, xmax], "
            "integers from 0 to 1000 (0 = top/left edge, 1000 = bottom/right "
            "edge), or null."
        ),
    )


class FrameBoxes(BaseModel):
    boxes: list[FrameBox] = Field(default_factory=list)


@dataclass
class RefineStats:
    """What the refinement did, for logging."""

    findings: int = 0
    calls: int = 0
    frames: int = 0
    refined: int = 0
    shifted: int = 0
    kept_original: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS`` (``HH:MM:SS`` from one hour), with a tenth if needed."""
    seconds = max(0.0, round(seconds, 1))
    whole = int(seconds)
    tenth = round((seconds - whole) * 10)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    return base if tenth == 0 else f"{base}.{tenth}"


def candidate_times(start: float, offsets: tuple[float, ...] = CANDIDATE_OFFSETS) -> list[float]:
    """Frame times to look at for a finding starting at ``start`` seconds."""
    return [round(start + offset, 3) for offset in offsets if start + offset >= 0]


def group_findings(report: VideoReport) -> dict[float, list[VideoFinding]]:
    """Group findings that have a box by their start time (to 0.1 s)."""
    groups: dict[float, list[VideoFinding]] = {}
    for finding in report.findings:
        if not (finding.location and finding.location.bbox):
            continue
        start = timestamp_to_seconds(finding.start_timestamp)
        if start is None:
            continue
        groups.setdefault(round(start, 1), []).append(finding)
    return groups


def build_prompt(times: list[float], findings: list[VideoFinding]) -> str:
    frames = ", ".join(f"frame {i} at {format_timestamp(t)}" for i, t in enumerate(times))
    lines = "\n".join(
        f"- {f.finding_id}: region: {(f.location.section if f.location else None) or 'unspecified'}"
        f" | quoted: {f.quoted_content or 'n/a'} | issue: {f.observation[:200]}"
        for f in findings
    )
    return (
        f"These are consecutive still frames from a promotional video ({frames}). "
        "For each finding below, locate the specific on-screen item it refers to. "
        "Return `frame`, the index of the earliest frame in which that item is clearly "
        "visible, and `box_2d`, the tight bounding box of the item in that frame as "
        "[ymin, xmin, ymax, xmax] with integers from 0 to 1000 (0 is the top/left edge "
        "of the frame, 1000 the bottom/right edge). Box only the item itself, not the "
        "whole frame, unless the finding concerns the whole frame. If the item is not "
        "visible in any of the frames, return null for both fields.\n\n"
        f"Findings:\n{lines}"
    )


def _to_xyxy(box_2d: list[int] | None) -> list[float] | None:
    if box_2d is None or len(box_2d) != 4:
        return None
    ymin, xmin, ymax, xmax = (min(max(value, 0), 1000) / 1000 for value in box_2d)
    repaired, _ = repair_bbox([xmin, ymin, xmax, ymax])
    return repaired


def _apply(
    finding: VideoFinding,
    result: FrameBox | None,
    times: list[float],
    stats: RefineStats,
) -> VideoFinding:
    if result is None or result.frame is None or not 0 <= result.frame < len(times):
        stats.kept_original += 1
        return finding
    bbox = _to_xyxy(result.box_2d)
    if bbox is None:
        stats.kept_original += 1
        return finding

    update: dict = {"location": finding.location.model_copy(update={"bbox": bbox})}
    original_start = timestamp_to_seconds(finding.start_timestamp)
    new_start = times[result.frame]
    if original_start is None or abs(new_start - original_start) > 0.01:
        update["start_timestamp"] = format_timestamp(new_start)
        end = timestamp_to_seconds(finding.end_timestamp)
        if end is not None and end < new_start:
            update["end_timestamp"] = format_timestamp(new_start)
        stats.shifted += 1
    stats.refined += 1
    return finding.model_copy(update=update)


async def refine_report(
    report: VideoReport,
    *,
    extract_frame: ExtractFrame,
    generate_boxes: GenerateBoxes,
    concurrency: int = 3,
) -> tuple[VideoReport, RefineStats]:
    """Re-box every finding that has a box using still frames.

    A finding keeps its original box when its frames cannot be extracted, the
    model call fails, or the model reports the item as not visible.
    """
    stats = RefineStats(findings=len(report.findings))
    groups = group_findings(report)
    if not groups:
        return report, stats

    wanted = sorted({t for start in groups for t in candidate_times(start)})
    extract_gate = asyncio.Semaphore(4)

    async def extract(seconds: float) -> tuple[float, bytes | None]:
        async with extract_gate:
            return seconds, await extract_frame(seconds)

    frames = dict(await asyncio.gather(*(extract(t) for t in wanted)))
    stats.frames = sum(1 for data in frames.values() if data)

    gate = asyncio.Semaphore(max(1, concurrency))
    results: dict[str, FrameBox] = {}
    group_times: dict[float, list[float]] = {}

    async def run_group(start: float, findings: list[VideoFinding]) -> None:
        times = [t for t in candidate_times(start) if frames.get(t)]
        group_times[start] = times
        if round(start, 3) not in times:
            return
        images = [frames[t] for t in times]
        async with gate:
            try:
                boxes, prompt_tokens, output_tokens = await generate_boxes(
                    images, build_prompt(times, findings)
                )
            except Exception as exc:
                logger.warning("Box refinement failed for frames at %.1fs: %s", start, exc)
                return
        stats.calls += 1
        stats.prompt_tokens += prompt_tokens
        stats.output_tokens += output_tokens
        for box in boxes.boxes:
            results[box.finding_id] = box

    await asyncio.gather(*(run_group(s, fs) for s, fs in groups.items()))

    updated: list[VideoFinding] = []
    for finding in report.findings:
        start = timestamp_to_seconds(finding.start_timestamp)
        key = round(start, 1) if start is not None else None
        if key in groups and finding in groups[key]:
            finding = _apply(
                finding, results.get(finding.finding_id), group_times.get(key, []), stats
            )
        updated.append(finding)
    return report.model_copy(update={"findings": updated}), stats


def make_generate_boxes(client, model: str) -> GenerateBoxes:
    """Build the model call used by ``refine_report`` from a ``genai.Client``."""

    async def generate(images: list[bytes], prompt: str):
        parts: list[types.Part] = []
        for index, image in enumerate(images):
            parts.append(types.Part(text=f"Frame {index}:"))
            parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
        parts.append(types.Part(text=prompt))
        response = await client.aio.models.generate_content(
            model=model,
            contents=types.Content(role="user", parts=parts),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FrameBoxes,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
            ),
        )
        usage = response.usage_metadata
        prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (
            ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
            if usage
            else 0
        )
        return FrameBoxes.model_validate_json(response.text), prompt_tokens, output_tokens

    return generate
