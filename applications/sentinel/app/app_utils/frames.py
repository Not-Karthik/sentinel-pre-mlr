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

"""Locate the submitted video and pull still frames out of it with ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from google.cloud import storage

from app.app_utils import media

logger = logging.getLogger(__name__)

FRAME_MAX_WIDTH = 1280
FFMPEG_TIMEOUT_SECONDS = 60
DEFAULT_MAX_DOWNLOAD_MB = 500


@dataclass
class VideoSource:
    """Where the submitted video can be read from."""

    gcs_uri: str | None = None
    data: bytes | None = None


def ffmpeg_executable() -> str | None:
    """Return a usable ffmpeg path: one on PATH, else the bundled imageio-ffmpeg."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_video_source(events: Iterable) -> VideoSource | None:
    """Find the newest user-supplied video that this process can read.

    ``gs://`` ``file_data`` and attached ``inline_data`` are readable; YouTube
    and other web URLs are not, so they return None.
    """
    for event in reversed(list(events)):
        if getattr(event, "author", None) != "user" or not event.content:
            continue
        for part in event.content.parts or []:
            if not media.is_video_part(part):
                continue
            if part.inline_data and part.inline_data.data:
                return VideoSource(data=part.inline_data.data)
            uri = part.file_data.file_uri if part.file_data else None
            if uri and uri.startswith("gs://"):
                return VideoSource(gcs_uri=uri)
    return None


def _download_gcs(uri: str, destination: Path, max_bytes: int) -> None:
    bucket_name, _, object_name = uri[len("gs://") :].partition("/")
    client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    blob = client.bucket(bucket_name).get_blob(object_name)
    if blob is None:
        raise FileNotFoundError(f"Object not found: {uri}")
    if blob.size is not None and blob.size > max_bytes:
        raise ValueError(
            f"Video is {blob.size} bytes, above the {max_bytes} byte limit for "
            "bbox refinement"
        )
    blob.download_to_filename(str(destination))


async def materialize(
    source: VideoSource,
    workdir: Path,
    *,
    bucket: str | None,
    folder: str,
    max_bytes: int | None = None,
) -> Path:
    """Write the video to ``workdir`` and return its path.

    Raises:
        ValueError: If a ``gs://`` URI is outside the allowed bucket and folder,
            or the object is larger than the limit.
    """
    limit = max_bytes or int(
        float(os.getenv("VIDEO_REFINE_MAX_MB", DEFAULT_MAX_DOWNLOAD_MB)) * 1024 * 1024
    )
    destination = workdir / "video.bin"
    if source.data is not None:
        if len(source.data) > limit:
            raise ValueError("Attached video is above the size limit for refinement")
        destination.write_bytes(source.data)
        return destination
    assert source.gcs_uri is not None
    media.validate_gcs_uri(source.gcs_uri, bucket, folder)
    await asyncio.to_thread(_download_gcs, source.gcs_uri, destination, limit)
    return destination


async def extract_frame(
    ffmpeg: str,
    video_path: Path,
    seconds: float,
    max_width: int = FRAME_MAX_WIDTH,
) -> bytes | None:
    """Return the JPEG of the frame at ``seconds``, or None if there is none."""
    if seconds < 0:
        return None
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale='min({max_width},iw)':-2",
        "-q:v",
        "3",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=FFMPEG_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logger.warning("ffmpeg timed out extracting frame at %.3fs", seconds)
        return None
    if process.returncode != 0:
        logger.warning(
            "ffmpeg failed at %.3fs: %s", seconds, stderr.decode(errors="replace")[:200]
        )
        return None
    return stdout or None


def make_workdir() -> Path:
    """Create a private temporary directory for downloaded video and frames."""
    return Path(tempfile.mkdtemp(prefix="sentinel_video_"))


def remove_workdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
