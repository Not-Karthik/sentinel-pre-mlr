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

"""Media input handling for the Sentinel agent.

Turns YouTube and ``gs://`` URLs typed into a user message into ``file_data``
parts (with the configured sampling rate for video) before the workflow runs,
and detects video input for routing.
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
import urllib.parse
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

logger = logging.getLogger(__name__)

ALLOWED_FPS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_FPS = 4.0
MAX_MEDIA_PER_MESSAGE = 10

VIDEO_MIME_BY_EXT = {
    "mp4": "video/mp4",
    "mpeg": "video/mpeg",
    "mpg": "video/mpg",
    "mov": "video/quicktime",
    "avi": "video/avi",
    "wmv": "video/wmv",
    "webm": "video/webm",
}
IMAGE_MIME_BY_EXT = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}

_YOUTUBE_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/watch\?\S+|youtu\.be/\S+)",
    re.IGNORECASE,
)
_GCS_RE = re.compile(r"gs://[^\s\"'<>]+")
_TRAILING_PUNCTUATION = ".,;:!?)]}"


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable (1/true/yes/on), with a default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def get_video_fps(raw: str | None = None) -> float:
    """Return the configured video sampling rate (``VIDEO_FPS``), default 4."""
    raw = os.getenv("VIDEO_FPS") if raw is None else raw
    if raw is None or not raw.strip():
        return DEFAULT_FPS
    try:
        value = float(raw)
    except ValueError:
        value = None
    if value in ALLOWED_FPS:
        return value
    logger.warning(
        "Ignoring VIDEO_FPS=%r; allowed values are %s", raw, list(ALLOWED_FPS)
    )
    return DEFAULT_FPS


def get_media_resolution(raw: str | None = None) -> types.MediaResolution | None:
    """Return the configured media resolution (``VIDEO_MEDIA_RESOLUTION``).

    ``HIGH`` (the default) maps to ``MEDIA_RESOLUTION_HIGH``; ``DEFAULT`` leaves
    the model's own default in place (returns None).
    """
    raw = os.getenv("VIDEO_MEDIA_RESOLUTION") if raw is None else raw
    value = (raw or "HIGH").strip().upper()
    if value == "HIGH":
        return types.MediaResolution.MEDIA_RESOLUTION_HIGH
    if value == "DEFAULT":
        return None
    logger.warning(
        "Ignoring VIDEO_MEDIA_RESOLUTION=%r; allowed values are HIGH, DEFAULT", raw
    )
    return types.MediaResolution.MEDIA_RESOLUTION_HIGH


def clamp_fps(value: float) -> float:
    """Return the allowed sampling rate closest to ``value``."""
    return min(ALLOWED_FPS, key=lambda allowed: abs(allowed - value))


def validate_gcs_uri(uri: str, bucket: str | None, folder: str) -> str:
    """Validate that ``uri`` is inside the configured media bucket and folder.

    Raises:
        ValueError: If the bucket is not configured, the URI contains unsafe
            characters or path traversal, or it points outside
            ``gs://{bucket}/{folder}/``.
    """
    if not bucket:
        raise ValueError("Unauthorized GCS URI: GCS_BUCKET_NAME is not configured")

    unquoted = urllib.parse.unquote(uri)
    if "\x00" in unquoted or "\\" in unquoted:
        raise ValueError(f"Invalid media URI '{uri}': disallowed characters detected")
    if ".." in uri.split("/") or ".." in unquoted.split("/"):
        raise ValueError(f"Invalid media URI '{uri}': path traversal is not allowed")

    folder = folder.strip("/")
    expected_prefix = f"gs://{bucket}/{folder}/" if folder else f"gs://{bucket}/"
    if not uri.startswith(expected_prefix) or not unquoted.startswith(
        expected_prefix
    ):
        raise ValueError(
            f"Unauthorized GCS URI '{uri}': must start with '{expected_prefix}'"
        )

    object_path = unquoted[len(expected_prefix) :]
    if not object_path or object_path.startswith("/"):
        raise ValueError(
            f"Unauthorized GCS URI '{uri}': missing object path under "
            f"'{expected_prefix}'"
        )

    bucket_path = unquoted[len(f"gs://{bucket}/") :]
    normalized = posixpath.normpath(bucket_path)
    folder_prefix = f"{folder}/" if folder else ""
    if folder_prefix and not normalized.startswith(folder_prefix):
        raise ValueError(
            f"Unauthorized GCS URI '{uri}': path escapes authorized folder '{folder}'"
        )
    return uri


def _blob(part: types.Part) -> types.FileData | types.Blob | None:
    return part.file_data or part.inline_data


def is_video_part(part: types.Part) -> bool:
    """True when ``part`` carries video via ``file_data`` or ``inline_data``."""
    blob = _blob(part)
    mime_type = getattr(blob, "mime_type", None) if blob else None
    return bool(mime_type and mime_type.lower().startswith("video/"))


def has_video(content: types.Content | None) -> bool:
    """True when any part of ``content`` is a video."""
    if content is None:
        return False
    return any(is_video_part(part) for part in content.parts or [])


def sampling_fps(events) -> float | None:
    """Return the sampling rate set on the newest user-supplied video, if any."""
    for event in reversed(list(events)):
        if getattr(event, "author", None) != "user" or not event.content:
            continue
        for part in event.content.parts or []:
            if is_video_part(part) and part.video_metadata and part.video_metadata.fps:
                return float(part.video_metadata.fps)
    return None


def sampling_note(fps: float) -> str:
    """Tell the model how far apart the sampled frames are."""
    step = 1 / fps
    frames = "frame" if fps == 1 else "frames"
    seconds = "second" if step == 1 else "seconds"
    return (
        "## Sampling\n\n"
        f"The video was sampled at {fps:g} {frames} per second, so the frames you "
        f"can see are {step:g} {seconds} apart. Report timestamps that match a "
        f"sampled frame (a multiple of {step:g} {seconds} from the start of the "
        "video); when an item first becomes clearly visible between two sampled "
        "frames, use the later frame."
    )


def _strip_trailing_punctuation(url: str) -> str:
    return url.rstrip(_TRAILING_PUNCTUATION)


def _extension(uri: str) -> str:
    path = urllib.parse.urlparse(uri).path
    return posixpath.splitext(path)[1].lstrip(".").lower()


def _with_fps(part: types.Part, default_fps: float) -> types.Part:
    existing = part.video_metadata
    fps = clamp_fps(existing.fps) if existing and existing.fps else default_fps
    metadata = (
        existing.model_copy(update={"fps": fps})
        if existing
        else types.VideoMetadata(fps=fps)
    )
    return part.model_copy(update={"video_metadata": metadata})


def _media_parts_from_text(
    text: str, fps: float, bucket: str | None, folder: str
) -> list[types.Part]:
    found: list[tuple[int, str]] = []
    for match in _YOUTUBE_RE.finditer(text):
        url = _strip_trailing_punctuation(match.group(0))
        if "v=" in url or "youtu.be/" in url.lower():
            found.append((match.start(), url))
    for match in _GCS_RE.finditer(text):
        found.append((match.start(), _strip_trailing_punctuation(match.group(0))))

    parts: list[types.Part] = []
    seen: set[str] = set()
    for _, url in sorted(found):
        if url in seen:
            continue
        seen.add(url)
        if url.startswith("gs://"):
            ext = _extension(url)
            if ext in VIDEO_MIME_BY_EXT:
                mime_type = VIDEO_MIME_BY_EXT[ext]
            elif ext in IMAGE_MIME_BY_EXT:
                mime_type = IMAGE_MIME_BY_EXT[ext]
            else:
                logger.debug("Leaving gs:// URI with unknown extension as text: %s", url)
                continue
            validate_gcs_uri(url, bucket, folder)
        else:
            mime_type = "video/mp4"
        part = types.Part(file_data=types.FileData(file_uri=url, mime_type=mime_type))
        if mime_type.startswith("video/"):
            part = _with_fps(part, fps)
        parts.append(part)
    return parts


def normalize_user_message(
    content: types.Content,
    *,
    fps: float,
    bucket: str | None,
    folder: str,
) -> types.Content | None:
    """Return a rewritten message, or None when nothing needs to change.

    Text parts keep their text; YouTube and ``gs://`` media URLs found in them
    are appended as ``file_data`` parts. Video parts (attached or derived) get
    ``video_metadata.fps``, keeping a client-supplied value clamped to the
    allowed set.

    Raises:
        ValueError: If a ``gs://`` media URI is outside the allowed bucket and
            folder, or the message references too many media items.
    """
    changed = False
    parts: list[types.Part] = []
    media_count = 0
    for part in content.parts or []:
        if part.text:
            parts.append(part)
            media_parts = _media_parts_from_text(part.text, fps, bucket, folder)
            parts.extend(media_parts)
            media_count += len(media_parts)
            changed = changed or bool(media_parts)
        elif is_video_part(part):
            updated = _with_fps(part, fps)
            changed = changed or updated != part
            parts.append(updated)
            media_count += 1
        else:
            parts.append(part)

    if media_count > MAX_MEDIA_PER_MESSAGE:
        raise ValueError(
            f"Too many media items in one message ({media_count}); "
            f"the limit is {MAX_MEDIA_PER_MESSAGE}"
        )
    if not changed:
        return None
    return types.Content(role=content.role or "user", parts=parts)


class MediaInputPlugin(BasePlugin):
    """Rewrites the user message so media URLs reach the model as media parts."""

    def __init__(
        self,
        fps: float | None = None,
        bucket: str | None = None,
        folder: str | None = None,
    ) -> None:
        super().__init__(name="media_input")
        self._fps = fps
        self._bucket = bucket
        self._folder = folder

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        return normalize_user_message(
            user_message,
            fps=self._fps if self._fps is not None else get_video_fps(),
            bucket=self._bucket
            if self._bucket is not None
            else os.getenv("GCS_BUCKET_NAME"),
            folder=self._folder
            if self._folder is not None
            else os.getenv("GCS_MEDIA_FOLDER", "dev"),
        )
