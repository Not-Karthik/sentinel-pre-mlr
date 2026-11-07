# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client that resumes an interrupted query instead of starting it over.

Sends the query to ``/api/stream_reasoning_engine``. If the stream drops or
carries an error event, it calls ``resume_stream_query`` for the same session
with capped exponential backoff, so only the unfinished part of the pipeline
re-runs.

The server may keep running a query after the client's stream drops. A resume
that finds the invocation still active answers ``invocation_in_progress``; the
client then keeps polling instead of starting a second run. When the resume
finds the invocation already finished it returns nothing, so the client reads
the stored events of that invocation from the session. An error event marked
``resumable`` means the server's own run is dead, so the resume is forced.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

STREAM_PATH = "/api/stream_reasoning_engine"
QUERY_PATH = "/api/reasoning_engine"
IN_PROGRESS_STATUS = "invocation_in_progress"
POLL_INTERVAL = 15.0


class QueryFailedError(RuntimeError):
    """Raised when a query is still failing after the maximum resumes."""

    def __init__(self, message: str, *, resumable: bool = False) -> None:
        super().__init__(message)
        self.resumable = resumable


class StillRunningError(QueryFailedError):
    """Raised when the invocation is still running after the maximum wait."""


async def _session_events(
    client: httpx.AsyncClient, user_id: str, session_id: str
) -> list[dict[str, Any]]:
    response = await client.post(
        QUERY_PATH,
        json={
            "class_method": "async_get_session",
            "input": {"user_id": user_id, "session_id": session_id},
        },
    )
    response.raise_for_status()
    session = (response.json() or {}).get("output") or {}
    events = session.get("events") or []
    if not events:
        return []
    latest = events[-1].get("invocation_id")
    return [event for event in events if event.get("invocation_id") == latest]


async def _stream(
    client: httpx.AsyncClient, class_method: str, payload: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    body = {"class_method": class_method, "input": payload}
    async with client.stream("POST", STREAM_PATH, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.strip():
                yield json.loads(line)


async def stream_query_with_resume(
    client: httpx.AsyncClient,
    *,
    message: str,
    user_id: str,
    session_id: str,
    max_resumes: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    poll_interval: float = POLL_INTERVAL,
    max_wait: float = 1500.0,
) -> AsyncIterator[dict[str, Any]]:
    """Yield events for a query, resuming from the session after a failure."""
    class_method = "async_stream_query"
    payload: dict[str, Any] = {
        "message": message,
        "user_id": user_id,
        "session_id": session_id,
    }
    resume_payload = {"user_id": user_id, "session_id": session_id}
    delay = initial_delay
    seen: set[str] = set()
    resume_count = 0
    waited = 0.0

    while True:
        try:
            in_progress: dict[str, Any] | None = None
            received = False
            async for event in _stream(client, class_method, payload):
                if event.get("error_code") or event.get("error_message"):
                    raise QueryFailedError(
                        event.get("error_message") or event["error_code"],
                        resumable=bool(event.get("resumable")),
                    )
                if event.get("status") == IN_PROGRESS_STATUS:
                    in_progress = event
                    continue
                received = True
                if event.get("id"):
                    seen.add(event["id"])
                yield event
            if in_progress is not None and not received:
                pause = min(poll_interval, float(in_progress["retry_after_seconds"]))
                if waited + pause > max_wait:
                    raise StillRunningError(
                        f"Invocation is still running after {max_wait:.0f}s of waiting"
                    )
                await asyncio.sleep(pause)
                waited += pause
                continue
            if class_method == "resume_stream_query" and not received:
                for event in await _session_events(client, user_id, session_id):
                    if event.get("id") not in seen:
                        yield event
            return
        except (httpx.HTTPError, QueryFailedError) as exc:
            if isinstance(exc, StillRunningError):
                raise
            if resume_count == max_resumes:
                raise QueryFailedError(
                    f"Query failed after {max_resumes} resumes: {exc}"
                ) from exc
            resume_count += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
            class_method = "resume_stream_query"
            payload = dict(resume_payload)
            if isinstance(exc, QueryFailedError) and exc.resumable:
                payload["force"] = True
