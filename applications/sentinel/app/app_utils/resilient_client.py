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
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

STREAM_PATH = "/api/stream_reasoning_engine"


class QueryFailedError(RuntimeError):
    """Raised when a query is still failing after the maximum resumes."""


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

    for resume_count in range(max_resumes + 1):
        try:
            async for event in _stream(client, class_method, payload):
                if event.get("error_code") or event.get("error_message"):
                    raise QueryFailedError(
                        event.get("error_message") or event["error_code"]
                    )
                yield event
            return
        except (httpx.HTTPError, QueryFailedError) as exc:
            if resume_count == max_resumes:
                raise QueryFailedError(
                    f"Query failed after {max_resumes} resumes: {exc}"
                ) from exc
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
            class_method = "resume_stream_query"
            payload = resume_payload
