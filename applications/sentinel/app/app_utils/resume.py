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

"""Resume an interrupted invocation from the events persisted in the session.

An invocation is finished once one of the workflow's terminal nodes has written
its output (or the root emitted ``end_of_agent``, for a resumable app). An
invocation without either was interrupted, and
``Runner.run_async(invocation_id=...)`` picks it back up while the workflow
replays the nodes whose outputs are already in the session.
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from google.adk.runners import Runner

logger = logging.getLogger(__name__)

AUTO_RESUME_MAX = int(os.getenv("AUTO_RESUME_MAX", "3"))
AUTO_RESUME_INITIAL_DELAY = float(os.getenv("AUTO_RESUME_INITIAL_DELAY", "2.0"))
AUTO_RESUME_MAX_DELAY = float(os.getenv("AUTO_RESUME_MAX_DELAY", "60.0"))


def _terminal_node_names(agent: Any) -> set[str]:
    graph = getattr(agent, "graph", None)
    return set(getattr(graph, "_terminal_node_names", None) or ())


def _node_name(path: str | None) -> str | None:
    if not path:
        return None
    return path.rsplit("/", 1)[-1].split("@")[0]


async def find_unfinished_invocation_id(
    runner: Runner, user_id: str, session_id: str
) -> str | None:
    """Return the latest invocation id with no end-of-agent marker, if any."""
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    if session is None:
        return None

    root_name = runner.agent.name
    terminal_names = _terminal_node_names(runner.agent)
    order: list[str] = []
    finished: set[str] = set()
    for event in session.events:
        invocation_id = event.invocation_id
        if not invocation_id:
            continue
        if invocation_id not in order:
            order.append(invocation_id)
        if (
            event.author == root_name
            and event.actions
            and event.actions.end_of_agent
        ) or (
            event.output is not None
            and _node_name(getattr(event.node_info, "path", None)) in terminal_names
        ):
            finished.add(invocation_id)

    if not order or order[-1] in finished:
        return None
    return order[-1]


async def resume_stream_query(
    runner: Runner,
    *,
    user_id: str,
    session_id: str,
    invocation_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream the remaining events of an interrupted invocation.

    Yields nothing when the latest invocation already completed.
    """
    if invocation_id is None:
        invocation_id = await find_unfinished_invocation_id(
            runner, user_id, session_id
        )
    if invocation_id is None:
        return

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, invocation_id=invocation_id
    ):
        yield json.loads(event.model_dump_json(exclude_none=True))


async def with_auto_resume(
    runner: Runner,
    events: AsyncIterator[dict[str, Any]],
    *,
    user_id: str,
    session_id: str,
    max_resumes: int = AUTO_RESUME_MAX,
    initial_delay: float = AUTO_RESUME_INITIAL_DELAY,
    max_delay: float = AUTO_RESUME_MAX_DELAY,
) -> AsyncIterator[dict[str, Any]]:
    """Yield ``events``; if it fails midway, resume the session in place.

    Re-raises the original error when the resume budget is spent or when no
    unfinished invocation is persisted to resume from.
    """
    resumes = 0
    delay = initial_delay
    while True:
        try:
            async for event in events:
                yield event
            return
        except Exception:
            try:
                invocation_id = await find_unfinished_invocation_id(
                    runner, user_id, session_id
                )
            except Exception:
                invocation_id = None
            if resumes >= max_resumes or invocation_id is None:
                raise
            resumes += 1
            logger.warning(
                "Query failed; resuming invocation %s of session %s (%d/%d)",
                invocation_id,
                session_id,
                resumes,
                max_resumes,
                exc_info=True,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
            events = resume_stream_query(
                runner,
                user_id=user_id,
                session_id=session_id,
                invocation_id=invocation_id,
            )


async def failure_event(
    runner: Runner, user_id: str, session_id: str, exc: Exception
) -> dict[str, Any]:
    """Terminal stream event for a query that failed after automatic resumes.

    ``resumable`` tells the caller whether a ``resume_stream_query`` retry can
    continue from the saved session, or the original query must be re-sent.
    """
    try:
        invocation_id = await find_unfinished_invocation_id(
            runner, user_id, session_id
        )
    except Exception:
        invocation_id = None
    return {
        "error_code": type(exc).__name__,
        "error_message": str(exc) or type(exc).__name__,
        "resumable": invocation_id is not None,
        "invocation_id": invocation_id,
        "session_id": session_id,
        "user_id": user_id,
    }
