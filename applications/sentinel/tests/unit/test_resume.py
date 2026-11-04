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

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow
from google.genai import types

from app.agent import app as sentinel_app
from app.app_utils.reasoning_engine_adapter import attach_reasoning_engine_routes
from app.app_utils.resilient_client import QueryFailedError, stream_query_with_resume
from app.app_utils.resume import (
    _terminal_node_names,
    find_unfinished_invocation_id,
    resume_stream_query,
    with_auto_resume,
)

USER_ID = "user"
SESSION_ID = "session"


def _build_runner(calls: dict[str, int], fail_first: dict[str, bool]) -> Runner:
    async def step_one(ctx: Context, node_input: Any) -> str:
        calls["step_one"] += 1
        return "one"

    async def step_two(ctx: Context, node_input: Any) -> str:
        calls["step_two"] += 1
        if fail_first["step_two"]:
            if not fail_first.get("always"):
                fail_first["step_two"] = False
            raise RuntimeError("simulated crash")
        return "two"

    async def step_three(ctx: Context, node_input: Any) -> str:
        calls["step_three"] += 1
        return "three"

    workflow = Workflow(
        name="resume_test",
        edges=[(START, step_one), (step_one, step_two), (step_two, step_three)],
    )
    return Runner(
        app=App(root_agent=workflow, name="app"),
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )


async def _drain(agen) -> list[Any]:
    return [event async for event in agen]


async def _as_dicts(agen):
    async for event in agen:
        yield json.loads(event.model_dump_json(exclude_none=True))


def test_sentinel_terminal_nodes_mark_completion() -> None:
    assert _terminal_node_names(sentinel_app.root_agent) == {
        "synthesizer",
        "direct_responder",
        "finalize_video_report",
    }


@pytest.mark.asyncio
async def test_resume_skips_completed_nodes() -> None:
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    runner = _build_runner(calls, {"step_two": True})
    message = types.Content(role="user", parts=[types.Part(text="go")])

    with pytest.raises(Exception):
        await _drain(
            runner.run_async(
                user_id=USER_ID, session_id=SESSION_ID, new_message=message
            )
        )

    assert calls == {"step_one": 1, "step_two": 1, "step_three": 0}
    invocation_id = await find_unfinished_invocation_id(runner, USER_ID, SESSION_ID)
    assert invocation_id is not None

    events = await _drain(
        resume_stream_query(runner, user_id=USER_ID, session_id=SESSION_ID)
    )

    assert events
    assert calls == {"step_one": 1, "step_two": 2, "step_three": 1}
    assert await find_unfinished_invocation_id(runner, USER_ID, SESSION_ID) is None


@pytest.mark.asyncio
async def test_resume_is_noop_when_invocation_completed() -> None:
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    runner = _build_runner(calls, {"step_two": False})
    message = types.Content(role="user", parts=[types.Part(text="go")])

    await _drain(
        runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=message)
    )

    events = await _drain(
        resume_stream_query(runner, user_id=USER_ID, session_id=SESSION_ID)
    )

    assert events == []
    assert calls == {"step_one": 1, "step_two": 1, "step_three": 1}


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )


@pytest.mark.asyncio
async def test_client_resumes_after_error_event() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = json.loads(request.content)["class_method"]
        methods.append(method)
        if method == "async_stream_query":
            return httpx.Response(200, text='{"error_message": "boom"}\n')
        return httpx.Response(200, text='{"id": "done"}\n')

    async with _mock_client(handler) as client:
        events = [
            event
            async for event in stream_query_with_resume(
                client,
                message="hi",
                user_id=USER_ID,
                session_id=SESSION_ID,
                initial_delay=0,
            )
        ]

    assert events == [{"id": "done"}]
    assert methods == ["async_stream_query", "resume_stream_query"]


@pytest.mark.asyncio
async def test_client_gives_up_after_max_resumes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with _mock_client(handler) as client:
        with pytest.raises(QueryFailedError):
            async for _ in stream_query_with_resume(
                client,
                message="hi",
                user_id=USER_ID,
                session_id=SESSION_ID,
                max_resumes=2,
                initial_delay=0,
            ):
                pass


@pytest.mark.asyncio
async def test_resume_without_session_is_noop() -> None:
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    runner = _build_runner(calls, {"step_two": False})

    events = await _drain(
        resume_stream_query(runner, user_id=USER_ID, session_id="missing")
    )

    assert events == []


@pytest.mark.asyncio
async def test_auto_resume_recovers_from_midstream_failure() -> None:
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    runner = _build_runner(calls, {"step_two": True})
    message = types.Content(role="user", parts=[types.Part(text="go")])

    events = await _drain(
        with_auto_resume(
            runner,
            _as_dicts(
                runner.run_async(
                    user_id=USER_ID, session_id=SESSION_ID, new_message=message
                )
            ),
            user_id=USER_ID,
            session_id=SESSION_ID,
            initial_delay=0,
        )
    )

    assert events
    assert calls == {"step_one": 1, "step_two": 2, "step_three": 1}
    assert await find_unfinished_invocation_id(runner, USER_ID, SESSION_ID) is None


@pytest.mark.asyncio
async def test_auto_resume_gives_up_after_max_resumes() -> None:
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    runner = _build_runner(calls, {"step_two": True, "always": True})
    message = types.Content(role="user", parts=[types.Part(text="go")])

    with pytest.raises(Exception):
        await _drain(
            with_auto_resume(
                runner,
                _as_dicts(
                    runner.run_async(
                        user_id=USER_ID, session_id=SESSION_ID, new_message=message
                    )
                ),
                user_id=USER_ID,
                session_id=SESSION_ID,
                max_resumes=2,
                initial_delay=0,
            )
        )

    assert calls["step_one"] == 1
    assert calls["step_two"] == 3


@pytest.mark.asyncio
async def test_auto_resume_reraises_when_nothing_persisted() -> None:
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    runner = _build_runner(calls, {"step_two": False})

    async def failing():
        raise ConnectionError("before anything persisted")
        yield {}

    with pytest.raises(ConnectionError):
        await _drain(
            with_auto_resume(
                runner,
                failing(),
                user_id=USER_ID,
                session_id="never-created",
                initial_delay=0,
            )
        )


def _resume_client(runner: Runner) -> TestClient:
    api = FastAPI()
    api.state.runner = runner
    attach_reasoning_engine_routes(api)
    return TestClient(api)


def _post_resume(client: TestClient) -> list[dict[str, Any]]:
    body = {
        "class_method": "resume_stream_query",
        "input": {"user_id": USER_ID, "session_id": SESSION_ID},
    }
    with client.stream("POST", "/api/stream_reasoning_engine", json=body) as response:
        assert response.status_code == 200
        return [json.loads(raw) for raw in response.iter_lines() if raw.strip()]


@pytest.mark.asyncio
async def test_route_ends_with_resumable_failure_then_retry_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(with_auto_resume.__kwdefaults__, "initial_delay", 0)
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    fail = {"step_two": True, "always": True}
    runner = _build_runner(calls, fail)
    message = types.Content(role="user", parts=[types.Part(text="go")])
    with pytest.raises(Exception):
        await _drain(
            runner.run_async(
                user_id=USER_ID, session_id=SESSION_ID, new_message=message
            )
        )
    client = _resume_client(runner)

    events = _post_resume(client)

    failure = events[-1]
    assert failure["error_code"] == "RuntimeError"
    assert failure["resumable"] is True
    assert failure["invocation_id"]
    assert failure["session_id"] == SESSION_ID
    assert calls["step_three"] == 0

    fail["step_two"] = False
    events = _post_resume(client)

    assert events
    assert not any("error_code" in event for event in events)
    assert calls["step_three"] == 1
    assert await find_unfinished_invocation_id(runner, USER_ID, SESSION_ID) is None


def test_route_resume_requires_session() -> None:
    calls = {"step_one": 0, "step_two": 0, "step_three": 0}
    client = _resume_client(_build_runner(calls, {"step_two": False}))
    response = client.post(
        "/api/stream_reasoning_engine",
        json={"class_method": "resume_stream_query", "input": {"user_id": USER_ID}},
    )
    assert response.status_code == 422
