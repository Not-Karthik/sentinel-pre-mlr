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

"""Unit tests for media URL normalisation, routing and the video branch wiring."""

import pytest
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, Workflow
from google.genai import types

from app.agent import root_agent, route_media
from app.app_utils import media
from app.app_utils.media import MediaInputPlugin

BUCKET = "media-bucket"
FOLDER = "dev"
YOUTUBE = "https://www.youtube.com/watch?v=ly9epln4fbE"


def _text(value: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=value)])


def _normalize(content: types.Content, fps: float = 1.0) -> types.Content | None:
    return media.normalize_user_message(
        content, fps=fps, bucket=BUCKET, folder=FOLDER
    )


def test_get_video_fps_defaults_and_validates() -> None:
    assert media.get_video_fps("") == 1.0
    assert media.get_video_fps("0.5") == 0.5
    assert media.get_video_fps("2") == 2.0
    assert media.get_video_fps("3") == 1.0
    assert media.get_video_fps("fast") == 1.0


def test_get_media_resolution() -> None:
    assert (
        media.get_media_resolution("high")
        == types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )
    assert media.get_media_resolution("DEFAULT") is None
    assert (
        media.get_media_resolution("bogus")
        == types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )


def test_clamp_fps_picks_nearest_allowed() -> None:
    assert media.clamp_fps(0.1) == 0.5
    assert media.clamp_fps(1.4) == 1.0
    assert media.clamp_fps(10) == 2.0


def test_youtube_url_becomes_video_file_data() -> None:
    result = _normalize(_text(f"Review {YOUTUBE}."), fps=2.0)
    assert result is not None
    assert result.parts[0].text == f"Review {YOUTUBE}."
    video = result.parts[1]
    assert video.file_data.file_uri == YOUTUBE
    assert video.file_data.mime_type == "video/mp4"
    assert video.video_metadata.fps == 2.0
    assert media.has_video(result)


def test_short_youtube_url() -> None:
    result = _normalize(_text("https://youtu.be/ly9epln4fbE"))
    assert result is not None
    assert result.parts[1].file_data.file_uri == "https://youtu.be/ly9epln4fbE"


def test_gcs_video_uri_gets_mime_and_fps() -> None:
    uri = f"gs://{BUCKET}/{FOLDER}/ad.mov"
    result = _normalize(_text(uri))
    assert result is not None
    assert result.parts[1].file_data.mime_type == "video/quicktime"
    assert result.parts[1].video_metadata.fps == 1.0


def test_gcs_image_uri_has_no_video_metadata() -> None:
    uri = f"gs://{BUCKET}/{FOLDER}/banner.png"
    result = _normalize(_text(f"check {uri}"))
    assert result is not None
    image = result.parts[1]
    assert image.file_data.mime_type == "image/png"
    assert image.video_metadata is None
    assert not media.has_video(result)


@pytest.mark.parametrize(
    "uri",
    [
        f"gs://other-bucket/{FOLDER}/ad.mp4",
        f"gs://{BUCKET}/other/ad.mp4",
        f"gs://{BUCKET}/{FOLDER}/../secret/ad.mp4",
        f"gs://{BUCKET}/{FOLDER}//ad.mp4",
    ],
)
def test_gcs_uri_outside_allow_list_is_rejected(uri: str) -> None:
    with pytest.raises(ValueError):
        _normalize(_text(uri))


def test_gcs_uri_without_configured_bucket_is_rejected() -> None:
    with pytest.raises(ValueError, match="GCS_BUCKET_NAME"):
        media.normalize_user_message(
            _text(f"gs://{BUCKET}/{FOLDER}/ad.mp4"), fps=1.0, bucket=None, folder=FOLDER
        )


def test_gcs_uri_with_unknown_extension_is_left_as_text() -> None:
    assert _normalize(_text(f"see gs://{BUCKET}/{FOLDER}/notes.txt")) is None


def test_plain_text_and_images_are_unchanged() -> None:
    assert _normalize(_text("What is fair balance?")) is None
    image = types.Content(
        role="user",
        parts=[
            types.Part(text="Review"),
            types.Part(inline_data=types.Blob(data=b"x", mime_type="image/png")),
        ],
    )
    assert _normalize(image) is None


def test_inline_video_gets_fps() -> None:
    content = types.Content(
        role="user",
        parts=[types.Part(inline_data=types.Blob(data=b"x", mime_type="video/mp4"))],
    )
    result = _normalize(content, fps=0.5)
    assert result is not None
    assert result.parts[0].video_metadata.fps == 0.5
    assert result.parts[0].inline_data.data == b"x"


def test_client_supplied_fps_is_kept_and_clamped() -> None:
    def video_with_fps(fps: float) -> types.Content:
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=f"gs://{BUCKET}/{FOLDER}/ad.mp4",
                        mime_type="video/mp4",
                    ),
                    video_metadata=types.VideoMetadata(fps=fps),
                )
            ],
        )

    assert _normalize(video_with_fps(2.0), fps=1.0) is None
    clamped = _normalize(video_with_fps(7.0), fps=1.0)
    assert clamped is not None
    assert clamped.parts[0].video_metadata.fps == 2.0


def test_video_metadata_offsets_are_preserved() -> None:
    content = types.Content(
        role="user",
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=YOUTUBE, mime_type="video/mp4"),
                video_metadata=types.VideoMetadata(
                    fps=7.0, start_offset="10s", end_offset="40s"
                ),
            )
        ],
    )
    result = _normalize(content)
    assert result is not None
    metadata = result.parts[0].video_metadata
    assert (metadata.fps, metadata.start_offset, metadata.end_offset) == (
        2.0,
        "10s",
        "40s",
    )


def test_too_many_media_items_is_rejected() -> None:
    urls = " ".join(f"gs://{BUCKET}/{FOLDER}/v{i}.mp4" for i in range(11))
    with pytest.raises(ValueError, match="Too many"):
        _normalize(_text(urls))


def test_route_media_routes_by_video_presence() -> None:
    ctx = Context.__new__(Context)
    video = _normalize(_text(YOUTUBE))
    assert route_media(ctx, video).actions.route == "video"
    assert route_media(ctx, _text("hello")).actions.route == "mlr_review"
    assert route_media(ctx, None).actions.route == "mlr_review"


def test_real_graph_wires_the_video_branch() -> None:
    graph = root_agent.graph
    edges = {
        (edge.from_node.name, edge.to_node.name, edge.route) for edge in graph.edges
    }
    assert any(src == "__START__" and dst == "route_media" for src, dst, _ in edges)
    assert ("route_media", "video_reviewer", "video") in edges
    assert ("route_media", "_load_custom_rules", "mlr_review") in edges
    assert any(
        src == "video_reviewer" and dst == "finalize_video_report"
        for src, dst, _ in edges
    )


@pytest.mark.asyncio
async def test_plugin_and_router_deliver_one_video_part_to_the_video_branch() -> None:
    seen: dict[str, object] = {}

    async def mock_mlr(ctx: Context, node_input: object) -> dict:
        seen["mlr"] = True
        return {"branch": "mlr"}

    async def mock_video(ctx: Context, node_input: object) -> dict:
        seen["video_input"] = node_input
        return {"branch": "video"}

    workflow = Workflow(
        name="sim_media",
        edges=[
            (START, route_media),
            (route_media, {"mlr_review": mock_mlr, "video": mock_video}),
        ],
    )
    sim_app = App(
        name="sim_media_app",
        root_agent=workflow,
        plugins=[MediaInputPlugin(fps=2.0, bucket=BUCKET, folder=FOLDER)],
    )
    runner = InMemoryRunner(app=sim_app)
    session = await runner.session_service.create_session(
        app_name="sim_media_app", user_id="u"
    )

    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=_text(f"Review gs://{BUCKET}/{FOLDER}/ad.mp4"),
        )
    ]
    assert events

    assert "mlr" not in seen
    assert "video_input" in seen
    assert seen["video_input"] is None

    stored = await runner.session_service.get_session(
        app_name="sim_media_app", user_id="u", session_id=session.id
    )
    user_events = [e for e in stored.events if e.author == "user"]
    assert len(user_events) == 1
    parts = user_events[0].content.parts
    video_parts = [p for p in parts if p.file_data]
    assert len(video_parts) == 1
    assert video_parts[0].file_data.file_uri == f"gs://{BUCKET}/{FOLDER}/ad.mp4"
    assert video_parts[0].video_metadata.fps == 2.0


@pytest.mark.asyncio
async def test_plain_text_takes_the_existing_branch() -> None:
    seen: dict[str, object] = {}

    async def mock_mlr(ctx: Context, node_input: object) -> dict:
        seen["mlr"] = True
        return {"branch": "mlr"}

    async def mock_video(ctx: Context, node_input: object) -> dict:
        seen["video"] = True
        return {"branch": "video"}

    workflow = Workflow(
        name="sim_media2",
        edges=[
            (START, route_media),
            (route_media, {"mlr_review": mock_mlr, "video": mock_video}),
        ],
    )
    sim_app = App(
        name="sim_media_app2",
        root_agent=workflow,
        plugins=[MediaInputPlugin(fps=1.0, bucket=BUCKET, folder=FOLDER)],
    )
    runner = InMemoryRunner(app=sim_app)
    session = await runner.session_service.create_session(
        app_name="sim_media_app2", user_id="u"
    )
    async for _ in runner.run_async(
        user_id="u", session_id=session.id, new_message=_text("Review this claim")
    ):
        pass
    assert seen == {"mlr": True}
