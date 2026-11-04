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

"""Per-call token usage logging, so runs can be priced from real counts."""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

logger = logging.getLogger(__name__)


def format_usage(usage: types.GenerateContentResponseUsageMetadata) -> str:
    """Render a usage record as a single ``key=value`` log line."""
    fields = {
        "prompt": usage.prompt_token_count,
        "candidates": usage.candidates_token_count,
        "thoughts": usage.thoughts_token_count,
        "cached": usage.cached_content_token_count,
        "total": usage.total_token_count,
    }
    details = ",".join(
        f"{str(detail.modality).rsplit('.', 1)[-1].lower()}={detail.token_count}"
        for detail in usage.prompt_tokens_details or []
    )
    line = " ".join(f"{key}={value}" for key, value in fields.items() if value)
    return f"{line} prompt_details[{details}]" if details else line


class UsageLoggingPlugin(BasePlugin):
    """Logs ``usage_metadata`` of every model response with the agent name."""

    def __init__(self) -> None:
        super().__init__(name="usage_logging")

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        usage = llm_response.usage_metadata
        if usage is not None and not llm_response.partial:
            logger.info(
                "model usage agent=%s %s",
                callback_context.agent_name,
                format_usage(usage),
            )
        return None
