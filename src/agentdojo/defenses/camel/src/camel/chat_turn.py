# Copyright 2025 Google LLC
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

import dataclasses
from collections.abc import Sequence

from agentdojo.agent_pipeline.llms.cohere_llm import ChatUserMessage
from agentdojo.types import ChatMessage, ChatToolResultMessage, get_text_content_as_str
from typing_extensions import Self


@dataclasses.dataclass
class Turn:
    code: str
    tool_results: list[ChatToolResultMessage]
    has_error: bool
    error_message: ChatUserMessage | None

    @classmethod
    def from_list(cls, turn_list: list[ChatMessage]) -> Self:
        code = next(msg for msg in turn_list if msg["role"] == "assistant")["content"]
        assert code is not None
        error_message = next((msg for msg in turn_list if msg["role"] == "user"), None)
        tool_results = [msg for msg in turn_list if msg["role"] == "tool"]
        return cls(get_text_content_as_str(code), tool_results, error_message is not None, error_message)


def make_turns(messages: Sequence[ChatMessage]) -> list[Turn]:
    turns_lists: list[list[ChatMessage]] = [[]]
    for message in messages:
        # Append message to the latest turn
        turns_lists[-1].append(message)
        if message["role"] == "user":
            # Create new turn if the last message was an error (i.e., user message)
            turns_lists.append([])

    # Remove last message if from assistant and is empty (it means the last code sent was an error)
    if turns_lists[-1] and turns_lists[-1][-1]["role"] == "assistant":
        if turns_lists[-1][-1]["content"] is None:
            turns_lists.pop()

    turns = map(Turn.from_list, turns_lists)
    return list(turns)
