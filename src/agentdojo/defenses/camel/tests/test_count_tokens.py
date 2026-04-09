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

from agentdojo.agent_pipeline.llms.cohere_llm import (
    ChatAssistantMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
)
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import text_content_block_from_string

from camel.count_tokens import (
    get_input_and_output_text_agentdojo,
    get_input_and_output_text_camel,
    get_input_and_output_text_tool_filter,
)
from camel.system_prompt_generator import default_system_prompt_generator

system_prompt = default_system_prompt_generator([])


def test_get_input_and_output_text_agentdojo_no_error():
    function_call = FunctionCall(function="function", args={"arg": 0})
    conversation = [
        ChatSystemMessage(role="system", content=[text_content_block_from_string("system message")]),
        ChatUserMessage(role="user", content=[text_content_block_from_string("query")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("response")], tool_calls=[function_call]
        ),
        ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("tool result")],
            tool_call=function_call,
            tool_call_id=None,
            error=None,
        ),
        ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("final")], tool_calls=None),
    ]
    input, output = get_input_and_output_text_agentdojo(conversation)

    assert input == [
        "<system message><query>",
        f"<system message><query><response | {function_call.function!s} {function_call.args!s}><tool result | {function_call.function!s} {function_call.args!s}>",
    ]
    assert output == [f"<response | {function_call.function!s} {function_call.args!s}>", "<final>"]


def test_get_input_and_output_text_tool_filter():
    function_call = FunctionCall(function="function", args={"arg": 0})
    conversation = [
        ChatSystemMessage(role="system", content=[text_content_block_from_string("system message")]),
        ChatUserMessage(role="user", content=[text_content_block_from_string("query")]),
        ChatUserMessage(role="user", content=[text_content_block_from_string("tool filter query")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("tool filter response")], tool_calls=None
        ),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("response")], tool_calls=[function_call]
        ),
        ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("tool result")],
            tool_call=function_call,
            tool_call_id=None,
            error=None,
        ),
        ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("final")], tool_calls=None),
    ]
    input, output = get_input_and_output_text_tool_filter(conversation)

    assert input == [
        "<system message><query><tool filter query>",
        "<system message><query>",
        f"<system message><query><response | {function_call.function!s} {function_call.args!s}><tool result | {function_call.function!s} {function_call.args!s}>",
    ]
    assert output == [
        "<tool filter response>",
        f"<response | {function_call.function!s} {function_call.args!s}>",
        "<final>",
    ]


def test_get_input_and_output_text_agentdojo_error():
    function_call = FunctionCall(function="function", args={"arg": 0})
    conversation = [
        ChatSystemMessage(role="system", content=[text_content_block_from_string("system message")]),
        ChatUserMessage(role="user", content=[text_content_block_from_string("query")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("response")], tool_calls=[function_call]
        ),
        ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("tool result")],
            tool_call=function_call,
            tool_call_id=None,
            error="error",
        ),
        ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("final")], tool_calls=None),
    ]
    input, output = get_input_and_output_text_agentdojo(conversation)

    assert input == [
        "<system message><query>",
        f"<system message><query><response | {function_call.function!s} {function_call.args!s}><tool result | {function_call.function!s} {function_call.args!s} | error>",
    ]
    assert output == [f"<response | {function_call.function!s} {function_call.args!s}>", "<final>"]


def test_get_input_and_output_text_camel_no_error():
    function_call = FunctionCall(function="query_ai_assistant", args={"query": "ai assistant query"})
    conversation = [
        ChatSystemMessage(role="system", content=[text_content_block_from_string("a")]),
        ChatUserMessage(role="user", content=[text_content_block_from_string("query")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("code")], tool_calls=[function_call]
        ),
        ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("ai assistant result")],
            tool_call=function_call,
            tool_call_id=None,
            error=None,
        ),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("this should be ignored")], tool_calls=None
        ),
    ]
    input, output = get_input_and_output_text_camel(conversation)

    assert input == [f"<{system_prompt}><query>", "<ai assistant query>"]
    assert output == ["<code>", "<ai assistant result>"]


def test_get_input_and_output_text_camel_middle_error():
    function_call = FunctionCall(function="query_ai_assistant", args={"query": "ai assistant query"})
    conversation = [
        ChatSystemMessage(role="system", content=[text_content_block_from_string("a")]),
        ChatUserMessage(role="user", content=[text_content_block_from_string("query")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("code")], tool_calls=[function_call]
        ),
        ChatUserMessage(role="user", content=[text_content_block_from_string("exception")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("fixed code")], tool_calls=[function_call]
        ),
        ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("ai assistant result")],
            tool_call=function_call,
            tool_call_id=None,
            error=None,
        ),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("this should be ignored")], tool_calls=None
        ),
    ]
    input, output = get_input_and_output_text_camel(conversation)

    assert input == [f"<{system_prompt}><query>", f"<{system_prompt}><query><code><exception>", "<ai assistant query>"]
    assert output == ["<code>", "<fixed code>", "<ai assistant result>"]


def test_get_input_and_output_text_camel_final_error():
    function_call = FunctionCall(function="query_ai_assistant", args={"query": "ai assistant query"})
    conversation = [
        ChatSystemMessage(role="system", content=[text_content_block_from_string("a")]),
        ChatUserMessage(role="user", content=[text_content_block_from_string("query")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("code")], tool_calls=[function_call]
        ),
        ChatUserMessage(role="user", content=[text_content_block_from_string("exception")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("fixed code")], tool_calls=[function_call]
        ),
        ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("ai assistant result")],
            tool_call=function_call,
            tool_call_id=None,
            error=None,
        ),
        ChatUserMessage(role="user", content=[text_content_block_from_string("second exception")]),
        ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string("more fixed code")], tool_calls=None
        ),
    ]
    input, output = get_input_and_output_text_camel(conversation)

    assert input == [
        f"<{system_prompt}><query>",
        f"<{system_prompt}><query><code><exception>",
        f"<{system_prompt}><query><code><exception><fixed code><second exception>",
        "<ai assistant query>",
    ]
    assert output == ["<code>", "<fixed code>", "<more fixed code>", "<ai assistant result>"]
