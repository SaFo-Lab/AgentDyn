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

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias, TypedDict

import tiktoken
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    FunctionCall,
    get_text_content_as_str,
)

from camel.system_prompt_generator import default_system_prompt_generator

FunctionCall.model_rebuild()

from agentdojo.benchmark import TaskResults

TaskResults.model_rebuild()

from camel.chat_turn import make_turns as make_camel_turns

TokensChatMessage: TypeAlias = ChatAssistantMessage | ChatToolResultMessage | ChatUserMessage


def make_agentdojo_turns(messages: Sequence[ChatMessage]) -> list[list[ChatMessage]]:
    turns = [[]]
    for message in messages:
        if message["role"] == "assistant":
            turns.append([])
        turns[-1].append(message)

    if turns[-1][-1]["role"] == "assistant" and len(turns[-1]) == 1:
        # exclude last turn containing only the last assistant message
        turns = turns[:-1]

    return turns


def wrap_message(message: str) -> str:
    return f"<{message}>"


def transform_turn(messages: Sequence[ChatMessage]) -> str:
    turn_content = ""
    for message in messages:
        if message["role"] == "system":
            turn_content += wrap_message(get_text_content_as_str(message["content"]))
        if message["role"] == "user":
            turn_content += wrap_message(extract_content_from_user_message(message))
        if message["role"] == "assistant":
            turn_content += wrap_message(extract_content_from_assistant_message(message))
        if message["role"] == "tool":
            turn_content += wrap_message(extract_content_from_tool_result_message(message))
    return turn_content


def extract_content_from_tool_call(tool_call: FunctionCall):
    return f"{tool_call.function} {dict(tool_call.args)!s}"


def extract_content_from_assistant_message(message: ChatAssistantMessage) -> str:
    content = ""

    if message["content"] is not None:
        content += get_text_content_as_str(message["content"])

    if message["tool_calls"] is not None:
        for tool_call in message["tool_calls"]:
            content += f" | {extract_content_from_tool_call(tool_call)}"

    return content


def extract_content_from_tool_result_message(message: ChatToolResultMessage) -> str:
    error_part = f" | {message['error']}" if message["error"] is not None else ""
    return f"{get_text_content_as_str(message['content'])} | {extract_content_from_tool_call(message['tool_call'])}{error_part}"


def extract_content_from_user_message(message: ChatUserMessage) -> str:
    return get_text_content_as_str(message["content"])


def get_input_and_output_text_agentdojo(messages: Sequence[ChatMessage]) -> tuple[list[str], list[str]]:
    output_messages = [
        wrap_message(extract_content_from_assistant_message(message))
        for message in messages
        if message["role"] == "assistant"
    ]
    turns = make_agentdojo_turns(messages)
    turns_as_strings = [transform_turn(turn) for turn in turns]
    input_content_list = [turns_as_strings[0]]
    for turn_as_string in turns_as_strings[1:]:
        turn_overall_content = input_content_list[-1] + turn_as_string
        input_content_list.append(turn_overall_content)
    return input_content_list, output_messages


def get_input_and_output_text_tool_filter(messages: Sequence[ChatMessage]) -> tuple[list[str], list[str]]:
    # the shape of tool filter conversations is
    # system message
    # user query
    # tool filter query
    # tool filter response
    # rest of conversation
    output_messages = [
        wrap_message(extract_content_from_assistant_message(message))
        for message in messages
        if message["role"] == "assistant"
    ]
    tool_filter_input = "".join(
        [wrap_message(get_text_content_as_str(message["content"] or [])) for message in messages[:3]]
    )
    agent_messages = [*messages[:2], *messages[4:]]
    turns = make_agentdojo_turns(agent_messages)
    turns_as_strings = [transform_turn(turn) for turn in turns]
    input_content_list = [tool_filter_input, turns_as_strings[0]]
    for turn_as_string in turns_as_strings[1:]:
        turn_overall_content = input_content_list[-1] + turn_as_string
        input_content_list.append(turn_overall_content)
    return input_content_list, output_messages


def get_q_llm_input_outputs(messages: list[ChatToolResultMessage]) -> tuple[list[str], list[str]]:
    ai_assistant_calls_messages = [
        message for message in messages if message["tool_call"].function == "query_ai_assistant"
    ]
    q_llm_inputs = [wrap_message(str(call["tool_call"].args["query"])) for call in ai_assistant_calls_messages]
    q_llm_outputs = [wrap_message(get_text_content_as_str(call["content"])) for call in ai_assistant_calls_messages]

    return q_llm_inputs, q_llm_outputs


def get_input_and_output_text_camel_separate(
    messages: Sequence[ChatMessage],
) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
    messages = [message for message in messages if message["role"] != "system"]
    turns = make_camel_turns(messages[1:])

    p_llm_output_content = []
    q_llm_input_content = []
    q_llm_output_content = []
    for turn in turns:
        turn_q_llm_inputs, turn_q_llm_outputs = get_q_llm_input_outputs(turn.tool_results)
        q_llm_input_content += turn_q_llm_inputs
        q_llm_output_content += turn_q_llm_outputs
        p_llm_output_content.append(wrap_message(turn.code))

    # Do not include any function to keep it fair with not counting tools transformation when not using camel
    system_prompt = wrap_message(default_system_prompt_generator([]))
    p_llm_input_content = [system_prompt + wrap_message(get_text_content_as_str(messages[0]["content"] or []))]
    for turn in turns[:-1]:
        turn_overall_content = p_llm_input_content[-1] + wrap_message(turn.code)
        if turn.error_message is not None:
            turn_overall_content += wrap_message(get_text_content_as_str(turn.error_message["content"] or []))
        p_llm_input_content.append(turn_overall_content)

    return (p_llm_input_content, p_llm_output_content), (q_llm_input_content, q_llm_output_content)


def get_input_and_output_text_camel(messages: Sequence[ChatMessage]) -> tuple[list[str], list[str]]:
    (p_llm_input_content, p_llm_output_content), (q_llm_input_content, q_llm_output_content) = (
        get_input_and_output_text_camel_separate(messages)
    )

    input_content = [*p_llm_input_content, *q_llm_input_content]
    output_content = [*p_llm_output_content, *q_llm_output_content]

    return input_content, output_content


def count_tokens(input_content: list[str], output_content: list[str]) -> tuple[int, int]:
    enc = tiktoken.encoding_for_model("gpt-4o")
    n_input_tokens = [len(enc.encode(t)) for t in input_content]
    n_output_tokens = [len(enc.encode(t)) for t in output_content]
    return sum(n_input_tokens), sum(n_output_tokens)


def count_tokens_for_task(
    logs_path: Path, model: str, suite: str, user_task: str, injection_task: str | None, attack_name: str | None
) -> tuple[int, int]:
    task_path = logs_path / model / suite / user_task / (attack_name or "none") / f"{injection_task or 'none'}.json"
    task_result = TaskResults.model_validate_json(task_path.read_text())
    if "+camel" in model:
        input_content, output_content = get_input_and_output_text_camel(task_result.messages)
    elif "+tool_filter" in model:
        input_content, output_content = get_input_and_output_text_tool_filter(task_result.messages)
    else:
        input_content, output_content = get_input_and_output_text_agentdojo(task_result.messages)
    return count_tokens(input_content, output_content)


class TokensPerTask(TypedDict):
    suite: str
    user_task: str
    injection_task: str | None
    input_tokens: int
    output_tokens: int


def count_tokens_for_suite_no_attacks(logs_path: Path, model: str, suite_name: str) -> list[TokensPerTask]:
    suite = get_suite("v1.2", suite_name)
    results: list[TokensPerTask] = []
    for user_task in suite.user_tasks.keys():
        input_tokens, output_tokens = count_tokens_for_task(logs_path, model, suite_name, user_task, None, None)
        results.append(
            TokensPerTask(
                suite=suite_name,
                user_task=user_task,
                injection_task=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
    return results


def count_tokens_for_suite_with_attacks(
    logs_path: Path, model: str, suite_name: str, attack_name: str
) -> list[TokensPerTask]:
    suite = get_suite("v1.2", suite_name)
    results: list[TokensPerTask] = []
    for user_task in suite.user_tasks.keys():
        for injection_task in suite.injection_tasks.keys():
            input_tokens, output_tokens = count_tokens_for_task(
                logs_path, model, suite_name, user_task, injection_task, attack_name
            )
            results.append(
                TokensPerTask(
                    suite=suite_name,
                    user_task=user_task,
                    injection_task=injection_task,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

    return results


def count_tokens_for_model(logs_path: Path, model: str, attack_name: str | None) -> list[TokensPerTask]:
    suites = get_suites("v1.2")
    results = []
    for suite in suites.keys():
        if attack_name is None:
            results += count_tokens_for_suite_no_attacks(logs_path, model, suite)
        else:
            results += count_tokens_for_suite_with_attacks(logs_path, model, suite, attack_name)
    return results
