import re
from collections.abc import Sequence
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage


def _tool_call_to_bedrock(tool_call: FunctionCall) -> dict[str, Any]:
    if tool_call.id is None:
        raise ValueError("Tool call ID is required for Bedrock Anthropic")
    return {
        "toolUse": {
            "toolUseId": tool_call.id,
            "name": tool_call.function,
            "input": dict(tool_call.args),
        }
    }


def _message_to_bedrock(message: ChatMessage) -> dict[str, Any]:
    match message["role"]:
        case "user":
            return {"role": "user", "content": [{"text": message["content"]}]}
        case "assistant":
            if message["tool_calls"] is not None:
                content = [_tool_call_to_bedrock(tool_call) for tool_call in message["tool_calls"]]
            else:
                content = [{"text": message["content"] or ""}]
            return {"role": "assistant", "content": content}
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` is required for Bedrock Anthropic")
            tool_result_content = ""
            if message["error"] is not None:
                tool_result_content = message["error"]
            elif message["content"]:
                tool_result_content = message["content"]
            return {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": message["tool_call_id"],
                            "content": [{"text": tool_result_content}],
                            "status": "error" if message["error"] is not None else "success",
                        }
                    }
                ],
            }
        case _:
            raise ValueError(f"Invalid message role for Bedrock Anthropic: {message['role']}")


def _merge_tool_result_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = list(messages)
    i = 0
    while i < len(messages) - 1:
        message_1 = messages[i]
        message_2 = messages[i + 1]

        is_tool_result_1 = (
            message_1["role"] == "user"
            and len(message_1["content"]) > 0
            and "toolResult" in message_1["content"][0]
        )
        is_tool_result_2 = (
            message_2["role"] == "user"
            and len(message_2["content"]) > 0
            and "toolResult" in message_2["content"][0]
        )

        if not (is_tool_result_1 and is_tool_result_2):
            i += 1
            continue

        message_1["content"] = [*message_1["content"], *message_2["content"]]
        messages.pop(i + 1)
    return messages


def _conversation_to_bedrock(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    if len(messages) == 0:
        raise ValueError("At least one message is required for Bedrock Anthropic")

    system_prompt = None
    if messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        bedrock_messages = [_message_to_bedrock(message) for message in messages[1:]]
    else:
        bedrock_messages = [_message_to_bedrock(message) for message in messages]

    bedrock_messages = _merge_tool_result_messages(bedrock_messages)
    return system_prompt, bedrock_messages


def _function_to_bedrock(tool: Function) -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"json": tool.parameters.model_json_schema()},
        }
    }


def _bedrock_to_assistant_message(response: dict[str, Any]) -> ChatAssistantMessage:
    message = response["output"]["message"]

    text_content = " ".join([block["text"] for block in message["content"] if "text" in block])
    tool_calls = []
    for block in message["content"]:
        if "toolUse" not in block:
            continue
        tool_use = block["toolUse"]
        tool_calls.append(
            FunctionCall(
                id=tool_use.get("toolUseId"),
                function=tool_use.get("name", ""),
                args=tool_use.get("input", {}),
            )
        )

    return ChatAssistantMessage(role="assistant", content=text_content, tool_calls=tool_calls or None)


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type(ValueError),
)
def chat_completion_request(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    system_prompt: str | None = None,
    temperature: float | None = 0.0,
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
        },
    }

    if temperature is not None:
        request_payload["inferenceConfig"]["temperature"] = temperature
    if tools:
        request_payload["toolConfig"] = {"tools": tools}
    if system_prompt is not None:
        request_payload["system"] = [{"text": system_prompt}]

    try:
        return client.converse(**request_payload)
    except (ClientError, BotoCoreError):
        raise


class BedrockAnthropicLLM(BasePipelineElement):
    """LLM pipeline element using AWS Bedrock Converse API with Anthropic models."""

    _COT_PROMPT = """\
Answer the user's request using relevant tools (if they are available). \
Before calling a tool, do some analysis within <thinking></thinking> tags. \
First, think about which of the provided tools is the relevant tool to answer \
the user's request. Second, go through each of the required parameters of the \
relevant tool and determine if the user has directly provided or given enough \
information to infer a value. When deciding if the parameter can be inferred, \
carefully consider all the context to see if it supports a specific value. \
If all of the required parameters are present or can be reasonably inferred, \
close the thinking tag and proceed with the tool call. BUT, if one of the values \
for a required parameter is missing, DO NOT invoke the function (not even with \
fillers for the missing params) and instead, ask the user to provide the missing \
parameters. DO NOT ask for more information on optional parameters if it is not provided."""
    _MAX_TOKENS = 1024

    def __init__(self, client: Any, model: str, temperature: float | None = 0.0) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        system_prompt, bedrock_messages = _conversation_to_bedrock(messages)
        if "claude-3-sonnet" in self.model or "claude-3-haiku" in self.model:
            system_prompt = f"{self._COT_PROMPT}\n\n{system_prompt}"

        bedrock_tools = [_function_to_bedrock(tool) for tool in runtime.functions.values()]
        completion = chat_completion_request(
            self.client,
            self.model,
            bedrock_messages,
            bedrock_tools,
            max_tokens=self._MAX_TOKENS,
            system_prompt=system_prompt,
            temperature=self.temperature,
        )
        output = _bedrock_to_assistant_message(completion)
        if output["tool_calls"] is not None:
            invalid_tool_calls: list[int] = []
            for i, tool_call in enumerate(output["tool_calls"]):
                if re.match("^[a-zA-Z0-9_-]{1,64}$", tool_call.function) is None:
                    invalid_tool_calls.append(i)
            for index in sorted(invalid_tool_calls, reverse=True):
                del output["tool_calls"][index]

        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
