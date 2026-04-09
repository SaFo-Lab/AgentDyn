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
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeAlias, TypeVar

import pydantic
import yaml
from agentdojo import attacks, functions_runtime
from agentdojo import types as ad_types
from agentdojo.agent_pipeline import AgentPipeline, BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment
from agentdojo.default_suites.v1.workspace.user_tasks import WorkspaceEnvironment
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from camel import quarantined_llm
from camel.chat_turn import make_turns
from camel.interpreter import interpreter, result
from camel.interpreter.namespace import Namespace
from camel.pipeline_elements.agentdojo_function import make_agentdojo_namespace
from camel.pipeline_elements.privileged_llm import (
    _highlight_exception_code,
    extract_print_output,
    make_ad_tool_calls,
)
from camel.pipeline_elements.security_policies.agentdojo_security_policies import (
    AgentDojoSecurityPolicyEngine,
)
from camel.security_policy import NoSecurityPolicyEngine, SecurityPolicyDeniedError

FunctionCall.model_rebuild()

from agentdojo.benchmark import TaskResults

TaskResults.model_rebuild()


_E = TypeVar("_E", bound=functions_runtime.TaskEnvironment)
_T = TypeVar("_T", bound=str | int | float | pydantic.BaseModel)

_RelevantMessages: TypeAlias = ad_types.ChatAssistantMessage | ad_types.ChatToolResultMessage


def _make_quarantined_llm_fn(messages: list[ad_types.ChatToolResultMessage]) -> Callable[[str, type[_T]], _T]:
    calls = [message for message in messages if message["tool_call"].function == "query_ai_assistant"]
    i = 0

    def query_ai_assistant(query: str, output_schema: type[_T]) -> _T:
        nonlocal i
        if i >= len(calls):
            raise quarantined_llm.NotEnoughInformationError()
        message = calls[i]
        i += 1
        output = yaml.safe_load(ad_types.get_text_content_as_str(message["content"]))
        if issubclass(output_schema, pydantic.BaseModel):
            return output_schema(**output)
        return output_schema(output)

    return query_ai_assistant


def format_camel_exception(camel_exception: interpreter.CaMeLException, code: str) -> str:
    exception = camel_exception.exception
    node = camel_exception.nodes[-1]
    try:
        extracted_code = interpreter.extract_code_block(code)
    except interpreter.InvalidOutputError:
        extracted_code = code
    formatted_code = _highlight_exception_code(
        extracted_code,
        camel_exception.exception,
        node.lineno,
        node.col_offset if hasattr(node, "col_offset") else 0,
        node.end_lineno if hasattr(node, "end_lineno") else None,
        node.end_col_offset if hasattr(node, "end_col_offset") else None,
    )
    return f"""
Traceback (most recent call last):
File "<stdin>", line {camel_exception.nodes[-1].lineno}, in <module>
{formatted_code}

{type(exception).__name__}: {exception}
"""


def make_error_messages(code: str, interpretation_error: interpreter.CaMeLException) -> list[ad_types.ChatMessage]:
    return [
        ad_types.ChatAssistantMessage(
            role="assistant",
            content=[ad_types.text_content_block_from_string(code)],
            tool_calls=None,
        ),
        ad_types.ChatUserMessage(
            role="user",
            content=[
                ad_types.text_content_block_from_string(f"""\
Running the code gave the following error:
{format_camel_exception(interpretation_error, code)}
Provide the new code with the error fixed. Provide *all the code* so that I \
can directly run it. If the error comes from a search query that did not \
return any results, then try the query with different parameters. The code \
up to the line before the one where the exception was thrown has already been \
executed and the variables and defined classes will \
still be accessible to you. It's very important that you do not re-write code to run \
functions that have side-effects (e.g., functions that send an email).
""")
            ],
        ),
    ]


def replay_task(
    pipeline_name: str,
    suite_name: str,
    user_task_id: str,
    injection_task_id: str | None,
    attack_name: str | None,
    eval_args: interpreter.EvalArgs,
    env: _E,
    runtime: FunctionsRuntime,
) -> tuple[list[ad_types.ChatMessage], _E]:
    trace_path = (
        Path("logs")
        / pipeline_name
        / suite_name
        / user_task_id
        / (attack_name or "none")
        / f"{(injection_task_id or 'none')}.json"
    )

    execution_trace = TaskResults.model_validate_json(trace_path.read_text())

    # exclude query message
    relevant_messages = execution_trace.messages[1:]
    if relevant_messages[-1]["role"] == "assistant" and relevant_messages[-1]["content"] is None:
        relevant_messages = relevant_messages[:-1]
    turns = make_turns(relevant_messages)

    namespace = Namespace.with_builtins()
    classes_to_exclude = (
        {
            "datetime",
            "timedelta",
            "date",
            "time",
            "NaiveDatetime",
            "timezone",
        }
        if not isinstance(env, WorkspaceEnvironment | TravelEnvironment)
        else set()
    )

    if not isinstance(env, WorkspaceEnvironment):
        new_variables = {k: v for k, v in namespace.variables.items() if k not in classes_to_exclude}
        namespace = dataclasses.replace(namespace, variables=new_variables)

    printed_output = ""
    messages = []
    dependencies = ()

    for turn in turns:
        # Update query_ai_assistant function with this turn's calls
        query_ai_assistant: Callable = _make_quarantined_llm_fn(turn.tool_results)
        query_ai_assistant.__doc__ = quarantined_llm.query_quarantined_llm.__doc__
        runtime.register_function(query_ai_assistant)
        namespace = namespace.add_variables(make_agentdojo_namespace(namespace, runtime, env))

        try:
            res, namespace, tool_calls_chain, dependencies = interpreter.parse_and_interpret_code(
                turn.code, namespace, [], dependencies, eval_args
            )
        except SecurityPolicyDeniedError as e:
            error_message = ad_types.ChatUserMessage(
                role="user", content=[ad_types.text_content_block_from_string(f"SecurityPolicyDeniedError: {e}")]
            )
            assistant_message = ad_types.ChatAssistantMessage(
                role="assistant",
                content=[ad_types.text_content_block_from_string(printed_output)],
                tool_calls=None,
            )
            return [*messages, error_message, assistant_message], env
        turn_tool_calls = make_ad_tool_calls(tool_calls_chain)
        tool_call_messages = []
        tool_calls = []
        for tool_call, tool_result in turn_tool_calls:
            tool_call_messages.append(
                ad_types.ChatToolResultMessage(
                    role="tool",
                    tool_call=tool_call,
                    content=[ad_types.text_content_block_from_string(tool_result_to_str(tool_result))],
                    tool_call_id=None,
                    error=None,
                )
            )
            tool_calls.append(tool_call)
        if isinstance(res, result.Ok):
            printed_result = str(res.value.raw) if res.value.raw is not None else ""
            printed_output = f"{printed_output}\n{extract_print_output(tool_calls_chain)}\n{printed_result}"
            messages = [
                *messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(turn.code)],
                    tool_calls=tool_calls,
                ),
                *tool_call_messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(printed_output)],
                    tool_calls=None,
                ),
            ]
        else:
            printed_output = f"{printed_output}\n{extract_print_output(tool_calls_chain)}"
            error_messages = make_error_messages(turn.code, res.error)
            messages = [*messages, *tool_call_messages, *error_messages]

    if messages[-1]["role"] == "user" and "\n\nTraceback" in ad_types.get_text_content_as_str(messages[-1]["content"]):
        messages.append(ad_types.ChatAssistantMessage(role="assistant", content=None, tool_calls=None))

    return messages, env


class UserInjectionTasksGetter(BasePipelineElement):
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: _E = EmptyEnv(),
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, _E, Sequence[ad_types.ChatMessage], dict]:
        stack = inspect.stack()
        # Hacy way to retrieve current suite and user/injection task ids
        # Assuming the call stack is:
        # - run_task_with{out}_injection_tasks
        # - run_task_with_pipeline
        # - AgentPipeline.query
        # - UserInjectionTasksGetter.query

        # get suite name from run_task_with{out}_injection_tasks and save in extra args
        run_task_with_out_injection_tasks_stack = stack[3]
        extra_args["task_suite_name"] = run_task_with_out_injection_tasks_stack.frame.f_locals["suite"].name

        # get user and injection task ids from run_task_with_pipeline and save in extra args
        run_task_with_pipeline_stack = stack[2]
        user_task: BaseUserTask = run_task_with_pipeline_stack.frame.f_locals["user_task"]
        injection_task: BaseInjectionTask | None = run_task_with_pipeline_stack.frame.f_locals.get("injection_task")
        injection_task_id = injection_task.ID if injection_task is not None else None
        extra_args["user_task_id"] = user_task.ID
        extra_args["injection_task_id"] = injection_task_id

        return query, runtime, env, messages, extra_args


class PrivilegedLLMReplayer(BasePipelineElement):
    def __init__(
        self,
        pipeline_name: str,
        attack_name: str,
        security_policy_engine: type[AgentDojoSecurityPolicyEngine],
        eval_mode: interpreter.MetadataEvalMode = interpreter.MetadataEvalMode.NORMAL,
    ) -> None:
        super().__init__()
        self._pipeline_name = pipeline_name
        self._attack_name = attack_name
        self._eval_mode = eval_mode
        self._security_policy_engine = security_policy_engine

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: _E = EmptyEnv(),
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, _E, Sequence[ad_types.ChatMessage], dict]:
        if (
            "user_task_id" not in extra_args
            or "injection_task_id" not in extra_args
            or "task_suite_name" not in extra_args
        ):
            raise ValueError("Unable to access the current user task and/or injection task ids")
        policy_engine = self._security_policy_engine(env)
        eval_args = interpreter.EvalArgs(policy_engine, self._eval_mode)
        # Attacks are not run when evaluating injection tasks as user tasks
        attack_name = self._attack_name if extra_args["injection_task_id"] is not None else None
        new_messages, env = replay_task(
            self._pipeline_name,
            extra_args["task_suite_name"],
            extra_args["user_task_id"],
            extra_args["injection_task_id"],
            attack_name,
            eval_args,
            env,
            runtime,
        )
        return query, runtime, env, [*messages, *new_messages], extra_args


def replay_user_task(
    pipeline_name: str, suite_name: str, user_task_id: str, attack_name: str | None, eval_args: interpreter.EvalArgs
):
    path = Path("logs") / pipeline_name / suite_name / user_task_id / (attack_name or "none")
    dummy_pipeline = AgentPipeline([])
    dummy_pipeline.name = pipeline_name
    suite = get_suite("v1.2", suite_name)
    runtime = FunctionsRuntime(suite.tools)
    for task_id_path in path.glob("*"):
        injection_task_id = task_id_path.stem
        if attack_name is not None:
            assert injection_task_id is not None
            attack = attacks.load_attack(attack_name, suite, dummy_pipeline)
            user_task = suite.get_user_task_by_id(user_task_id)
            injection_task = suite.get_injection_task_by_id(injection_task_id)
            injections = attack.attack(user_task, injection_task)
            env = suite.load_and_inject_default_environment(injections)
        else:
            env = suite.load_and_inject_default_environment({})
        replay_task(pipeline_name, suite_name, user_task_id, injection_task_id, attack_name, eval_args, env, runtime)


def replay_suite(pipeline_name: str, suite_name: str, attack_name: str | None, eval_args: interpreter.EvalArgs):
    path = Path("logs") / pipeline_name / suite_name
    for user_task_path in path.glob("user_task_*"):
        user_task_id = user_task_path.stem
        replay_user_task(pipeline_name, suite_name, user_task_id, attack_name, eval_args)


def replay_benchmark():
    suites = ["workspace", "banking", "travel", "slack"]
    pipeline_name = "claude-3-5-sonnet-latest+camel"
    attack_name = None  # "important_instructions"
    eval_args = interpreter.EvalArgs(
        security_policy_engine=NoSecurityPolicyEngine(),
        eval_mode=interpreter.MetadataEvalMode.NORMAL,
    )
    for suite_name in suites:
        replay_suite(pipeline_name, suite_name, attack_name, eval_args)
