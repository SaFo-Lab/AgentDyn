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

"""Module containing the implementation for the PrivilegedLLM."""

import dataclasses
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import pydantic
import yaml
from agentdojo import agent_pipeline, functions_runtime
from agentdojo import types as ad_types
from agentdojo.agent_pipeline import tool_execution
from agentdojo.default_suites.v1.banking.task_suite import BankingEnvironment
from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment
from agentdojo.default_suites.v1.workspace.task_suite import (
    WorkspaceEnvironment,
)
from pydantic_ai.models import KnownModelName

from camel import quarantined_llm, system_prompt_generator
from camel.capabilities import is_trusted
from camel.interpreter import interpreter, result
from camel.interpreter import namespace as ns
from camel.interpreter.value import CaMeLValue
from camel.pipeline_elements.agentdojo_function import (
    make_agentdojo_namespace,
)
from camel.pipeline_elements.security_policies import (
    AgentDojoSecurityPolicyEngine,
)


# Now yaml.safe_dump will work with datetime objects
# But you'd need to use yaml.dump instead of yaml.safe_dump
def custom_yaml_dump(obj: dict | list) -> str:
    return yaml.dump(obj, default_flow_style=False)


_T = TypeVar("_T", bound=str | int | float | pydantic.BaseModel)
_E = TypeVar("_E", bound=functions_runtime.TaskEnvironment)


def extract_print_output(tool_calls: Sequence[interpreter.FunctionCall]) -> str:
    """Extracts and concatenates arguments from print calls."""

    printed_output = ""
    print_calls = [tc for tc in tool_calls if tc.function == "print"]
    for print_call in print_calls:
        for arg_value in print_call.args.values():
            printed_output += str(arg_value)
    return printed_output


def function_call_from_ad_function_call(function_call: interpreter.FunctionCall) -> functions_runtime.FunctionCall:
    args = {k: (v if not isinstance(v, type) else repr(v)) for k, v in function_call.args.items()}
    return functions_runtime.FunctionCall(function=function_call.function, args=args)


def make_ad_tool_calls(
    tool_calls: Sequence[interpreter.FunctionCall],
) -> list[tuple[functions_runtime.FunctionCall, Any]]:
    return [
        (
            function_call_from_ad_function_call(tool_call),
            tool_call.output,
        )
        for tool_call in tool_calls
        if not isinstance(tool_call.output, Exception) and not tool_call.is_builtin
    ]


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
    if is_trusted(camel_exception):
        exception_text = str(exception)
    else:
        exception_text = "<The exception was redacted because it came from an untrusted source. Try to infer what the problem was from the context provided.>"
    return f"""
Traceback (most recent call last):
File "<stdin>", line {camel_exception.nodes[-1].lineno}, in <module>
{formatted_code}

{type(exception).__name__}: {exception_text}
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


def _highlight_exception_code(
    code: str,
    exception: Exception,
    lineno: int,
    col_offset: int,
    end_lineno: int | None,
    end_col_offset: int | None,
) -> str:
    """Highlights the affected code in a multi-line Python code string.

    Args:
        code: The multi-line Python code.
        exception: The exception that occurred.
        lineno: The line number where the exception occurred (1-based).
        col_offset: The column offset where the exception occurred (0-based).
        end_lineno: The end line number of the affected code (1-based), or None if it's a single line.
        end_col_offset: The end column offset of the affected code (0-based), or None if it's a single line.

    Returns:
        A string with the affected code highlighted, or an error message if there's an issue.
    """

    lines = code.splitlines(keepends=True)

    if not 1 <= lineno <= len(lines):
        return f"Error: lineno {lineno} is out of range (1-{len(lines)})."

    highlighted_code = ""

    if end_lineno is None or end_lineno == lineno:  # Single line error
        line = lines[lineno - 1]
        highlighted_code += line
        highlighted_code += (
            " " * col_offset + "^" * (1 if end_col_offset is None else max(1, end_col_offset - col_offset)) + "\n"
        )
    elif 1 <= end_lineno <= len(lines) and end_lineno > lineno:  # Multiline error
        for i in range(lineno - 1, end_lineno):
            current_line = lines[i]
            highlighted_code += current_line
            if i == lineno - 1:
                highlighted_code += " " * col_offset + "^" * (len(current_line) - 1 - col_offset) + "\n"
            elif i == end_lineno - 1:
                highlighted_code += (
                    "^" * (len(current_line) - 1 if end_col_offset is None else end_col_offset) + "\n"
                )  # Fixed here
            else:
                highlighted_code += "^" * (len(current_line) - 1) + "\n"

    else:
        return f"Error: end_lineno {end_lineno} is out of range or not after lineno."

    return highlighted_code


def _get_quarantined_llm(model: KnownModelName) -> KnownModelName:
    if "openai" in model and "o1" in model:
        return "openai:gpt-4o"
    return model


class PrivilegedLLM(agent_pipeline.BasePipelineElement):
    """A pipeline element that generates and interprets code expressing the user query.

    Args:
        llm: the LLM to use to generate the code.
        system_prompt_generator: a function which takes as argument the functions available
          to the LLM and returns the system prompt to use.
        security_policies: a list of security policy tuples, where the first element is the tool name and the second one is the security policy to apply.
        eval_mode: the evaluation mode for the interpreter.
        quarantined_llm_retries: the number of retries for the quarantined LLM.
    """

    def __init__(
        self,
        llm: agent_pipeline.BasePipelineElement,
        security_policy_engine: type[AgentDojoSecurityPolicyEngine],
        quarantined_llm_model: KnownModelName,
        system_prompt_generator: Callable[
            [Iterable[functions_runtime.Function], set[str]], str
        ] = system_prompt_generator.default_system_prompt_generator,
        eval_mode: interpreter.MetadataEvalMode = interpreter.MetadataEvalMode.NORMAL,
        quarantined_llm_retries: int = 10,
        max_attempts: int = 10,
    ) -> None:
        """Initializes the PrivilegedLLM."""
        self.llm = llm
        self.system_prompt_generator = system_prompt_generator
        self.security_policy_engine = security_policy_engine
        self.eval_mode = eval_mode
        self.quarantined_llm_retries = quarantined_llm_retries
        self.dummy_runtime = functions_runtime.FunctionsRuntime()
        self.max_attempts = max_attempts
        # Gemini thinking and o1-* do not support JSON mode, so we fall back to base base Gemini/4o
        self.quarantined_llm_model: KnownModelName = _get_quarantined_llm(quarantined_llm_model)

    def run_code(
        self,
        code: str,
        env: functions_runtime.TaskEnvironment,
        namespace: ns.Namespace,
        dependencies: Iterable[CaMeLValue],
    ) -> tuple[
        str,
        Sequence[tuple[functions_runtime.FunctionCall, functions_runtime.FunctionReturnType]],
        interpreter.CaMeLException | None,  # Add optional error return
        ns.Namespace,
        Iterable[CaMeLValue],
    ]:
        """Interprets the code in `code` by calling the provided functions.

        Args:
            code: a string with the code to interpret
            env: the task environment.
            ad_functions_runtime: the functions runtime.

        Returns:
            A tuple containing the final model output, a list of tuples where the first
            element is the function call that was done and the second one is the result,
            and an optional Exception if one occurred during interpretation.
        """

        eval_args = interpreter.EvalArgs(self.security_policy_engine(env), self.eval_mode)

        print(code)

        interpreter_res, updated_namespace, tool_calls, dependencies = interpreter.parse_and_interpret_code(
            code, namespace, [], dependencies, eval_args
        )

        printed_output = extract_print_output(tool_calls)
        ad_tool_calls = make_ad_tool_calls(tool_calls)

        match interpreter_res:
            case result.Error(error):
                return (printed_output, ad_tool_calls, error, updated_namespace, dependencies)  # Return the exception
            case result.Ok(v):
                res = v

        return (
            f"{printed_output}\n{res.raw if res.raw is not None else ''}",
            ad_tool_calls,
            None,
            updated_namespace,
            dependencies,
        )  # No error

    def _generate_and_interpret_code(
        self,
        query: str,
        runtime: functions_runtime.FunctionsRuntime,
        namespace: ns.Namespace,
        env: functions_runtime.TaskEnvironment,
        messages: Sequence[ad_types.ChatMessage],
        privileged_llm_messages: Sequence[ad_types.ChatMessage],
        system_prompt: str,
        previous_printed_output: str,
        dependencies: Iterable[CaMeLValue],
    ) -> tuple[
        str,
        Sequence[tuple[functions_runtime.FunctionCall, functions_runtime.FunctionReturnType]],
        interpreter.CaMeLException | None,
        list[ad_types.ChatMessage],
        list[ad_types.ChatMessage],
        ns.Namespace,
        Iterable[CaMeLValue],
    ]:
        if len(privileged_llm_messages) == 0:
            privileged_llm_messages = [
                ad_types.ChatSystemMessage(
                    role="system", content=[ad_types.text_content_block_from_string(system_prompt)]
                ),
                ad_types.ChatUserMessage(role="user", content=[ad_types.text_content_block_from_string(query)]),
            ]

        code = None
        attempts = 5
        while (code is None or code == "") and attempts > 0:
            _, _, _, [*_, code_message], _ = self.llm.query(
                query=query,
                runtime=self.dummy_runtime,
                messages=privileged_llm_messages,
            )
            if self.llm.name is not None and "gemini" in self.llm.name and "exp" in self.llm.name:
                time.sleep(6)
            assert code_message["role"] == "assistant"
            if code_message["content"]:
                code = ad_types.get_text_content_as_str(code_message["content"])
            attempts -= 1

        if code is None or code == "":
            empty_message = ad_types.ChatAssistantMessage(
                role="assistant",
                content=[
                    ad_types.text_content_block_from_string(
                        "The generated code was `None`. This message is added by the system does not come from the assistant."
                    )
                ],
                tool_calls=None,
            )
            return (
                previous_printed_output,
                [],
                None,
                [*messages, empty_message],
                list(privileged_llm_messages),
                namespace,
                dependencies,
            )

        model_output, tool_calls_results, interpretation_error, namespace, dependencies = self.run_code(
            code, env, namespace, dependencies
        )

        tool_calls: list[functions_runtime.FunctionCall] = []
        tool_call_messages: list[ad_types.ChatMessage] = []

        for tool_call, tool_result in tool_calls_results:
            tool_call_messages.append(
                ad_types.ChatToolResultMessage(
                    role="tool",
                    tool_call=tool_call,
                    content=[
                        ad_types.text_content_block_from_string(
                            tool_execution.tool_result_to_str(tool_result, dump_fn=custom_yaml_dump)
                        )
                    ],
                    tool_call_id=None,
                    error=None,
                )
            )
            tool_calls.append(tool_call)

        if interpretation_error:
            error_messages = make_error_messages(code, interpretation_error)
            updated_messages = [*messages, *tool_call_messages, *error_messages]
            privileged_llm_messages = [*privileged_llm_messages, *error_messages]
        else:
            updated_messages = [
                *messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(code)],
                    tool_calls=tool_calls,
                ),
                *tool_call_messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(previous_printed_output + model_output)],
                    tool_calls=None,
                ),
            ]

        return (
            previous_printed_output + model_output,
            tool_calls_results,
            interpretation_error,
            updated_messages,
            list(privileged_llm_messages),
            namespace,
            dependencies,
        )

    def query(
        self,
        query: str,
        runtime: functions_runtime.FunctionsRuntime,
        env: _E = functions_runtime.EmptyEnv(),
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[
        str,
        functions_runtime.FunctionsRuntime,
        _E,
        Sequence[ad_types.ChatMessage],
        dict,
    ]:
        """Generates and interprets code which expresses the user query."""
        privileged_llm_messages = []

        def query_ai_assistant(query: str, output_schema: type[_T]) -> _T:
            return quarantined_llm.query_quarantined_llm(
                llm=(self.quarantined_llm_model),
                query=query,
                output_schema=output_schema,
                retries=self.quarantined_llm_retries,
            )

        query_ai_assistant.__doc__ = quarantined_llm.query_quarantined_llm.__doc__
        runtime.register_function(query_ai_assistant)

        builtins_namespace = ns.Namespace.with_builtins()
        # Models get confused in the other suites and should not use datetime stuff
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

        system_prompt = self.system_prompt_generator(runtime.functions.values(), classes_to_exclude)

        if isinstance(env, BankingEnvironment):
            system_prompt += "\n\nNote that, in the transaction history, the transactions from the user have 'me' as sender, and still habe positive amounts."

        if not isinstance(env, WorkspaceEnvironment):
            new_variables = {k: v for k, v in builtins_namespace.variables.items() if k not in classes_to_exclude}
            builtins_namespace = dataclasses.replace(builtins_namespace, variables=new_variables)

        namespace = builtins_namespace.add_variables(make_agentdojo_namespace(builtins_namespace, runtime, env))

        model_output = ""
        dependencies = ()

        for _ in range(self.max_attempts):
            (model_output, _, interpretation_error, messages, privileged_llm_messages, namespace, dependencies) = (
                self._generate_and_interpret_code(
                    query,
                    runtime,
                    namespace,
                    env,
                    messages,
                    privileged_llm_messages,
                    system_prompt,
                    model_output,
                    dependencies,
                )
            )

            if not interpretation_error:
                break  # Exit loop if no error

        extra_args["camel_namespace"] = namespace
        if messages[-1]["role"] == "user" and "\n\nTraceback" in ad_types.get_text_content_as_str(
            messages[-1]["content"]
        ):
            messages = [*messages, ad_types.ChatAssistantMessage(role="assistant", content=None, tool_calls=None)]

        return query, runtime, env, messages, extra_args
