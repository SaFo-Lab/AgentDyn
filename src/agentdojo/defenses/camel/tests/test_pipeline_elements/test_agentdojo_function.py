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

import ast
import dataclasses
from typing import ClassVar

from agentdojo import functions_runtime
from pydantic import BaseModel

from camel import security_policy
from camel.capabilities import Capabilities, sources
from camel.capabilities.readers import Public
from camel.interpreter import (
    interpreter,
    result,
    value,
)
from camel.interpreter import namespace as ns
from camel.pipeline_elements import agentdojo_function


def pytest_generate_tests(metafunc):
    # We are not using pytest.parametrize directly to make it easier to move
    # to GoogleTest later if needed.
    idlist = []
    argvalues = []
    argnames = []
    for scenario in metafunc.cls.test_data:
        idlist.append(scenario["testcase_name"])
        test_params = {k: v for k, v in scenario.items() if k != "testcase_name"}
        argnames = list(test_params.keys())
        argvalues.append(list(test_params.values()))
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


def add(a: int, b: int) -> int:
    """Sums two numbers.

    :param a: The first number.
    :param b: The second number.
    """
    return a + b


class TestAgentDojoFunction:
    test_data: ClassVar = [
        dict(
            testcase_name="simple_function",
            code="add(1, 2)",
            expected=value.CaMeLInt(
                3,
                Capabilities(frozenset({sources.Tool("add")}), Public()),
                (),
            ),
            namespace=ns.Namespace.with_builtins(),
            expected_namespace=None,
            expected_tool_calls=[
                interpreter.FunctionCall(
                    function="add",
                    object_type=None,
                    args={"a": 1, "b": 2},
                    output=3,
                    is_builtin=False,
                )
            ],
        ),
    ]

    def test_agentdojo_function(
        self,
        code: str,
        expected: value.CaMeLValue | BaseModel,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
        expected_tool_calls: list[interpreter.FunctionCall],
    ):
        runtime = functions_runtime.FunctionsRuntime()
        runtime.register_function(add)
        namespace = dataclasses.replace(
            namespace,
            variables=namespace.variables | agentdojo_function.make_agentdojo_namespace(namespace, runtime, None),
        )
        got_code, got_namespace, got_tool_calls, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(
                security_policy.NoSecurityPolicyEngine(),
                interpreter.MetadataEvalMode.NORMAL,
            ),
        )
        assert isinstance(got_code, result.Ok)
        if isinstance(expected, BaseModel):
            assert expected == got_code.value.raw
        else:
            assert expected.raw == got_code.value.raw
            assert expected._metadata == got_code.value._metadata

            assert (
                namespace.variables["add"],
                value.CaMeLTuple(
                    (
                        value.CaMeLInt(1, Capabilities.default(), ()),
                        value.CaMeLInt(2, Capabilities.default(), ()),
                    ),
                    Capabilities.default(),
                    (),
                ),
                value.CaMeLDict({}, Capabilities.default(), ()),
            ) == got_code.value._dependencies

        assert expected_tool_calls == got_tool_calls
