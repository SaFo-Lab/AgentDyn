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
from datetime import datetime
from typing import ClassVar

from camel import security_policy
from camel.capabilities import Capabilities
from camel.capabilities.readers import Public
from camel.capabilities.sources import Tool
from camel.interpreter import (
    interpreter,
    result,
    value,
)
from camel.interpreter import namespace as ns


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


@dataclasses.dataclass
class AClass:
    a: int


def add(a: int, b: int) -> AClass:
    return AClass(a + b)


_a_1 = value.CaMeLInt(1, Capabilities.default(), ())
_a_2 = value.CaMeLInt(2, Capabilities.default(), ())
_a_3 = value.CaMeLInt(3, Capabilities.default(), ())
_empty_dict = value.CaMeLDict({}, Capabilities.default(), ())

_a_class = value.CaMeLClass("AClass", AClass, Capabilities.camel(), (), {})
_a_class_instance = _a_class.init(ns.Namespace(), value.CaMeLInt(3, Capabilities.camel(), ()))

builtins_namespace = ns.Namespace.with_builtins()
_from_isoformat_result = value.CaMeLClassInstance(
    datetime.fromisoformat("2024-05-19 10:00"),
    builtins_namespace.variables["datetime"],  # type: ignore
    Capabilities(frozenset({Tool("fromisoformat")}), Public()),
    builtins_namespace,
    (
        builtins_namespace.variables["datetime"]
        ._methods["fromisoformat"]  # type: ignore
        .new_with_dependencies((builtins_namespace.variables["datetime"],)),
        value.CaMeLTuple(
            (
                value.CaMeLStr.from_raw(
                    "2024-05-19 10:00",
                    Capabilities.default(),
                    (),
                ),
            ),
            Capabilities.default(),
            (),
        ),
        _empty_dict,
    ),
)

camel_add = value.CaMeLBuiltin("add", add, Capabilities.camel(), ())


class TestAssignments:
    test_data: ClassVar = [
        dict(
            testcase_name="simple_builtin_function",
            code="len([1, 2, 3])",
            expected=value.CaMeLInt(
                3,
                Capabilities(frozenset({Tool("len")}), Public()),
                (
                    builtins_namespace.variables["len"],
                    value.CaMeLTuple(
                        (value.CaMeLList([_a_1, _a_2, _a_3], Capabilities.default(), ()),),
                        Capabilities.default(),
                        (),
                    ),
                    _empty_dict,
                ),
            ),
            namespace=ns.Namespace.with_builtins(),
            expected_namespace=ns.Namespace.with_builtins(),
            expected_tool_calls=[
                interpreter.FunctionCall(
                    function="len",
                    object_type=None,
                    args={"0": [1, 2, 3]},
                    output=3,
                    is_builtin=True,
                )
            ],
        ),
        dict(
            testcase_name="simple_builtin_function_variadic_arguments",
            code="max(1, 2, 3)",
            expected=value.CaMeLInt(
                3,
                Capabilities(frozenset({Tool("max")}), Public()),
                (
                    builtins_namespace.variables["max"],
                    value.CaMeLTuple(
                        (_a_1, _a_2, _a_3),
                        Capabilities.default(),
                        (),
                    ),
                    _empty_dict,
                ),
            ),
            namespace=ns.Namespace.with_builtins(),
            expected_namespace=ns.Namespace.with_builtins(),
            expected_tool_calls=[
                interpreter.FunctionCall(
                    function="max",
                    object_type=None,
                    args={"0": 1, "1": 2, "2": 3},
                    output=3,
                    is_builtin=True,
                )
            ],
        ),
        dict(
            testcase_name="simple_builtin_function_with_data",
            code="bool(1)",
            expected=value.CaMeLTrue(
                Capabilities(frozenset({Tool("bool")}), Public()),
                (
                    builtins_namespace.variables["bool"],
                    value.CaMeLTuple(
                        (_a_1,),
                        Capabilities.default(),
                        (),
                    ),
                    _empty_dict,
                ),
            ),
            namespace=ns.Namespace.with_builtins(),
            expected_namespace=ns.Namespace.with_builtins(),
            expected_tool_calls=[
                interpreter.FunctionCall(
                    function="bool",
                    object_type=None,
                    args={"0": 1},
                    output=True,
                    is_builtin=True,
                )
            ],
        ),
        dict(
            testcase_name="simple_builtin_method",
            code="'h'.upper()",
            expected=value.CaMeLStr.from_raw(
                "H",
                Capabilities(frozenset({Tool("upper")}), Public()),
                (
                    value.make_camel_builtin("upper", str.upper).new_with_dependencies(
                        (
                            value.CaMeLStr.from_raw(
                                "h",
                                Capabilities.default(),
                                (),
                            ),
                        )
                    ),
                    value.CaMeLTuple(
                        (
                            value.CaMeLStr.from_raw(
                                "h",
                                Capabilities.default(),
                                (),
                            ),
                        ),
                        Capabilities.default(),
                        (),
                    ),
                    _empty_dict,
                ),
            ),
            namespace=ns.Namespace.with_builtins(),
            expected_namespace=ns.Namespace.with_builtins(),
            expected_tool_calls=[
                interpreter.FunctionCall(
                    function="upper",
                    object_type="str",
                    args={"0": "h"},
                    output="H",
                    is_builtin=True,
                )
            ],
        ),
        dict(
            testcase_name="simple_arbitrary_return_type",
            code="add(a=1, b=2)",
            expected=_a_class_instance.new_with_dependencies(
                (
                    camel_add,
                    value.CaMeLTuple((), Capabilities.default(), ()),
                    value.CaMeLDict(
                        {
                            value.CaMeLStr.from_raw("a", Capabilities.default(), ()): _a_1,
                            value.CaMeLStr.from_raw("b", Capabilities.default(), ()): _a_2,
                        },
                        Capabilities.default(),
                        (),
                    ),
                )
            ).new_with_metadata(Capabilities(frozenset({Tool("add")}), Public())),
            namespace=ns.Namespace(
                {
                    "AClass": _a_class,
                    "add": camel_add,
                }
            ),
            expected_namespace=None,
            expected_tool_calls=[
                interpreter.FunctionCall(
                    function="add",
                    object_type=None,
                    args={"a": 1, "b": 2},
                    output=AClass(3),
                    is_builtin=True,
                )
            ],
        ),
        dict(
            testcase_name="classmethod",
            code="datetime.fromisoformat('2024-05-19 10:00')",
            expected=_from_isoformat_result,
            namespace=builtins_namespace,
            expected_namespace=builtins_namespace,
            expected_tool_calls=[
                interpreter.FunctionCall(
                    function="fromisoformat",
                    object_type="type",
                    args={"0": "2024-05-19 10:00"},
                    output=datetime.fromisoformat("2024-05-19 10:00"),
                    is_builtin=True,
                )
            ],
        ),
    ]

    def test_builtin_function_calls(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
        expected_tool_calls: list[interpreter.FunctionCall],
    ):
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
        assert expected == got_code.value
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace
        assert expected_tool_calls == got_tool_calls
