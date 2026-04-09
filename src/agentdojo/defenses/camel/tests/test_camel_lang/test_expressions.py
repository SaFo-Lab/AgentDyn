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

from camel.capabilities import Capabilities
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.security_policy import NoSecurityPolicyEngine


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
    a: str


_a_class = value.CaMeLClass("AClass", AClass, Capabilities.default(), (), {})
_a_string = value.CaMeLStr.from_raw("a", Capabilities.default(), ())
_a_class_instance = _a_class.init(ns.Namespace(), _a_string)
_a_0 = value.CaMeLInt(0, Capabilities.default(), ())
_a_1 = value.CaMeLInt(1, Capabilities.default(), ())
_a_list = value.CaMeLList([_a_1], Capabilities.default(), ())
_a_dict = value.CaMeLDict({_a_string: _a_1}, Capabilities.default(), ())
_a_true = value.CaMeLTrue(Capabilities.default(), ())
_a_false = value.CaMeLFalse(Capabilities.default(), ())
_a_none = value.CaMeLNone(Capabilities.default(), ())


class TestExpr:
    test_data: ClassVar = [
        dict(
            testcase_name="simple_expr",
            code="'a'",
            expected=value.CaMeLStr.from_raw("a", Capabilities.default(), ()),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="load_name",
            code="a",
            expected=_a_string,
            namespace=ns.Namespace({"a": _a_string}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="load_attr",
            code="a.a",
            expected=_a_string.new_with_dependencies((_a_class_instance,)),
            namespace=ns.Namespace({"a": _a_class_instance}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="load_subscript_list",
            code="l[0]",
            expected=_a_1.new_with_dependencies((_a_list, _a_0)),
            namespace=ns.Namespace({"l": _a_list}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="load_subscript_dict",
            code="d['a']",
            expected=_a_1.new_with_dependencies((_a_dict, _a_string)),
            namespace=ns.Namespace({"d": _a_dict}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="named_expr",
            code="b = (a := 'a')",
            expected=_a_string,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_string, "b": _a_string}),
        ),
        dict(
            testcase_name="if_expr_true",
            code="'a' if True else 'b'",
            expected=_a_string.new_with_dependencies((_a_true,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="if_expr_false",
            code="'b' if False else 'a'",
            expected=_a_string.new_with_dependencies((_a_false,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
    ]

    def test_expressions(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
    ):
        got_code, got_namespace, _, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.STRICT),
        )
        assert isinstance(got_code, result.Ok)
        assert expected == got_code.value
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace


class TestUnaryOp:
    test_data: ClassVar = [
        dict(
            testcase_name="-",
            code="-1",
            expected=value.CaMeLInt(-1, Capabilities.default(), (_a_1,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="+",
            code="+1",
            expected=value.CaMeLInt(1, Capabilities.default(), (_a_1,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="not",
            code="not True",
            expected=value.CaMeLFalse(Capabilities.camel(), (_a_true,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="~",
            code="~0",
            expected=value.CaMeLInt(-1, Capabilities.default(), (_a_0,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
    ]

    def test_unary_op(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
    ):
        got_code, got_namespace, _, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
        )
        assert isinstance(got_code, result.Ok)
        assert expected == got_code.value
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace


_a_542 = value.CaMeLInt(542, Capabilities.default(), ())
_a_2 = value.CaMeLInt(2, Capabilities.default(), ())
_a_2_float = value.CaMeLFloat(2.0, Capabilities.default(), ())


class TestBinOp:
    test_data: ClassVar = [
        *[
            dict(
                testcase_name=op,
                code=f"542 {op} 2",
                expected=value.CaMeLInt(eval(f"542 {op} 2"), Capabilities.camel(), (_a_542, _a_2)),
                namespace=ns.Namespace(),
                expected_namespace=ns.Namespace(),
            )
            for op in ["+", "-", "*", "//", "%", "**", "<<", ">>", "&", "|", "^"]
        ],
        *[
            dict(
                testcase_name=f"{op} int and float",
                code=f"542 {op} 2.0",
                expected=value.CaMeLFloat(
                    eval(f"542 {op} 2.0"),
                    Capabilities.camel(),
                    (_a_542, _a_2_float),
                ),
                namespace=ns.Namespace(),
                expected_namespace=ns.Namespace(),
            )
            for op in ["+", "-", "*", "//", "%", "**"]
        ],
        dict(
            testcase_name="/",
            code="542 / 2",
            expected=value.CaMeLFloat(eval("542 / 2"), Capabilities.camel(), (_a_542, _a_2)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="string_left",
            code="'a' * 2",
            expected=value.CaMeLStr(
                [
                    value._CaMeLChar("a", Capabilities.default(), ()),
                    value._CaMeLChar("a", Capabilities.default(), ()),
                ],
                Capabilities.camel(),
                (_a_string, _a_2),
            ),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="string_right",
            code="2 * 'a'",
            expected=value.CaMeLStr(
                [
                    value._CaMeLChar("a", Capabilities.default(), ()),
                    value._CaMeLChar("a", Capabilities.default(), ()),
                ],
                Capabilities.camel(),
                (_a_string, _a_2),
            ),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
    ]

    def test_binary_op(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
    ):
        got_code, got_namespace, _, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
        )
        assert isinstance(got_code, result.Ok)
        assert expected._python_value == got_code.value._python_value
        assert set(expected._dependencies) == set(got_code.value._dependencies)
        assert expected._metadata == got_code.value._metadata
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace


class TestBoolOp:
    test_data: ClassVar = [
        dict(
            testcase_name="or",
            code="True or False",
            expected=value.CaMeLTrue(Capabilities.default(), ()),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="and",
            code="True and False",
            expected=value.CaMeLFalse(Capabilities.default(), (_a_true,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="or_eval_true",
            code="True or (a := 'a')",
            expected=value.CaMeLTrue(Capabilities.default(), ()),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="or_eval_false",
            code="False or (a := 'a')",
            expected=_a_string.new_with_dependencies((_a_false,)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_string}),
        ),
        dict(
            testcase_name="three_elements",
            code="True and True and False",
            expected=value.CaMeLFalse(
                Capabilities.default(),
                (_a_true.new_with_dependencies((_a_true,)),),
            ),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
    ]

    def test_binary_op(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
    ):
        got_code, got_namespace, _, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
        )
        assert isinstance(got_code, result.Ok)
        assert expected == got_code.value
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace


class TestCompareOp:
    test_data: ClassVar = [
        *[
            dict(
                testcase_name=op,
                code=f"1 {op} 2",
                expected=(
                    value.CaMeLTrue(Capabilities.camel(), (_a_1, _a_2))
                    if eval(f"1 {op} 2")
                    else value.CaMeLFalse(Capabilities.camel(), (_a_1, _a_2))
                ),
                namespace=ns.Namespace(),
                expected_namespace=ns.Namespace(),
            )
            for op in ["!=", "=="]
        ],
        *[
            dict(
                testcase_name=op,
                code=f"1 {op} 2",
                expected=(
                    value.CaMeLTrue(Capabilities.camel(), (_a_1, _a_2))
                    if eval(f"1 {op} 2")
                    else value.CaMeLFalse(Capabilities.camel(), (_a_1, _a_2))
                ),
                namespace=ns.Namespace(),
                expected_namespace=ns.Namespace(),
            )
            for op in ["<", "<=", ">", ">="]
        ],
        dict(
            testcase_name="is",
            code="1 is 1",
            expected=value.CaMeLTrue(Capabilities.camel(), (_a_1, _a_1)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="is not",
            code="1 is not None",
            expected=value.CaMeLTrue(Capabilities.camel(), (_a_1, _a_none)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="in",
            code="1 in [1]",
            expected=value.CaMeLTrue(Capabilities.camel(), (_a_list, _a_1, _a_1)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
        dict(
            testcase_name="not in",
            code="1 not in [1]",
            expected=value.CaMeLFalse(Capabilities.camel(), (_a_list, _a_1, _a_1)),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
        ),
    ]

    def test_compare_op(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
    ):
        got_code, got_namespace, _, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
        )
        assert isinstance(got_code, result.Ok)
        assert expected == got_code.value
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace
