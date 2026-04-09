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


_a_string = value.CaMeLStr.from_raw("a", Capabilities.default(), ())
_a_0 = value.CaMeLInt(0, Capabilities.default(), ())
_a_1 = value.CaMeLInt(1, Capabilities.default(), ())
_a_2 = value.CaMeLInt(2, Capabilities.default(), ())
_a_list = value.CaMeLList([_a_0, _a_1, _a_2], Capabilities.default(), ())
_a_tuple = value.CaMeLTuple((_a_0, _a_1, _a_2), Capabilities.default(), ())
_a_dict = value.CaMeLDict({_a_string: _a_1}, Capabilities.default(), ())
_a_true = value.CaMeLTrue(Capabilities.default(), ())
_a_false = value.CaMeLFalse(Capabilities.default(), ())
_a_none = value.CaMeLNone(Capabilities.default(), ())
_tuple_list = value.CaMeLList(
    [
        value.CaMeLTuple(
            (
                value.CaMeLInt(1, Capabilities.default(), ()),
                value.CaMeLStr.from_raw("a", Capabilities.default(), ()),
            ),
            Capabilities.default(),
            (),
        ),
        value.CaMeLTuple(
            (
                value.CaMeLInt(2, Capabilities.default(), ()),
                value.CaMeLStr.from_raw("b", Capabilities.default(), ()),
            ),
            Capabilities.default(),
            (),
        ),
        value.CaMeLTuple(
            (
                value.CaMeLInt(3, Capabilities.default(), ()),
                value.CaMeLStr.from_raw("c", Capabilities.default(), ()),
            ),
            Capabilities.default(),
            (),
        ),
    ],
    Capabilities.default(),
    (),
)
_value_specifier_str = value.CaMeLStr(
    [
        value._CaMeLChar(".", Capabilities.default(), ()),
        value._CaMeLChar("1", Capabilities.default(), ()),
        value._CaMeLChar("f", Capabilities.default(), ()),
    ],
    Capabilities.camel(),
    (),
)


class TestLiterals:
    test_data: ClassVar = [
        dict(
            testcase_name="simple",
            code="'a'",
            expected=_a_string,
            namespace=ns.Namespace(),
            expected_namespace=None,
        ),
        dict(
            testcase_name="f-string",
            code="f'a {1:.1f}'",
            # "a 1.0"
            expected=value.CaMeLStr(
                [
                    value._CaMeLChar("a", Capabilities.default(), ()),
                    value._CaMeLChar(" ", Capabilities.default(), ()),
                    value._CaMeLChar("1", Capabilities.camel(), (_a_1, _value_specifier_str)),
                    value._CaMeLChar(".", Capabilities.camel(), (_a_1, _value_specifier_str)),
                    value._CaMeLChar("0", Capabilities.camel(), (_a_1, _value_specifier_str)),
                ],
                Capabilities.camel(),
                (),
            ),
            namespace=ns.Namespace(),
            expected_namespace=None,
        ),
        dict(
            testcase_name="f-string with value",
            code="f'a {b}'",
            expected=value.CaMeLStr(
                [
                    value._CaMeLChar("a", Capabilities.default(), ()),
                    value._CaMeLChar(" ", Capabilities.default(), ()),
                    value._CaMeLChar("1", Capabilities.camel(), (_a_1,)),
                ],
                Capabilities.camel(),
                (),
            ),
            namespace=ns.Namespace({"b": _a_1}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="f-string with string",
            code="f'a{b}'",
            expected=value.CaMeLStr(
                [
                    value._CaMeLChar("a", Capabilities.default(), ()),
                    value._CaMeLChar("a", Capabilities.camel(), (_a_string,)),
                ],
                Capabilities.camel(),
                (),
            ),
            namespace=ns.Namespace({"b": _a_string}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="tuple",
            code="(0, 1, b)",
            expected=_a_tuple,
            namespace=ns.Namespace({"b": _a_2}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="list",
            code="[0, 1, b]",
            expected=_a_list,
            namespace=ns.Namespace({"b": _a_2}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="list with unpacking",
            code="['a', b, *l]",
            expected=value.CaMeLList(
                [_a_string, _a_2, *_a_list._python_value],
                Capabilities.default(),
                (_a_list,),
            ),
            namespace=ns.Namespace({"b": _a_2, "l": _a_list}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="set",
            code="{'a', b}",
            expected=value.CaMeLSet({_a_string, _a_1}, Capabilities.default(), ()),
            namespace=ns.Namespace({"b": _a_1}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="dict",
            code="{'a': b}",
            expected=_a_dict,
            namespace=ns.Namespace({"b": _a_1}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="expanded_dict",
            code="{'d': 0, **{'a': b}}",
            expected=value.CaMeLDict(
                {
                    value.CaMeLStr.from_raw("d", Capabilities.default(), ()): _a_0,
                    value.CaMeLStr.from_raw("a", Capabilities.default(), ()): _a_1,
                },
                Capabilities.default(),
                (_a_dict,),
            ),
            namespace=ns.Namespace({"b": _a_1}),
            expected_namespace=None,
        ),
    ]

    def test_literals(
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


class TestComprehensions:
    test_data: ClassVar = [
        dict(
            testcase_name="list comprehension",
            code="[-x for x in [0, 1, 2]]",
            expected=value.CaMeLList(
                [
                    value.CaMeLInt(
                        0,
                        Capabilities.default(),
                        (value.CaMeLInt(0, Capabilities.default(), ()),),
                    ),
                    value.CaMeLInt(
                        -1,
                        Capabilities.default(),
                        (value.CaMeLInt(1, Capabilities.default(), ()),),
                    ),
                    value.CaMeLInt(
                        -2,
                        Capabilities.default(),
                        (value.CaMeLInt(2, Capabilities.default(), ()),),
                    ),
                ],
                Capabilities.camel(),
                (_a_list,),
            ),
            namespace=ns.Namespace(),
            expected_namespace=None,
        ),
        dict(
            testcase_name="list comprehension tuple assignment",
            code="[(x, y) for x, y in [(1, 'a'), (2, 'b'), (3, 'c')]]",
            expected=_tuple_list.new_with_metadata(Capabilities.camel()).new_with_dependencies((_tuple_list,)),
            namespace=ns.Namespace.with_builtins(),
            expected_namespace=None,
        ),
        dict(
            testcase_name="set comprehension",
            code="{x for x in [1, 1, 2]}",
            expected=value.CaMeLSet(
                {_a_1, _a_2},
                Capabilities.camel(),
                (value.CaMeLList([_a_1, _a_1, _a_2], Capabilities.default(), ()),),
            ),
            namespace=ns.Namespace(),
            expected_namespace=None,
        ),
        dict(
            testcase_name="dict comprehension",
            code="{x: y for x, y in [(1, 'a'), (2, 'b'), (3, 'c')]}",
            expected=value.CaMeLDict(
                {
                    value.CaMeLInt(1, Capabilities.default(), ()): value.CaMeLStr.from_raw(
                        "a", Capabilities.default(), ()
                    ),
                    value.CaMeLInt(2, Capabilities.default(), ()): value.CaMeLStr.from_raw(
                        "b", Capabilities.default(), ()
                    ),
                    value.CaMeLInt(3, Capabilities.default(), ()): value.CaMeLStr.from_raw(
                        "c", Capabilities.default(), ()
                    ),
                },
                Capabilities.camel(),
                (_tuple_list,),
            ),
            namespace=ns.Namespace.with_builtins(),
            expected_namespace=None,
        ),
        dict(
            testcase_name="list comprehension with if",
            code="[x for x in [0, 1, 2] if x != 0]",
            expected=value.CaMeLList(
                [
                    value.CaMeLInt(1, Capabilities.default(), ()),
                    value.CaMeLInt(2, Capabilities.default(), ()),
                ],
                Capabilities.camel(),
                (_a_list,),  # TODO: this should actually have a dependency on the x != 0 expression
            ),
            namespace=ns.Namespace(),
            expected_namespace=None,
        ),
        dict(
            testcase_name="list comprehension with assignment",
            code="[(y := x) for x in [0, 1, 2]]",
            expected=value.CaMeLList(
                [
                    value.CaMeLInt(0, Capabilities.default(), ()),
                    value.CaMeLInt(1, Capabilities.default(), ()),
                    value.CaMeLInt(2, Capabilities.default(), ()),
                ],
                Capabilities.camel(),
                (_a_list,),
            ),
            namespace=ns.Namespace({"y": value.CaMeLInt(2, Capabilities.default(), ())}),
            expected_namespace=None,
        ),
        dict(
            testcase_name="nested list comprehension",
            code="[x for y in [[1, 2, 3], [4, 5, 6]] for x in y]",
            expected=value.CaMeLList(
                [
                    value.CaMeLInt(1, Capabilities.default(), ()),
                    value.CaMeLInt(2, Capabilities.default(), ()),
                    value.CaMeLInt(3, Capabilities.default(), ()),
                    value.CaMeLInt(4, Capabilities.default(), ()),
                    value.CaMeLInt(5, Capabilities.default(), ()),
                    value.CaMeLInt(6, Capabilities.default(), ()),
                ],
                Capabilities.camel(),
                (
                    value.CaMeLList(
                        [
                            value.CaMeLInt(1, Capabilities.default(), ()),
                            value.CaMeLInt(2, Capabilities.default(), ()),
                            value.CaMeLInt(3, Capabilities.default(), ()),
                        ],
                        Capabilities.default(),
                        (),
                    ),
                    value.CaMeLList(
                        [
                            value.CaMeLInt(4, Capabilities.default(), ()),
                            value.CaMeLInt(5, Capabilities.default(), ()),
                            value.CaMeLInt(6, Capabilities.default(), ()),
                        ],
                        Capabilities.default(),
                        (),
                    ),
                    value.CaMeLList(
                        [
                            value.CaMeLList(
                                [
                                    value.CaMeLInt(1, Capabilities.default(), ()),
                                    value.CaMeLInt(2, Capabilities.default(), ()),
                                    value.CaMeLInt(3, Capabilities.default(), ()),
                                ],
                                Capabilities.default(),
                                (),
                            ),
                            value.CaMeLList(
                                [
                                    value.CaMeLInt(4, Capabilities.default(), ()),
                                    value.CaMeLInt(5, Capabilities.default(), ()),
                                    value.CaMeLInt(6, Capabilities.default(), ()),
                                ],
                                Capabilities.default(),
                                (),
                            ),
                        ],
                        Capabilities.default(),
                        (),
                    ),
                ),
            ),
            namespace=ns.Namespace(),
            expected_namespace=None,
        ),
    ]

    def test_comprehensions(
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
