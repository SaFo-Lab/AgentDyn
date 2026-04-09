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


class AClass:
    def __init__(self, a: str) -> None:
        self.a = a

    def __repr__(self) -> str:
        return f"AClass(a={self.a})"

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, AClass):
            return False
        return self.a == value.a


_a_string = value.CaMeLStr.from_raw("a", Capabilities.default(), ())
_b_string = value.CaMeLStr.from_raw("b", Capabilities.default(), ())
_a_0 = value.CaMeLInt(0, Capabilities.default(), ())
_a_1 = value.CaMeLInt(1, Capabilities.default(), ())
_a_2 = value.CaMeLInt(2, Capabilities.default(), ())
_a_class = value.CaMeLClass("AClass", AClass, Capabilities.default(), (), {})
_a_class_instance = _a_class.init(ns.Namespace(), _a_1)
_a_list = value.CaMeLList([_a_1], Capabilities.default(), ())
_a_none = value.CaMeLNone(Capabilities.default(), ())


class TestAssignments:
    test_data: ClassVar = [
        dict(
            testcase_name="simple_assignment",
            code="a = 'a'",
            expected=_a_string,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_string}),
        ),
        dict(
            testcase_name="aug_assign",
            code="a += 1",
            expected=_a_none,
            namespace=ns.Namespace({"a": _a_1}),
            expected_namespace=ns.Namespace({"a": value.CaMeLInt(2, Capabilities.camel(), (_a_1, _a_1))}),
        ),
        dict(
            testcase_name="aug_assign_index",
            code="a[0] += 1",
            expected=_a_none,
            namespace=ns.Namespace({"a": _a_list}),
            expected_namespace=ns.Namespace(
                {
                    "a": value.CaMeLList(
                        [
                            value.CaMeLInt(
                                2,
                                Capabilities.camel(),
                                (_a_1.new_with_dependencies((_a_list, _a_0)), _a_1),
                            )
                        ],
                        Capabilities.default(),
                        (),
                    )
                }
            ),
        ),
        dict(
            testcase_name="aug_assign_class",
            code="a.a += 1",
            expected=_a_none,
            namespace=ns.Namespace({"a": _a_class_instance}),
            expected_namespace=ns.Namespace(
                {
                    "a": _a_class.init(
                        ns.Namespace(),
                        value.CaMeLInt(
                            2,
                            Capabilities.camel(),
                            (_a_1.new_with_dependencies((_a_class_instance,)), _a_1),
                        ),
                    )
                }
            ),
        ),
        dict(
            testcase_name="multi_assignment",
            code="a = b = 'a'",
            expected=_a_string,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_string, "b": _a_string}),
        ),
        dict(
            testcase_name="tuple_unpacking_assignment",
            code="a, b = 'a', 'b'",
            expected=value.CaMeLTuple((_a_string, _b_string), Capabilities.default(), ()),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_string, "b": _b_string}),
        ),
        dict(
            testcase_name="list_unpacking_assignment",
            code="[a, b] = ['a', 'b']",
            expected=value.CaMeLList((_a_string, _b_string), Capabilities.default(), ()),
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_string, "b": _b_string}),
        ),
        dict(
            testcase_name="ann_assignment",
            code="a: str = 'a'",
            expected=_a_string,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_string}),
        ),
        dict(
            testcase_name="index_assignment",
            code="a[1] = 2",
            expected=_a_2,
            namespace=ns.Namespace({"a": value.CaMeLList([_a_1, _a_1, _a_1], Capabilities.default(), ())}),
            expected_namespace=ns.Namespace({"a": value.CaMeLList([_a_1, _a_2, _a_1], Capabilities.default(), ())}),
        ),
        dict(
            testcase_name="attr_assignment",
            code="a.a = 2",
            expected=_a_2,
            namespace=ns.Namespace({"a": _a_class_instance}),
            expected_namespace=ns.Namespace(
                {"a": _a_class.init(ns.Namespace(), value.CaMeLInt(2, Capabilities.default(), ()))}
            ),
        ),
    ]

    def test_assignments(
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
            assert expected_namespace == got_namespace
        else:
            assert namespace == got_namespace
