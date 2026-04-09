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
    a: int

    def __gt__(self, other):
        if not isinstance(other, AClass):
            return NotImplemented
        return self.a > other.a

    def __lt__(self, other):
        if not isinstance(other, AClass):
            return NotImplemented
        return self.a < other.a


_a_class = value.CaMeLClass("AClass", AClass, Capabilities.default(), (), {}, is_totally_ordered=True)
_a_1 = value.CaMeLInt(1, Capabilities.default(), ())
_a_2 = value.CaMeLInt(2, Capabilities.default(), ())

_a_1_instance = _a_class.init(ns.Namespace(), _a_1)
_a_2_instance = _a_class.init(ns.Namespace(), _a_2)


class TestClasses:
    test_data: ClassVar = [
        dict(
            testcase_name="simple_class_instantiation",
            code="""
a = AClass(1)
""",
            expected=_a_class.init(ns.Namespace(), 1),
            namespace=ns.Namespace({"AClass": _a_class}),
            expected_namespace=ns.Namespace({"AClass": _a_class, "a": _a_class.init(ns.Namespace(), 1)}),
        ),
        dict(
            testcase_name="simple_class_comparison",
            code="""
a > b
""",
            expected=value.CaMeLTrue(Capabilities.camel(), (_a_2_instance, _a_1_instance)),
            namespace=ns.Namespace({"a": _a_2_instance, "b": _a_1_instance}),
            expected_namespace=None,
        ),
    ]

    def test_classes(
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
