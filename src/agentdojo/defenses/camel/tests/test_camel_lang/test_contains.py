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

from camel.interpreter import interpreter, result
from camel.interpreter.namespace import Namespace
from camel.security_policy import NoSecurityPolicyEngine


def test_contains_list():
    code = "1 in [1, 2, 3]"

    namespace = Namespace()

    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == True


def test_not_contains_list():
    code = "1 in [2, 3]"

    namespace = Namespace()

    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == False


def test_contains_str():
    code = "'a' in 'abc'"

    namespace = Namespace()

    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == True


def test_not_contains_str():
    code = "'a' in 'bcd'"

    namespace = Namespace()

    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == False


def test_contains_dict():
    code = "1 in {1: 'a'}"

    namespace = Namespace()

    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == True


def test_not_contains_dict():
    code = "1 in {2: 'a'}"

    namespace = Namespace()

    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == False
