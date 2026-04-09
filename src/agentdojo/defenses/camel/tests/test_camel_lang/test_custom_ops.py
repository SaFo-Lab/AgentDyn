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
from datetime import datetime, timedelta

from camel.interpreter import interpreter, result
from camel.interpreter import namespace as ns
from camel.security_policy import NoSecurityPolicyEngine


def test_datetime_ops():
    code = """\
dt1 = datetime(2023, 10, 26, 10, 0, 0)
dt2 = datetime(2023, 10, 27, 12, 30, 0)
dt2 - dt1
"""
    namespace = ns.Namespace.with_builtins()
    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )

    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == datetime(2023, 10, 27, 12, 30, 0) - datetime(2023, 10, 26, 10, 0, 0)


def test_radd_ops():
    code = """\
dt = datetime(2023, 10, 26, 10, 0, 0)
td = timedelta(days=3)
dt + td
"""
    namespace = ns.Namespace.with_builtins()
    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )

    assert isinstance(got_code, result.Ok)
    assert got_code.value.raw == datetime(2023, 10, 26, 10, 0, 0) + timedelta(days=3)
