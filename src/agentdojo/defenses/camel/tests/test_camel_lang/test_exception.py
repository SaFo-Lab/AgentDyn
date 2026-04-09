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
from camel.interpreter import namespace as ns
from camel.security_policy import NoSecurityPolicyEngine


def test_exception():
    code = "raise ValueError('Error!')"
    namespace = ns.Namespace.with_builtins()
    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Error)
    assert ValueError == type(got_code.error.exception)
    assert "Error!" == str(got_code.error.exception)
    assert got_namespace == namespace
