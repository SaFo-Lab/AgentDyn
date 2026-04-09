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

from agentdojo import functions_runtime, task_suite
from agentdojo.default_suites.v1.tools import email_client

from camel import security_policy
from camel.interpreter import interpreter, result
from camel.interpreter import namespace as ns
from camel.pipeline_elements import agentdojo_function


def test_send_email():
    suite = task_suite.get_suite("v1.1.1", "workspace")
    env = suite.load_and_inject_default_environment({})
    runtime = functions_runtime.FunctionsRuntime(suite.tools)
    builtins_namespace = ns.Namespace.with_builtins()
    namespace = builtins_namespace.add_variables(
        agentdojo_function.make_agentdojo_namespace(builtins_namespace, runtime, env)
    )
    code = "send_email(['test@example.com'], 'test subject', body='test body')"
    got_code, _, got_tool_calls, _ = interpreter.camel_eval(
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
    email: email_client.Email = got_code.value.raw
    assert email.recipients == ["test@example.com"]
    assert email.subject == "test subject"
    assert email.body == "test body"
    assert len(got_tool_calls) == 1
    assert got_tool_calls[0].function == "send_email"
    assert got_tool_calls[0].args == {
        "recipients": ["test@example.com"],
        "subject": "test subject",
        "body": "test body",
    }
