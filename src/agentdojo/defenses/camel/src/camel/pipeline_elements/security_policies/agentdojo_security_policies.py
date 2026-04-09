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

from collections.abc import Iterable, Mapping
from typing import Generic, TypeVar

from agentdojo.default_suites.v1.workspace.task_suite import (
    WorkspaceEnvironment,
)
from agentdojo.functions_runtime import TaskEnvironment

from camel.capabilities import is_trusted
from camel.interpreter.value import CaMeLValue
from camel.security_policy import Allowed, Denied, SecurityPolicy, SecurityPolicyEngine, SecurityPolicyResult

_E = TypeVar("_E", bound=TaskEnvironment)


class AgentDojoSecurityPolicyEngine(SecurityPolicyEngine, Generic[_E]):
    def __init__(self, env: _E) -> None:
        self._env = env


class ADNoSecurityPolicyEngine(AgentDojoSecurityPolicyEngine[WorkspaceEnvironment]):
    def __init__(self, env: WorkspaceEnvironment) -> None:
        super().__init__(env)
        self.policies = []
        self.no_side_effect_tools = set()

    def check_policy(
        self, tool_name: str, kwargs: Mapping[str, CaMeLValue], dependencies: Iterable[CaMeLValue]
    ) -> SecurityPolicyResult:
        return Allowed()


def make_trusted_fields_policy(fields: tuple[str, ...]) -> SecurityPolicy:
    def policy(tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        for arg_name in fields:
            arg = kwargs[arg_name]
            if not is_trusted(arg):
                return Denied(f"{arg_name} does not come directly from user")
        return Allowed()

    return policy
