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

"""Security policies for tools."""

import dataclasses
import fnmatch
from collections.abc import Iterable, Mapping
from typing import Protocol

from camel.capabilities import get_all_readers, is_public, readers
from camel.interpreter.value import CaMeLValue


@dataclasses.dataclass(frozen=True)
class Allowed:
    """Allowed access to the tool."""

    ...


@dataclasses.dataclass(frozen=True)
class Denied:
    """Denied access to the tool."""

    reason: str
    """Reason for denial."""


SecurityPolicyResult = Allowed | Denied


class SecurityPolicy(Protocol):
    def __call__(
        self,
        tool_name: str,
        kwargs: Mapping[str, CaMeLValue],
    ) -> SecurityPolicyResult: ...


NO_SIDE_EFFECT_TOOLS = {
    # Query AI assistant function
    "query_ai_assistant",
}


def base_security_policy(
    tool_name: str,
    kwargs: Mapping[str, CaMeLValue],
    no_side_effect_tools: set[str],
) -> SecurityPolicyResult:
    """Base security policy. If *anything* in the arguments is *not* public, then access is denied."""
    r = [get_all_readers(data) for data in kwargs.values()]
    if any(reader != readers.Public() for reader in r) and tool_name not in no_side_effect_tools:
        return Denied("Data is not public.")
    return Allowed()


class SecurityPolicyDeniedError(Exception): ...


class SecurityPolicyEngine(Protocol):
    policies: list[tuple[str, SecurityPolicy]]
    no_side_effect_tools: set[str]

    def check_policy(
        self,
        tool_name: str,
        kwargs: Mapping[str, CaMeLValue],
        dependencies: Iterable[CaMeLValue],
    ) -> SecurityPolicyResult:
        """Checks if the tool is allowed to be executed with the given data.

        Policies in `POLICIES` are evaluated in order. If any evaluates to Allowed(),
        then the tool is executed.
        """
        if tool_name in self.no_side_effect_tools:
            return Allowed()
        non_public_variables = [d.raw for d in dependencies if not is_public(d)]
        if len(non_public_variables) > 0:
            return Denied(f"{tool_name} is state-changing and depends on private values {non_public_variables}.")
        for policy_name, policy in self.policies:
            if fnmatch.fnmatch(tool_name, policy_name):
                return policy(tool_name, kwargs)
        return Denied("No security policy matched for tool. Defaulting to denial.")


class NoSecurityPolicyEngine(SecurityPolicyEngine):
    def __init__(self) -> None:
        self.policies = []
        self.no_side_effect_tools = set()

    def check_policy(
        self,
        tool_name: str,
        kwargs: Mapping[str, CaMeLValue],
        dependencies: Iterable[CaMeLValue],
    ) -> SecurityPolicyResult:
        return Allowed()
