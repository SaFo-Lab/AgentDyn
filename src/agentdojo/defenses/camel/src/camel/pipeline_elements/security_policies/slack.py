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

from collections.abc import Mapping

from agentdojo.default_suites.v1.slack.task_suite import SlackEnvironment

from camel.capabilities import can_readers_read_value, is_public, is_trusted
from camel.interpreter.value import CaMeLValue
from camel.pipeline_elements.security_policies.agentdojo_security_policies import (
    AgentDojoSecurityPolicyEngine,
    make_trusted_fields_policy,
)
from camel.security_policy import NO_SIDE_EFFECT_TOOLS, Allowed, Denied, SecurityPolicyResult

workspace_no_side_effect_tools = {
    "get_channels",
    "read_channel_messages",
    "read_inbox",
    "get_users_in_channel",
}


class SlackSecurityPolicyEngine(AgentDojoSecurityPolicyEngine[SlackEnvironment]):
    def __init__(self, env: SlackEnvironment) -> None:
        super().__init__(env)
        self.policies = [
            ("add_user_to_channel", make_trusted_fields_policy(("user", "channel"))),
            ("send_direct_message", self.send_direct_message_policy),
            ("send_channel_message", self.send_channel_message_policy),
            (
                "invite_user_to_slack",
                make_trusted_fields_policy(("user", "user_email")),
            ),
            ("remove_user_from_slack", make_trusted_fields_policy(("user",))),
            ("post_webpage", self.post_webpage_policy),
            ("get_webpage", self.get_webpage_policy),
        ]
        self.no_side_effect_tools = NO_SIDE_EFFECT_TOOLS | workspace_no_side_effect_tools

    def send_direct_message_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - body: str
        # Readers:
        # - recipient: str
        # Rationale: Message should be readable by the recipient (or recipient must come from user)
        recipient = kwargs["recipient"]
        if is_trusted(recipient):
            return Allowed()
        readers = {recipient.raw}
        if can_readers_read_value(readers, kwargs["body"]):
            return Allowed()
        return Denied("The recipient can't read the content of the message.")

    def send_channel_message_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - body: str
        # Readers:
        # - channel: str (can get the channel members)
        # Rationale: Message should be readable by the channel members
        channel = kwargs["channel"].raw
        channel_members = {user for user, channels in self._env.slack.user_channels.items() if channel in channels}
        if can_readers_read_value(channel_members, kwargs["body"]):
            return Allowed()
        return Denied("The recipient can't read the content of the message.")

    def post_webpage_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - url: str
        # - content: str
        # Readers:
        # - Public
        # Rationale: Only allow if content and url are public
        if not is_public(kwargs["url"]):
            return Denied("The url is not public")
        if not is_public(kwargs["content"]):
            return Denied("The content is not public")
        return Allowed()

    def get_webpage_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - url: str
        # Readers:
        # - Public
        # Rationale: Only allow if url is public
        if not is_public(kwargs["url"]):
            return Denied("The url is not public")
        return Allowed()
