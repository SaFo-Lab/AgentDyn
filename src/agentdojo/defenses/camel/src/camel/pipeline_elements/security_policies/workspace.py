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

from agentdojo.default_suites.v1.workspace.task_suite import (
    WorkspaceEnvironment,
)

from camel import security_policy
from camel.capabilities import can_readers_read_value, get_all_readers, is_trusted
from camel.interpreter.value import CaMeLDict, CaMeLList, CaMeLNone, CaMeLStr, CaMeLValue
from camel.pipeline_elements.security_policies.agentdojo_security_policies import (
    AgentDojoSecurityPolicyEngine,
)
from camel.security_policy import Allowed, Denied, SecurityPolicyResult

workspace_no_side_effect_tools = {
    # Email tools
    "get_unread_emails",
    "get_sent_emails",
    "get_received_emails",
    "get_draft_emails",
    "search_emails",
    "search_contacts_by_name",
    "search_contacts_by_email",
    # Calendar tools
    "get_current_day",
    "search_calendar_events",
    "get_day_calendar_events",
    # Drive tools
    "search_files_by_filename",
    "get_file_by_id",
    "list_files",
    "search_files",
}


class WorkspaceSecurityPolicyEngine(AgentDojoSecurityPolicyEngine[WorkspaceEnvironment]):
    def __init__(self, env: WorkspaceEnvironment) -> None:
        super().__init__(env)
        self.policies = [
            ("send_email", self.send_email_policy),
            ("delete_email", self.delete_email_policy),
            ("create_calendar_event", self.create_calendar_event_policy),
            ("cancel_calendar_event", self.cancel_calendar_event_policy),
            ("reschedule_calendar_event", self.reschedule_calendar_event_policy),
            (
                "add_calendar_event_participants",
                self.add_calendar_event_participants_policy,
            ),
            ("create_file", self.create_file_policy),
            ("delete_file", self.delete_file_policy),
            ("share_file", self.share_file_policy),
            ("append_to_file", self.append_to_file_policy),
        ]
        self.no_side_effect_tools = security_policy.NO_SIDE_EFFECT_TOOLS | workspace_no_side_effect_tools

    def send_email_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        recipients: CaMeLList[CaMeLStr] = kwargs["recipients"]  # type: ignore
        if "cc" in kwargs and isinstance(kwargs["cc"], CaMeLList):
            recipients = recipients.add(kwargs["cc"])
        if "bcc" in kwargs and isinstance(kwargs["bcc"], CaMeLList):
            recipients = recipients.add(kwargs["bcc"])

        # Email address comes directly from the user
        if is_trusted(recipients):
            return Allowed()
        recipients_set = set(recipients.raw)

        # Check body
        body: CaMeLStr = kwargs["body"]  # type: ignore
        body_readers, _ = get_all_readers(body)
        if not can_readers_read_value(recipients_set, body):
            return Denied(
                f"Body is not public or shared with the recipients: recipients={recipients_set}, readers={body_readers}."
            )

        # Check subject
        subject: CaMeLStr = kwargs["subject"]  # type: ignore
        subject_readers, _ = get_all_readers(subject)
        if not can_readers_read_value(recipients_set, subject):
            return Denied(
                f"Subject is not public or shared with the recipients: recipients={recipients_set}, readers={subject_readers}."
            )

        # Check attachments
        if "attachments" not in kwargs:
            return Allowed()
        attachments: CaMeLList[CaMeLDict[CaMeLStr, CaMeLStr]] = kwargs["attachments"]  # type: ignore
        if isinstance(attachments, CaMeLNone):
            return Allowed()
        # TODO: one can add more detailed checks by fetching the files from the drive
        file_attachments = [attachment for attachment in attachments.raw if attachment["type"] == "file"]
        for attachment in file_attachments:
            file = self._env.cloud_drive.files[attachment["file_id"]]
            file_readers = set(file.shared_with.keys())
            if not recipients_set.issubset(file_readers):
                return Denied(
                    f"The file with id {file.id_} is not shared with the email recipients: recipients={set(recipients_set)}, file readers = {file_readers}"
                )

        attachments_readers, _ = get_all_readers(attachments)
        if not can_readers_read_value(recipients_set, attachments):
            return Denied(
                f"Attachments are not public or shared with the recipients: recipients={recipients_set}, readers={attachments_readers}."
            )
        return Allowed()

    def delete_email_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - email_id: str
        # Readers:
        # - None
        # Rationale:
        # The email_id should come from an email that was requested by the user directly
        email_id = kwargs["email_id"]
        if is_trusted(email_id):
            return Denied("Email to be deleted does not come directly from user")
        return Allowed()

    def create_calendar_event_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - title: str
        # - description: str
        # - location: str
        # - start_time: str -> parsed to datetime
        # - end_time: str -> parsed to datetime
        # Readers:
        # - participants: list[str]
        # Rationale:
        # Title, description, location, start_time, end_time should be readable by the participants or all participants should come from user
        participants: CaMeLList[CaMeLStr] = kwargs["participants"]  # type: ignore
        if is_trusted(participants):
            return Allowed()
        participants_set = set(participants.raw)
        if not can_readers_read_value(participants_set, kwargs["title"]):
            return Denied("The title is not public or shared with the recipients")
        if not can_readers_read_value(participants_set, kwargs["description"]):
            return Denied("The description is not public or shared with the recipients")
        if not can_readers_read_value(participants_set, kwargs["location"]):
            return Denied("The location is not public or shared with the recipients")
        if not can_readers_read_value(participants_set, kwargs["end_time"]):
            return Denied("The end_time is not public or shared with the recipients")
        if not can_readers_read_value(participants_set, kwargs["start_time"]):
            return Denied("The start_time is not public or shared with the recipients")
        return Allowed()

    def cancel_calendar_event_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - event_id: str
        # Readers:
        # None
        # Rationale:
        # The event_id should come from an email that was requested by the user directly
        event_id = kwargs["event_id"]
        if not is_trusted(event_id):
            return Denied("Event to be deleted does not come directly from user")
        return Allowed()

    def reschedule_calendar_event_policy(
        self, tool_name: str, kwargs: Mapping[str, CaMeLValue]
    ) -> SecurityPolicyResult:
        # Data:
        # - new_start_time: str -> parsed to datetime
        # - new_end_time: str | None -> parsed to datetime
        # Readers:
        # - the participants of the event (event_id)
        # Rationale:
        # New start and end times should be readable by the event participants
        event_id = kwargs["event_id"].raw
        event_participants = set(self._env.calendar.events[event_id].participants)

        new_start_time = kwargs["new_start_time"]
        new_start_time_readers, _ = get_all_readers(new_start_time)
        if not can_readers_read_value(event_participants, new_start_time):
            return Denied(
                f"Start time is not readable by the participants of the event: participants={event_participants} time_readers={new_start_time_readers}"
            )

        new_end_time = kwargs["new_end_time"]
        if isinstance(new_end_time, CaMeLNone):
            return Allowed()
        new_end_time_readers, _ = get_all_readers(new_end_time)
        if not can_readers_read_value(event_participants, new_end_time):
            return Denied(
                f"End time is not readable by the participants of the event: participants={event_participants} time_readers={new_end_time_readers}"
            )

        return Allowed()

    def add_calendar_event_participants_policy(
        self, tool_name: str, kwargs: Mapping[str, CaMeLValue]
    ) -> SecurityPolicyResult:
        # Data:
        # - event_id: str
        # Readers:
        # - participants: str
        # Rationale: Participants should be specified by the user, or event data must be public.
        # However, we can't fetch whether the event data is public, so the security policy only
        # passes if the participants come from the user.
        event_id = kwargs["event_id"]
        if not is_trusted(event_id):
            return Denied("Event to be deleted does not come directly from user")
        return Allowed()

    def create_file_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - filename: str
        # - content: str
        # Readers:
        # - None
        # Rationale: this does have side-effects, but this function only makes the content
        # accessible to the user. The model needs to call `share_file` to make this data visible.
        return Allowed()

    def delete_file_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - file_id: str
        # Readers:
        # - None
        # Rationale: the file_id should come directly from the user
        event_id = kwargs["file_id"]
        if not is_trusted(event_id):
            return Denied("File ID to be deleted does not come directly from user")
        return Allowed()

    def share_file_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - file_id: str
        # - permission: Enum
        # Readers:
        # - email: str
        # Rationale:
        email = kwargs["email"]
        if not is_trusted(email):
            return Denied("Email to share the document with does not come directly from user")
        # TODO: we could fetch the file from the Drive and we could then match if there is a corresponding
        # file in memory in the interpreter, and check if the data in the document is private. However this
        # seems challenging
        return Allowed()

    def append_to_file_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        # Data:
        # - file_id: str
        # - content: str
        # Readers:
        # - file readers: str
        #
        # Rationale:
        # Can't write on the file data that are not shared with the people with access to the file
        file_id = kwargs["file_id"].raw
        file = self._env.cloud_drive.files[file_id]
        file_readers = set(file.shared_with.keys())
        if not can_readers_read_value(file_readers, kwargs["content"]):
            return Denied(
                "The file readers do not have access to the content that is about to be appended to the file."
            )
        return Allowed()
