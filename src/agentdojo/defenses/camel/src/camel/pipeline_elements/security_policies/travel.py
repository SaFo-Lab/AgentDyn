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

from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment

from camel import security_policy
from camel.capabilities import can_readers_read_value, get_all_readers, is_trusted
from camel.interpreter.value import CaMeLDict, CaMeLList, CaMeLNone, CaMeLStr, CaMeLValue
from camel.pipeline_elements.security_policies.agentdojo_security_policies import (
    AgentDojoSecurityPolicyEngine,
    make_trusted_fields_policy,
)
from camel.security_policy import Allowed, Denied, SecurityPolicyResult

workspace_travel_no_side_effect_tools = {
    # User functions
    "get_user_information",
    # Hotel functions
    "get_all_hotels_in_city",
    "get_hotels_prices",
    "get_rating_reviews_for_hotels",
    "get_hotels_address",
    # Restaurant functions
    "get_all_restaurants_in_city",
    "get_cuisine_type_for_restaurants",
    "get_restaurants_address",
    "get_rating_reviews_for_restaurants",  # get_rating_reviews_for_restaurants
    "get_dietary_restrictions_for_all_restaurants",
    "get_contact_information_for_restaurants",
    "get_price_for_restaurants",
    "check_restaurant_opening_hours",
    # Car rental functions
    "get_all_car_rental_companies_in_city",
    "get_car_types_available",
    "get_rating_reviews_for_car_rental",
    "get_car_fuel_options",
    "get_car_rental_address",
    "get_car_price_per_day",
    # Calendar functions
    "search_calendar_events",
    "get_day_calendar_events",
    # For getting flights
    "get_flight_information",
}


class TravelSecurityPolicyEngine(AgentDojoSecurityPolicyEngine[TravelEnvironment]):
    def __init__(self, env: TravelEnvironment) -> None:
        super().__init__(env)
        self.policies = [
            ("send_email", self.send_email_policy),
            ("create_calendar_event", self.create_calendar_event_policy),
            ("cancel_calendar_event", self.cancel_calendar_event_policy),
            (
                "reserve_hotel",
                make_trusted_fields_policy(("hotel", "start_day", "end_day")),
            ),
            (
                "reserve_car_rental",
                make_trusted_fields_policy(("restaurant", "start_time")),
            ),
            (
                "reserve_restaurant",
                make_trusted_fields_policy(("company", "start_time", "end_time")),
            ),
        ]
        self.no_side_effect_tools = security_policy.NO_SIDE_EFFECT_TOOLS | workspace_travel_no_side_effect_tools

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

        attachments_readers, _ = get_all_readers(attachments)
        if not can_readers_read_value(recipients_set, attachments):
            return Denied(
                f"Attachments are not public or shared with the recipients: recipients={recipients_set}, readers={attachments_readers}."
            )
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
        raise NotImplementedError()

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
