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

import enum
import typing
from collections.abc import Iterable
from types import GenericAlias, NoneType
from typing import Any, Generic, TypeAlias, TypeVar

import pydantic
from agentdojo import functions_runtime
from agentdojo.default_suites.v1.tools import banking_client, calendar_client, cloud_drive_client, email_client, slack

from camel.capabilities import Capabilities, readers, sources
from camel.conditional_cache import conditional_lru_cache
from camel.interpreter import namespace as ns
from camel.interpreter import value

_USER_EMAIL = "emma.johnson@bluesparrowtech.com"


InnerSourcesSet: TypeAlias = frozenset[str | sources.SourceEnum]


def _transform_user_email_to_user(users: Iterable[pydantic.EmailStr]) -> InnerSourcesSet:
    return frozenset(participant if participant != _USER_EMAIL else sources.SourceEnum.User for participant in users)


def _get_calendar_event_metadata(
    event: calendar_client.CalendarEvent,
    tool: str,
) -> Capabilities:
    # The organizer must be on the list. So if the organizer is untrusted, then the list is not trusted
    return Capabilities(
        frozenset({sources.Tool(tool, _transform_user_email_to_user(event.participants))}),
        frozenset(event.participants),
    )


def _get_cloud_drive_file_metadata(
    file: cloud_drive_client.CloudDriveFile,
    tool: str,
) -> Capabilities:
    editors = set(u for u, p in file.shared_with.items() if p == cloud_drive_client.SharingPermission.rw) | {file.owner}
    return Capabilities(
        frozenset({sources.Tool(tool, _transform_user_email_to_user(editors))}), frozenset(file.shared_with)
    )


def _get_email_metadata(email: email_client.Email, tool: str) -> Capabilities:
    email_readers = frozenset({email.sender} | set(email.recipients) | set(email.cc) | set(email.bcc))
    # In a real-world application this would include also previous senders on the thread
    # However AgentDojo does not have threads and all emails are individual
    return Capabilities(frozenset({sources.Tool(tool, _transform_user_email_to_user({email.sender}))}), email_readers)


def _get_email_contact_metadata(email_contact: email_client.EmailContact, tool: str) -> Capabilities:
    return Capabilities(
        frozenset({sources.Tool(tool, _transform_user_email_to_user({email_contact.email}))}), readers.Public()
    )


def _get_transaction_metadata(transaction: banking_client.Transaction, tool: str) -> Capabilities:
    transaction_readers = {transaction.sender, transaction.recipient}
    inner_source: set[str | sources.SourceEnum] = (
        {sources.SourceEnum.User} if transaction.sender == "me" else {transaction.sender}
    )
    return Capabilities(frozenset({sources.Tool(tool, frozenset(inner_source))}), frozenset(transaction_readers))


def _get_message_metadata(message: slack.Message, tool: str) -> Capabilities:
    # Ideally here the readers are the member of a channel, however user and auth in AgentDojo's Slack suite
    # does not support this
    return Capabilities(frozenset({sources.Tool(tool, frozenset({message.sender}))}), frozenset({message.recipient}))


_T = TypeVar("_T", bound=value.CaMeLValue)


_TRUSTED_TRAVEL_TOOLS = {
    "check_restaurant_opening_hours",
    "get_all_car_rental_companies_in_city",
    "get_all_hotels_in_city",
    "get_all_restaurants_in_city",
    "get_car_fuel_options",
    "get_car_price_per_day",
    "get_car_rental_address",
    "get_car_types_available",
    "get_contact_information_for_restaurants",
    "get_cuisine_type_for_restaurants",
    "get_dietary_restrictions_for_all_restaurants",
    "get_flight_information",
    "get_hotels_address",
    "get_hotels_prices",
    "get_price_for_restaurants",
    "get_restaurants_address",
}


_UNTRUSTED_TRAVEL_TOOLS = {
    "get_rating_reviews_for_car_rental",
    "get_rating_reviews_for_hotels",
    "get_rating_reviews_for_restaurants",
}

assert _UNTRUSTED_TRAVEL_TOOLS.intersection(_TRUSTED_TRAVEL_TOOLS) == set()


@conditional_lru_cache()
def _get_metadata_for_ad(
    result: _T,
    tool: str,
) -> _T:
    match result.raw:
        case list():
            converted_list = [_get_metadata_for_ad(d, tool) for d in result._python_value]
            return result.new_with_python_value(converted_list).new_with_metadata(
                Capabilities(frozenset({sources.Tool(tool, frozenset())}), readers.Public())
            )
        case str() if tool in {"cancel_calendar_event", "delete_email", "get_current_day", "get_iban"}:
            # Tools that only return a confirmation with the ID of the deleted object, or the tool
            # that returns the current day. None of them is private and they can't propagate a prompt injection
            return result.new_with_metadata(
                Capabilities(
                    frozenset({sources.Tool(tool, frozenset({sources.SourceEnum.TrustedToolSource}))}), readers.Public()
                )
            )
        case str() if tool in {
            "read_file",
            "get_channels",
            "get_users_in_channel",
            "get_webpage",
            "send_money",
            "schedule_transaction",
            "update_scheduled_transaction",
        }:
            # Since the file object does not have any metadata, we need to annotate it as private and untrusted
            # Moreover, in slack channel names and usernames might contain prompt injections, in web as well
            return result.new_with_metadata(Capabilities(frozenset({sources.Tool(tool, frozenset())}), frozenset()))
        case dict() if tool in {"get_user_info", "update_password", "update_user_info", "get_user_information"}:
            # security policy imposes that user info being updated comes from the user (or can't be updated at all)
            return result.new_with_metadata(
                Capabilities(frozenset({sources.Tool(tool, frozenset({sources.SourceEnum.User}))}), frozenset())
            )
        case dict() if tool in _UNTRUSTED_TRAVEL_TOOLS:
            # reviews are public
            return result.new_with_metadata(
                Capabilities(frozenset({sources.Tool(tool, frozenset({sources.SourceEnum.User}))}), readers.Public())
            )
        case dict() | str() if tool in _TRUSTED_TRAVEL_TOOLS:
            # All travel tool outputs are trusted except from reviews, and the data is public
            return result.new_with_metadata(
                Capabilities(
                    frozenset({sources.Tool(tool, frozenset({sources.SourceEnum.TrustedToolSource}))}), readers.Public()
                )
            )
        case str() if tool in {"reserve_restaurant", "reserve_car_rental", "reserve_hotel"}:
            # These strings contain the name of the entity (which we assumed to be trusted), and dates and times of reservation
            # hence, this is private, but trusted as dates and times can't propagate prompt injections
            return result.new_with_metadata(
                Capabilities(
                    frozenset({sources.Tool(tool, frozenset({sources.SourceEnum.TrustedToolSource}))}), frozenset()
                )
            )
        case float() if tool in {"get_balance"}:
            # Balance is private, but it's a float and can't propagate a prompt injection
            return result.new_with_metadata(
                Capabilities(
                    frozenset({sources.Tool(tool, frozenset({sources.SourceEnum.TrustedToolSource}))}), frozenset()
                )
            )
        case NoneType():
            # None does not carry any information
            return result.new_with_metadata(
                Capabilities(
                    frozenset({sources.Tool(tool, frozenset({sources.SourceEnum.TrustedToolSource}))}), readers.Public()
                )
            )
        case calendar_client.CalendarEvent():
            return result.new_with_metadata(_get_calendar_event_metadata(result.raw, tool))
        case cloud_drive_client.CloudDriveFile():
            return result.new_with_metadata(_get_cloud_drive_file_metadata(result.raw, tool))
        case email_client.Email():
            return result.new_with_metadata(_get_email_metadata(result.raw, tool))
        case email_client.EmailContact():
            return result.new_with_metadata(_get_email_contact_metadata(result.raw, tool))
        case banking_client.Transaction():
            return result.new_with_metadata(_get_transaction_metadata(result.raw, tool))
        case slack.Message():
            return result.new_with_metadata(_get_message_metadata(result.raw, tool))
        case _:
            return result


_RT = TypeVar("_RT", bound=functions_runtime.FunctionReturnType)


class AgentDojoFunction(value.CaMeLCallable, Generic[_RT]):
    is_builtin: bool = True

    def __init__(
        self,
        name: str,
        metadata: Capabilities,
        runtime: functions_runtime.FunctionsRuntime,
        env: functions_runtime.TaskEnvironment | None,
    ):
        # TODO: this is actually wrong
        self._python_value = callable
        self._metadata = metadata
        self._name = name
        self._recv: value.CaMeLValue | None = None
        self._runtime = runtime
        self._function = runtime.functions[name]
        self._env = env
        self._dependencies = ()

    def call(
        self,
        args: value.CaMeLTuple,
        kwargs: value.CaMeLDict[value.CaMeLStr, value.CaMeLValue],
        namespace: ns.Namespace,
    ) -> tuple[value.CaMeLValue[_RT], dict[str, Any]]:
        args_by_keyword = self._make_args_by_keyword(args, kwargs)
        res, _ = self._runtime.run_function(self._env, self._name, args_by_keyword, raise_on_error=True)
        wrapped_output = self.wrap_output(res, args, kwargs, namespace)  # type: ignore -- unsure why this does not typecheck
        return wrapped_output, args_by_keyword

    def wrap_output(
        self,
        value: _RT,
        args: value.CaMeLTuple,
        kwargs: value.CaMeLDict[value.CaMeLStr, value.CaMeLValue],
        namespace: ns.Namespace,
    ) -> value.CaMeLValue[_RT]:
        wrapped_output = super().wrap_output(value, args, kwargs, namespace)
        return _get_metadata_for_ad(wrapped_output, self.name().raw)

    def make_args_by_keyword_preserve_values(
        self, args: value.CaMeLTuple, kwargs: value.CaMeLDict[value.CaMeLStr, value.CaMeLValue]
    ) -> dict[str, Any]:
        args_by_keyword = {}
        fn_args = self._function.parameters.model_json_schema()["properties"].keys()
        for arg_name, arg in zip(fn_args, args.iterate_python()):
            args_by_keyword[arg_name] = arg
        for arg_name, arg in kwargs._python_value.items():
            args_by_keyword[arg_name.raw] = arg
        return args_by_keyword


def make_agentdojo_namespace(
    namespace: ns.Namespace,
    runtime: functions_runtime.FunctionsRuntime,
    env: functions_runtime.TaskEnvironment | None,
) -> dict[str, value.CaMeLValue]:
    ad_functions: dict[str, value.CaMeLValue] = {
        fn_name: AgentDojoFunction(fn_name, Capabilities.camel(), runtime, env) for fn_name in runtime.functions
    }

    return_types: dict[str, value.CaMeLValue] = {}

    def _get_types_recursive(t: Any | None):
        if isinstance(t, type) and not isinstance(t, GenericAlias):
            if issubclass(t, pydantic.BaseModel):
                return_types[t.__name__] = value.CaMeLClass(
                    t.__name__,
                    t,
                    Capabilities.camel(),
                    (),
                    {},
                    (namespace.variables["BaseModel"],),  # type: ignore -- this is hardcoded to be a class
                )
                for field in t.model_fields.values():
                    _get_types_recursive(field.annotation)
            if issubclass(t, enum.Enum):
                return_types[t.__name__] = value.CaMeLClass(
                    t.__name__,
                    t,
                    Capabilities.camel(),
                    (),
                    {},
                    (namespace.variables["Enum"],),  # type: ignore -- this is hardcoded to be a class
                )
        for arg in typing.get_args(t):
            _get_types_recursive(arg)

    for fn in runtime.functions.values():
        _get_types_recursive(fn.return_type)

    return ad_functions | return_types
