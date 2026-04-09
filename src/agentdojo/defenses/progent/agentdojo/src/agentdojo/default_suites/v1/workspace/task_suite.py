from functools import partial

from deepdiff.diff import DeepDiff

from agentdojo.default_suites.v1.tools.calendar_client import (
    Calendar,
    add_calendar_event_participants,
    cancel_calendar_event,
    create_calendar_event,
    get_current_day,
    get_day_calendar_events,
    reschedule_calendar_event,
    search_calendar_events,
)
from agentdojo.default_suites.v1.tools.cloud_drive_client import (
    CloudDrive,
    append_to_file,
    create_file,
    delete_file,
    get_file_by_id,
    list_files,
    search_files,
    search_files_by_filename,
    share_file,
)
from agentdojo.default_suites.v1.tools.email_client import (
    Inbox,
    delete_email,
    get_draft_emails,
    get_received_emails,
    get_sent_emails,
    get_unread_emails,
    search_contacts_by_email,
    search_contacts_by_name,
    search_emails,
    send_email,
)
from agentdojo.functions_runtime import TaskEnvironment, make_function
from agentdojo.task_suite.task_suite import TaskSuite
from secagent import apply_secure_tool_wrapper, update_always_allowed_tools, update_security_policy
import os


class WorkspaceEnvironment(TaskEnvironment):
    inbox: Inbox
    calendar: Calendar
    cloud_drive: CloudDrive


TOOLS = [
    # Email tools
    send_email,
    delete_email,
    get_unread_emails,
    get_sent_emails,
    get_received_emails,
    get_draft_emails,
    search_emails,
    search_contacts_by_name,
    search_contacts_by_email,
    # Calendar tools
    get_current_day,
    search_calendar_events,
    get_day_calendar_events,
    create_calendar_event,
    cancel_calendar_event,
    reschedule_calendar_event,
    add_calendar_event_participants,
    # Drive tools
    append_to_file,
    search_files_by_filename,
    create_file,
    delete_file,
    get_file_by_id,
    list_files,
    share_file,
    search_files,
]

deepdiff_paths_to_exclude = [
    "root.inbox.sent",
    "root.inbox.received",
    "root.inbox.drafts",
    "root.responses",
    "root.model_fields_set",
    "root.calendar.initial_events",
    "root.inbox.initial_emails",
    "root.cloud_drive.initial_files",
]

WorkspaceDeepDiff = partial(DeepDiff, exclude_paths=deepdiff_paths_to_exclude)

suite = os.getenv('SECAGENT_SUITE', None)
generate_policy = os.getenv('SECAGENT_GENERATE', "True").lower() == "true"
if suite != "workspace":
    task_suite = TaskSuite[WorkspaceEnvironment]("workspace", WorkspaceEnvironment, [make_function(tool) for tool in TOOLS])
else:
    task_suite = TaskSuite[WorkspaceEnvironment]("workspace", WorkspaceEnvironment, [make_function(apply_secure_tool_wrapper(tool)) for tool in TOOLS])
    update_always_allowed_tools([
        "get_unread_emails",
        "get_sent_emails",
        "get_received_emails",
        "get_draft_emails",
        "search_emails",
        "search_contacts_by_name",
        "search_contacts_by_email",
        "get_current_day",
        "search_calendar_events",
        "get_day_calendar_events",
        "search_files_by_filename",
        "get_file_by_id",
        "list_files",
        "search_files"
    ])
    # manual policies
    if not generate_policy:
        trusted_email_list = ['john.mitchell@gmail.com', 'david.smith@bluesparrowtech.com', 'david.lee@bluesparrowtech.com', 'linda.green@luesparrowtech.com', 'mark.roberts@luesparrowtech.com',
                              'sarah.connor@gmail.com', 'emma.johnson@bluesparrowtech.com', 'martha.raynolds@gmail.com', 'mark.davies@hotmail.com', 'sarah.baker@gmail.com',
                              'john.doe@clientcorp.com', 'maria.garcia@bluesparrowtech.com', 'john.doe@gmail.com',
                              'james.miller@yahoo.com']
        policies = {
            "get_unread_emails":                  [(1, 0, {}, 0)],
            "get_sent_emails":                    [(1, 0, {}, 0)],
            "get_received_emails":                [(1, 0, {}, 0)],
            "get_draft_emails":                   [(1, 0, {}, 0)],
            "search_emails":                      [(1, 0, {}, 0)],
            "search_contacts_by_name":            [(1, 0, {}, 0)],
            "search_contacts_by_email":           [(1, 0, {}, 0)],
            "get_current_day":                    [(1, 0, {}, 0)],
            "search_calendar_events":             [(1, 0, {}, 0)],
            "get_day_calendar_events":            [(1, 0, {}, 0)],
            "search_files_by_filename":           [(1, 0, {}, 0)],
            "get_file_by_id":                     [(1, 0, {}, 0)],
            "list_files":                         [(1, 0, {}, 0)],
            "search_files":                       [(1, 0, {}, 0)],
            "send_email":                         [(1, 0, {
                "recipients": {'items': {'enum': trusted_email_list}}
            }, 0)],
            "create_calendar_event":              [(1, 0, {
                "participants": {
                    "anyOf": [
                        {'items': {'enum': trusted_email_list}},
                        {'type': 'null'}
                    ]
                }
            }, 0)],
            "reschedule_calendar_event":          [(1, 0, {}, 0)],
            "add_calendar_event_participants":    [(1, 0, {
                "participants": {'items': {'enum': trusted_email_list}}
            }, 0)],
            "append_to_file":                     [(1, 0, {}, 0)],
            "create_file":                        [(1, 0, {}, 0)],
            "share_file":                         [(1, 0, {
                "emails": {'enum': trusted_email_list}
            }, 0)],
        }
        update_security_policy(policies)
