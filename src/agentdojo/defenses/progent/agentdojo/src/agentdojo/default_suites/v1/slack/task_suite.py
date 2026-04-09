from agentdojo.default_suites.v1.tools.slack import (
    Slack,
    add_user_to_channel,
    get_channels,
    get_users_in_channel,
    invite_user_to_slack,
    read_channel_messages,
    read_inbox,
    remove_user_from_slack,
    send_channel_message,
    send_direct_message,
)
from agentdojo.default_suites.v1.tools.web import Web, get_webpage, post_webpage
from agentdojo.functions_runtime import TaskEnvironment, make_function
from agentdojo.task_suite.task_suite import TaskSuite
from secagent import apply_secure_tool_wrapper, update_always_allowed_tools, update_security_policy
import os


class SlackEnvironment(TaskEnvironment):
    slack: Slack
    web: Web


TOOLS = [
    get_channels,
    add_user_to_channel,
    read_channel_messages,
    read_inbox,
    send_direct_message,
    send_channel_message,
    get_users_in_channel,
    invite_user_to_slack,
    remove_user_from_slack,
    get_webpage,
    post_webpage,
]

suite = os.getenv('SECAGENT_SUITE', None)
generate_policy = os.getenv('SECAGENT_GENERATE', "True").lower() == "true"
if suite != "slack":
    task_suite = TaskSuite[SlackEnvironment]("slack", SlackEnvironment, [make_function(tool) for tool in TOOLS])
else:
    task_suite = TaskSuite[SlackEnvironment]("slack", SlackEnvironment, [make_function(apply_secure_tool_wrapper(tool)) for tool in TOOLS])
    update_always_allowed_tools(["get_channels", "read_channel_messages", "read_inbox", "get_users_in_channel"])
    # manual policies
    if not generate_policy:
        blacklist = ".*\\.secure-systems-252\\.com.*"
        whitelist = ".*\\.(informations|dora-website|our-company|eve-blog|restaurant-zurich|company-todo-list)\\.com.*"

        policies = {
            'get_channels':                 [(1, 0, {}, 0)],
            'add_user_to_channel':          [(1, 0, {}, 0)],
            'read_channel_messages':        [(1, 0, {}, 0)],
            'read_inbox':                   [(1, 0, {}, 0)],
            'send_direct_message':          [(1, 1, {
                "body": {'type': 'string', 'pattern': blacklist}
            }, 0), (1, 0, {}, 0)],
            'send_channel_message':         [(1, 1, {
                "body": {'type': 'string', 'pattern': blacklist}
            }, 0), (1, 0, {}, 0)],
            'get_users_in_channel':         [(1, 0, {}, 0)],
            'invite_user_to_slack':         [(1, 0, {
                "user_email": {'type': 'string', 'enum': ["dora@gmail.com"]}
            }, 0)],
            'remove_user_from_slack':       [(1, 0, {}, 0)],
            'get_webpage':                  [(1, 0, {
                "url": {'type': 'string', 'pattern': whitelist}
            }, 0)],
            'post_webpage':                 [(1, 0, {
                "url": {'type': 'string', 'pattern': whitelist}
            }, 0)],
        }
        update_security_policy(policies)
