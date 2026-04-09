import os
import inspect

from typing import Annotated, Any

from pydantic import BaseModel

from agentdojo.functions_runtime import Depends
from agentdojo.default_suites.v1.tools.tool_registry import register_tool, get_tool

from agentdojo.default_suites.v1.tools.dynamic_email_client import EmailDatabase
from agentdojo.default_suites.v1.tools.shopping_client import (
    BankDatabase,
    ShoppingPlatform,
    # send_money,
    # get_balance,
    # login_shopping_account,
    # verify_shopping_account,
    # get_shopping_account_user_information,
    # update_shopping_account_password,
    # update_shopping_account_user_info,
    # search_product,
    # view_order_history,
    # view_cart,
    # cart_add_product,
    # cart_remove_product,
    # clear_cart,
    # cart_total,
    # checkout_selected_cart,
    # view_available_discounts,
    # get_shopping_linked_unread_emails,
)
from agentdojo.default_suites.v1.tools.dynamic_email_client import send_email, get_unread_emails, get_received_emails, get_sent_emails, delete_email, search_emails
from agentdojo.default_suites.v1.tools.file_client import Filesystem, list_directory, create_file, create_dir, read_file, delete_node, copy_node, move_node
from agentdojo.default_suites.v1.tools.dynamic_web_client import WebDatabase, get_webpage, standardize_url, fetch_web_instance, download_file_through_url, download_file_through_id

from agentdojo.default_suites.v1.tools.dynamic_calendar_client import (
    Calendar,
    add_calendar_event_participants,
    cancel_calendar_event,
    create_calendar_event,
    get_current_day,
    get_day_calendar_events,
    reschedule_calendar_event,
    search_calendar_events,
)

from agentdojo.default_suites.v1.tools.github_client import (
    GitHubPlatform,
    git_create_repo,
    git_delete_repo,
    git_transfer_repo_ownership,
    git_clone,
    git_invite_collaborators,
    git_push,
    git_pull,
    git_issue,
    git_star,
    git_unstar,
    get_github_linked_unread_emails,
    get_logged_in_github_user_information,
    get_github_account_user_information,
    get_github_repository_information,
    git_get_linked_ssh_keys,
    git_add_ssh_key,
    git_delete_ssh_key,
    login_github_account,
    update_github_account_password,
    verify_github_account
)

from agentdojo.functions_runtime import TaskEnvironment, make_function
from agentdojo.task_suite.task_suite import TaskSuite

from secagent import apply_secure_tool_wrapper, update_always_allowed_tools, update_security_policy

class GithubEnvironment(TaskEnvironment):
    email_database: EmailDatabase
    bank_database: BankDatabase
    web_database: WebDatabase
    filesystem: Filesystem
    calendar: Calendar
    github_platform: GitHubPlatform
    shopping_platform: ShoppingPlatform


# depends_params = {
#     "inbox": Annotated[DynamicInbox, Depends("inbox")],
#     "email_database": Annotated[EmailDatabase, Depends("email_database")],
#     "web_database": Annotated[WebDatabase, Depends("web_database")],
# }

def call_function(func_name: str, inputs: dict):
    """
    Dynamically calls a tool by filtering out redundant inputs based on 
    the function's actual parameter list.

    :param func_name: The identifier of the registered tool/function.
    :param inputs: A dictionary containing potential parameters (may be redundant).
    :return: The result of the function execution.
    """
    fn = get_tool(func_name)
    if fn is None:
        raise ValueError(f"Tool '{func_name}' not found in registry.")

    # Get the signature of the target function
    sig = inspect.signature(fn)
    
    # Check if the function has a variable keyword argument parameter (**kwargs)
    # If it does, we don't need to filter because it can handle redundancy natively.
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    if has_kwargs:
        # Pass everything if the function accepts **kwargs
        return fn(**inputs)
    else:
        # Filter inputs to only include keys that exist in the function's signature
        # We only consider parameters that can be passed as keyword arguments
        valid_params = {
            param_name for param_name, param in sig.parameters.items()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        
        filtered_inputs = {
            k: v for k, v in inputs.items() if k in valid_params
        }
        
        return fn(**filtered_inputs)

@register_tool("input_to_webpage")
def input_to_webpage(shopping_platform: Annotated[ShoppingPlatform, Depends("shopping_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], bank_database: Annotated[BankDatabase, Depends("bank_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], url: str, inputs: dict[str, Any]):
    """Interact with the target webpage. Strictly follow the url and inputs format below to fill the web form.

    :param url: The URL of the webpage.
    :param inputs: A mapping from HTML input element IDs to user-provided values.
                This represents form inputs on the webpage. These inputs are mapped internally to the corresponding handler parameters according to the webpage schema.
                Example:
                url="www.example.com"
                inputs={
                    "email_field": "example@mail.com",
                    "password_field": "mypassword"
                }
    """
    url = standardize_url(url)
    web = fetch_web_instance(web_database, url)
    mapped_inputs = {}
    for id, value in inputs.items():
        mapped_inputs[web.field_to_parameters[id]] = value

    depends_params = {
        "shopping_platform": shopping_platform,
        "email_database": email_database,
        "bank_database": bank_database,
        "filesystem": filesystem,   
        "web_database": web_database,      
    }

    for func in web.web_function:
        handler_name = func.handler
        params = {**depends_params, **func.static_parameters, **mapped_inputs}
        return call_function(handler_name, params)

@register_tool("browse_webpage")
def browse_webpage(shopping_platform: Annotated[ShoppingPlatform, Depends("shopping_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], bank_database: Annotated[BankDatabase, Depends("bank_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], github_platform: Annotated[GitHubPlatform, Depends("github_platform")], url: str):
    """Visits a URL to retrieve web content or trigger a web-based action. 
    Use this to read a web page or to 'click' a link that performs a specific task or confirmation.

    :param url: The URL to be accessed.
    """
    url = standardize_url(url)
    web = fetch_web_instance(web_database, url)

    depends_params = {
        "shopping_platform": shopping_platform,
        "email_database": email_database,
        "bank_database": bank_database,
        "filesystem": filesystem,   
        "web_database": web_database,      
        "github_platform": github_platform, 
    }

    if (web.field_to_parameters == {}) and (web.web_function != []):
        for func in web.web_function:
            handler_name = func.handler
            params = {**depends_params, **func.static_parameters}
            return call_function(handler_name, params)
    else:
        return web.web_content


TOOLS = [
    send_email,
    get_received_emails,
    get_sent_emails,
    download_file_through_url,
    download_file_through_id,
    list_directory,
    create_file,
    create_dir,
    read_file,
    delete_node,
    copy_node,
    move_node,
    input_to_webpage,
    browse_webpage,
    add_calendar_event_participants,
    cancel_calendar_event,
    create_calendar_event,
    get_current_day,
    get_day_calendar_events,
    reschedule_calendar_event,
    search_calendar_events,
    git_create_repo,
    git_delete_repo,
    git_transfer_repo_ownership,
    git_clone,
    git_invite_collaborators,
    git_push,
    git_pull,
    git_issue,
    git_star,
    git_unstar,
    get_github_linked_unread_emails,
    get_logged_in_github_user_information,
    get_github_account_user_information,
    get_github_repository_information,
    git_get_linked_ssh_keys,
    git_add_ssh_key,
    git_delete_ssh_key,
    login_github_account,
    update_github_account_password,
    verify_github_account
]

task_suite = TaskSuite[GithubEnvironment]("github", GithubEnvironment, [make_function(tool) for tool in TOOLS])

suite = os.getenv('SECAGENT_SUITE', None)
generate_policy = os.getenv('SECAGENT_GENERATE', "True").lower() == "true"
if suite != "github":
    task_suite = TaskSuite[GithubEnvironment]("github", GithubEnvironment, [make_function(tool) for tool in TOOLS])
else:
    task_suite = TaskSuite[GithubEnvironment]("github", GithubEnvironment, [make_function(apply_secure_tool_wrapper(tool)) for tool in TOOLS])
    update_always_allowed_tools(["get_received_emails", "get_sent_emails", "get_github_linked_unread_emails", "get_logged_in_github_user_information", "get_github_account_user_information", "git_get_linked_ssh_keys", "list_directory", "read_file", "get_github_repository_information", "get_current_day", "get_day_calendar_events"])