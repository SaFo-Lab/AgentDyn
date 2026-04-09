import inspect

from typing import Annotated

from pydantic import BaseModel

from agentdojo.functions_runtime import Depends
from agentdojo.default_suites.v1.tools.tool_registry import register_tool
from agentdojo.default_suites.v1.tools.file_client import Filesystem, check_node_existence, get_node_instance, add_node
from agentdojo.default_suites.v1.tools.types import Web, WebFile, File, Permissions
from agentdojo.default_suites.v1.tools.tool_registry import register_tool, get_tool

class WebDatabase(BaseModel):
    web_list: list[Web]
    file_list: dict[str, WebFile]

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

def fetch_web_instance(web_database: Annotated[WebDatabase, Depends("web_database")], url: str) -> Web:
    for web in web_database.web_list:
        if web.web_url == url:
            return web
    else:
        return Web(
            web_url=url,
            web_content="404 Not Found",
            field_to_parameters={},
            web_function=[],
            web_requests=[]
        )


def standardize_url(url):
    if url.startswith("https://"):
        url = url[8:]
    if url.startswith("http://"):
        url = url[7:]
    return url


def post_webpage(web_database: Annotated[WebDatabase, Depends("web_database")], url: str, content: str, field_to_parameters: dict = {}, web_function: list = [], download_source: str | None = None) -> None:
    """Posts a webpage at a given URL with the given content.

    :param url: The URL of the webpage.
    :param content: The content of the webpage.
    """
    url = standardize_url(url)
    web = fetch_web_instance(web_database, url)
    web.web_requests.append(url)
    web.web_content = content
    web.field_to_parameters = field_to_parameters
    web.web_function = web_function
    web.download_source = download_source


@register_tool("get_webpage")
def get_webpage(web_database: Annotated[WebDatabase, Depends("web_database")], url: str) -> str | None:
    """Returns the content of the webpage at a given URL.

    :param url: The URL of the webpage.
    """
    url = standardize_url(url)
    web = fetch_web_instance(web_database, url)
    web.web_requests.append(url)
    return web.web_content


@register_tool("post_webfile")
def post_webfile(web_database: Annotated[WebDatabase, Depends("web_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], name: str, content: str, read: bool = True, write: bool = False, execute: bool = False) -> str | None:
    """Downloads a file from a given URL to the local folder.

    :param content: The local file path.
    """
    web_file = WebFile(source_name=name, content=content, permissions=Permissions(read=read, write=write, execute=execute))
    file_id = str(len(web_database.file_list.keys()) + 1)
    web_database.file_list[file_id] = web_file

    return file_id


@register_tool("upload_file_to_web")
def upload_file_to_web(web_database: Annotated[WebDatabase, Depends("web_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], local_file_path: str) -> str | None:
    """Downloads a file from a given URL to the local folder.

    :param local_file_path: The local file path.
    """
    if not check_node_existence(filesystem, local_file_path):
        raise ValueError("The Upload File Path Not Found.")
    
    local_file, file_name = get_node_instance(filesystem, local_file_path)
    if not isinstance(local_file, File):
        raise ValueError("The path is not a File.")
    
    args_dict = local_file.model_dump()
    args_dict['source_name'] = file_name
    
    web_file = WebFile(**args_dict)
    file_id = str(len(web_database.file_list.keys()) + 1)
    web_database.file_list[file_id] = web_file

    return file_id


@register_tool("download_file_through_id")
def download_file_through_id(web_database: Annotated[WebDatabase, Depends("web_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], file_id: str, save_dir: str = "/downloads") -> str | None:
    """Downloads a file from a given email attachment ID to the local folder.

    :param file_id: The ID of the file.
    :param save_dir: The destination directory (parent directory) to save the file.
    """
    if file_id in web_database.file_list:
        fetch_file = web_database.file_list[file_id]
    else:
        return "Source Not Found"

    file_name = fetch_file.source_name

    msg = add_node(filesystem, f"{save_dir}/{file_name}", fetch_file)
    return msg

@register_tool("download_file_through_url")
def download_file_through_url(web_database: Annotated[WebDatabase, Depends("web_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], url: str, save_dir: str = "/downloads") -> str | None:
    """Downloads a file from a given URL to the local folder.

    :param url: The URL of the file.
    :param save_dir: The destination directory (parent directory) to save the file.
    """
    url = standardize_url(url)
    web = fetch_web_instance(web_database, url)
    file_id = web.download_source
    if file_id is None:
        return "Source Not Found. Try to directly access the URL."
    msg = download_file_through_id(web_database, filesystem, file_id, save_dir)

    return msg

