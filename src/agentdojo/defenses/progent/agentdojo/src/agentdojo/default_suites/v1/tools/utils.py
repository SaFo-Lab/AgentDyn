import datetime
from agentdojo.base_tasks import BaseUserTask
from agentdojo.default_suites.v1.shopping.task_suite import ShoppingEnvironment, task_suite
from agentdojo.functions_runtime import FunctionCall
from agentdojo.default_suites.v1.tools.types import File, Directory

ShoppingUserTask = BaseUserTask[ShoppingEnvironment]

def find_element(elements, condition):
    for element in elements:
        if condition(element):
            return element
    return None

def parse_date(date: str) -> datetime.date:
    return datetime.date.fromisoformat(date)


def parse_datetime(datetime_str: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(datetime_str)

from typing import Optional, Union

def get_node_by_path(
    root: Directory, 
    path: str
) -> File | Directory | None:
    """
    Starting from `root`, traverse the path `path` to find a node.
    Returns the node if it exists, otherwise returns None.
    `path` should be like "/user/github/linear_algebra_operation"
    """
    # Clean the path and split by "/"
    nodes = [p for p in path.strip("/").split("/") if p]
    current_node = root

    for i, node in enumerate(nodes):
        if not isinstance(current_node, Directory):
            # Current node is not a Directory
            return None
        if node in current_node.children:
            current_node = current_node.children[node]
        else:
            return None

    return current_node