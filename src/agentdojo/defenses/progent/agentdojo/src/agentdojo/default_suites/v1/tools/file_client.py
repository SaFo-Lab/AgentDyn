from typing import Annotated, Any, Dict, Literal, Union

from pydantic import BaseModel

from agentdojo.functions_runtime import Depends

from pydantic import BaseModel, Field

from agentdojo.default_suites.v1.tools.types import File, Directory

class Filesystem(BaseModel):
    # root: Directory = Field(default_factory=Directory)
    root: Directory


def check_node_existence(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str) -> bool:
    """Check whether if the local file or directory exists.

    :param path: The path to the file or directory.
    """
    if not path:
        return False
    
    if path == "/":
        return True
    else:
        parts = path.strip("/").split("/")
        current = filesystem.root
        for part in parts:
            if isinstance(current, Directory) and (part in current.children):
                current = current.children[part]
            else:
                return False
    return True

def get_node_instance(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str):
    """Get the instance of the local file or directory.

    :param path: The path to the file or directory.
    """
    if not path or path == "/":
        current = filesystem.root
        name = "/"
    else:
        parts = path.strip("/").split("/")
        current = filesystem.root
        name = parts[-1]
        
        for part in parts:
            if isinstance(current, Directory) and (part in current.children):
                current = current.children[part]
            else:
                raise ValueError(f"The path of {path} not found.")
        
    return current, name


def ls(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str) -> Directory | File | str:
    """List the metadata of all items inside a specific local directory.

    :param path: The path to the directory (e.g., '/system' or '/user/documents').
    """
    if not path or path == "/":
        current = filesystem.root
    else:
        parts = path.strip("/").split("/")
        current = filesystem.root
        for part in parts:
            if isinstance(current, Directory) and part in current.children:
                current = current.children[part]
            else:
                raise ValueError(f"The path of {path} not found.")

    if not isinstance(current, Directory):
        raise ValueError(f"{path}' is a file. Cannot list children of a file.")

    return current

# VISIBLE TO AGENT
def list_directory(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str) -> list | str:
    """List the metadata of all items inside a specific local directory.

    :param path: The path to the directory (e.g., '/system' or '/user/documents').
    """
    # 1. Navigate to the target node
    if not path or path == "/":
        current = filesystem.root
    else:
        parts = path.strip("/").split("/")
        current = filesystem.root
        for part in parts:
            if isinstance(current, Directory) and part in current.children:
                current = current.children[part]
            else:
                raise ValueError(f"The path of {path} not found.")

    # 2. Check if the target node is a directory
    if not isinstance(current, Directory):
        raise ValueError(f"Error: '{path}' is a file. Cannot list children of a file.")

    # 3. Collect metadata for each child
    children_list = []
    for name, node in current.children.items():
        children_list.append({
            "name": name,
            "type": "directory" if isinstance(node, Directory) else "file",
            "permissions": node.permissions.dict()
        })

    return str(children_list)

def create_dir(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str) -> str:
    """Create a new local directory at the specified path, including parent directories.

    :param path: The path where the directory should be created.
    """
    parts = path.strip("/").split("/")
    current = filesystem.root
    for part in parts:
        if isinstance(current, Directory):
            if part not in current.children:
                current.children[part] = Directory()
            current = current.children[part]
        else:
            raise ValueError(f"Cannot create directory: '{part}' is a file, not a directory.")

    return f"Directory {path} created."

def create_file(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str, content: str | Any = "") -> str:
    """Create a new local file at the specified path.

    :param path: The path to the file to create.
    :param content: The initial content of the file.
    """
    parts = path.strip("/").split("/")
    file_name = parts[-1]
    dir_path = "/".join(parts[:-1])
    
    parent = ls(filesystem, dir_path) if dir_path else filesystem.root
    if isinstance(parent, Directory):
        parent.children[file_name] = File(content=str(content))
        return f"File {path} created"
    raise ValueError(f"Parent directory for {path} not found")


def add_node(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str, node: File | Directory) -> str:
    """Create a new local file at the specified path.

    :param path: The path to the node to create.
    :param node: The node to be added under the given path.
    """
    parts = path.strip("/").split("/")
    file_name = parts[-1]
    dir_path = "/".join(parts[:-1])
    
    parent = ls(filesystem, dir_path) if dir_path else filesystem.root
    if isinstance(parent, Directory):
        if file_name in parent.children:
            n = 1
            while (file_name+f"({n})" in parent.children) and (n < 100):
                n += 1
            file_name = file_name+f"({n})"
        parent.children[file_name] = node
        return f"{file_name} has been added to {dir_path}."
    raise ValueError(f"Parent directory of {dir_path} not found")


def delete_node(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str) -> str:
    """Delete a local file or directory and all its contents at the given path.

    :param path: The path to the node to delete.
    """
    parts = path.strip("/").split("/")
    if not parts or path == "/":
        raise ValueError("Cannot delete root directory")
    
    parent_path = "/".join(parts[:-1])
    target_name = parts[-1]
    
    parent = ls(filesystem, parent_path) if parent_path else filesystem.root
    if isinstance(parent, Directory) and target_name in parent.children:
        del parent.children[target_name]
        return f"{path} successfully deleted."
    raise FileNotFoundError(f"Path {path} not found")


import copy

def copy_node(filesystem: Annotated[Filesystem, Depends("filesystem")], source_path: str, destination_path: str) -> str:
    """Copies a local file or directory from the source path to the destination path.
    
    :param source_path: The path of the file or directory to copy.
    :param destination_path: The destination directory or the new path for the copy.
    """
    # 1. Get the source instance
    source_node, source_name = get_node_instance(filesystem, source_path)
    
    # 2. Determine the destination parent and the new name
    if check_node_existence(filesystem, destination_path):
        dest_node, _ = get_node_instance(filesystem, destination_path)
        if isinstance(dest_node, Directory):
            # If destination is an existing directory, place the copy inside it
            parent_dir = dest_node
            new_name = source_name
        else:
            raise FileExistsError(f"Target '{destination_path}' already exists and is a file.")
    else:
        # If destination doesn't exist, treat the last part of destination_path as the new name
        dest_parts = destination_path.strip("/").split("/")
        dest_parent_path = "/".join(dest_parts[:-1])
        new_name = dest_parts[-1]
        parent_dir = ls(filesystem, dest_parent_path) if dest_parent_path else filesystem.root

    if not isinstance(parent_dir, Directory):
        raise ValueError(f"Destination parent path is not a directory.")

    # 3. Handle naming conflicts in the destination
    final_name = new_name
    if final_name in parent_dir.children:
        n = 1
        while (f"{new_name}_{n}" in parent_dir.children):
            n += 1
        final_name = f"{new_name}_{n}"

    # 4. Perform the deep copy
    parent_dir.children[final_name] = copy.deepcopy(source_node)
    
    return f"Successfully copied '{source_path}' to '{destination_path}' (as '{final_name}')."


def move_node(filesystem: Annotated[Filesystem, Depends("filesystem")], source_path: str, destination_path: str) -> str:
    """Moves or renames a local file or directory from the source path to the destination path.
    
    :param source_path: The path of the node to move.
    :param destination_path: The destination path.
    """
    if source_path == "/":
        raise ValueError("Cannot move the root directory.")

    # 1. Get the source node and its parent
    source_node, source_name = get_node_instance(filesystem, source_path)
    
    source_parts = source_path.strip("/").split("/")
    source_parent_path = "/".join(source_parts[:-1])
    source_parent = ls(filesystem, source_parent_path) if source_parent_path else filesystem.root

    # 2. Determine destination target
    if check_node_existence(filesystem, destination_path):
        dest_node, _ = get_node_instance(filesystem, destination_path)
        if isinstance(dest_node, Directory):
            # Move into the existing directory
            target_parent_dir = dest_node
            target_name = source_name
        else:
            raise FileExistsError(f"Destination '{destination_path}' already exists.")
    else:
        # Rename/Move to new path
        dest_parts = destination_path.strip("/").split("/")
        dest_parent_path = "/".join(dest_parts[:-1])
        target_name = dest_parts[-1]
        target_parent_dir = ls(filesystem, dest_parent_path) if dest_parent_path else filesystem.root

    # 3. Prevent moving a directory into itself (Infinite Loop Protection)
    if source_path == destination_path or destination_path.startswith(source_path.rstrip("/") + "/"):
        raise ValueError("Cannot move a directory into itself.")

    # 4. Update References
    if not isinstance(target_parent_dir, Directory):
        raise ValueError("Target destination parent is not a directory.")

    target_parent_dir.children[target_name] = source_node
    if isinstance(source_parent, Directory):
        del source_parent.children[source_name]

    return f"Successfully moved '{source_path}' to '{destination_path}'."

def read_file(filesystem: Annotated[Filesystem, Depends("filesystem")], path: str) -> str:
    """Read the contents of a local file at the given path.

    :param path: The path of the local file to read.
    """
    node, name = get_node_instance(filesystem, path)
    if node is None:
        raise FileNotFoundError(f"Path {path} not found")
    if isinstance(node, Directory):
        raise IsADirectoryError(f"Path {path} is a directory, not a file")
    if not node.permissions.read:
        raise PermissionError(f"Read permission denied for {path}")
    return node.content

