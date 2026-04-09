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

import json
import sys

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


def print_conversation_from_json(json_file_path):
    """
    Prints the conversation from a JSON file in a nice format using rich.
    Highlights utility and formats Python code blocks.

    Args:
        json_file_path (str): The path to the JSON file.
    """
    console = Console()

    with console.pager():
        try:
            with open(json_file_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            console.print(f"Error: File not found at path: {json_file_path}", style="bold red")
            return
        except json.JSONDecodeError:
            console.print(
                f"Error: Invalid JSON format in file: {json_file_path}",
                style="bold red",
            )
            return

        suite_name = data.get("suite_name", "N/A")
        pipeline_name = data.get("pipeline_name", "N/A")
        user_task_id = data.get("user_task_id", "N/A")
        utility = data.get("utility", False)

        if utility:
            utility_panel = Panel(
                Text.from_markup(
                    "[bold white]UTILITY[/bold white]: [bold green]True[/bold green] - This conversation is considered [bold green]UTILITARIAN[/bold green].",
                    justify="center",
                ),
                style="bold green",
                border_style="green",
                padding=(1, 2),
            )
            console.print(utility_panel)
            console.print()

        console.rule(f"[bold blue]Conversation: {suite_name} | {pipeline_name} | {user_task_id}[/bold blue]")

        for message in data.get("messages", []):
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                role_text = Text("User", style="bold blue")
                panel_content = Text.from_markup(content)

            elif role == "assistant":
                role_text = Text("Assistant", style="bold green")
                panel_elements = []
                parts = content.split("```python")  # Split by python code blocks

                panel_elements.append(Text.from_markup(parts[0]))  # Add text before first code block

                for i in range(1, len(parts)):
                    code_block_parts = parts[i].split("```")  # Split code block part by closing backticks
                    code = code_block_parts[0].strip()
                    syntax = Syntax(code, "python", theme="monokai", line_numbers=False, word_wrap=True)
                    panel_elements.append(syntax)  # Append the Syntax object directly
                    if len(code_block_parts) > 1:  # Add text after code block if any
                        panel_elements.append(Text.from_markup(code_block_parts[1]))
                    else:
                        panel_elements.append(Text(""))  # Add empty text if no text after code block

                panel_content = Group(*panel_elements)  # Wrap panel_elements in a Group

            elif role == "tool":
                role_text = Text(f"Tool - {message['tool_call']['function']}", style="bold magenta")
                panel_content = Text.from_markup(content)
            else:
                role_text = Text(role.capitalize(), style="bold yellow")
                panel_content = Text.from_markup(content)

            panel = Panel(
                panel_content,  # Pass panel_content (which is now a Group)
                title=role_text,
                border_style="cyan"
                if role == "user"
                else "green"
                if role == "assistant"
                else "magenta"
                if role == "tool"
                else "yellow",
                padding=(1, 2),
            )
            console.print(panel)
            console.print()

        console.rule("[bold blue]End of Conversation[/bold blue]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]  # Get file path from command line argument
        print_conversation_from_json(json_file)
    else:
        print("Usage: python print_conversation.py <path_to_json_file>")
        sys.exit(1)  # Exit with an error code to indicate incorrect usage
