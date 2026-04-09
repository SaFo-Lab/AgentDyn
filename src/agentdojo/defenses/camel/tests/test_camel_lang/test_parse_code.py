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

from camel.interpreter.interpreter import extract_code_block


def test_extract_fenced_with_language():
    markdown_text = """Some text.\n```python\ndef my_function():\n    print("Hello")\n```\nMore text."""
    expected_code = 'def my_function():\n    print("Hello")'
    assert extract_code_block(markdown_text) == expected_code


def test_extract_fenced_with_other_string():
    markdown_text = """Some text.\n```tool_code\ndef my_function():\n    print("Hello")\n```\nMore text."""
    expected_code = 'def my_function():\n    print("Hello")'
    assert extract_code_block(markdown_text) == expected_code


def test_extract_fenced_no_language():
    markdown_text = """Some text.\n```\ndef my_function():\n    print("Hello")\n```\nMore text."""
    expected_code = 'def my_function():\n    print("Hello")'
    assert extract_code_block(markdown_text) == expected_code
