import json
import re


def extract_json(text, enforce_code_block=False):
    """
    Extract JSON content from the given text.

    This function checks:
      - If the text starts with "No", returns None.
      - If a code block with triple backticks exists, it extracts the JSON within.
      - Otherwise, it removes a "Yes" header (if present) and attempts to parse the remaining text.

    Returns:
      The parsed JSON object if extraction and parsing succeed; otherwise, None.
    """
    text = text.strip()

    if text.lower().startswith("no"):
        return None

    code_block_pattern = r"```json(.*?)```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        if enforce_code_block:
            return None
        if text.lower().startswith("yes"):
            json_str = text[3:].strip()
        else:
            json_str = text

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        raise e
