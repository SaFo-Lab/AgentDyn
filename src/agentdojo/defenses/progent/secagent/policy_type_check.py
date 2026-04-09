from jsonschema.validators import validator_for

allowed_keywords_by_type = {
    "string": {"pattern", "maxLength", "minLength"},
    "number": {"multipleOf", "maximum", "exclusiveMaximum", "minimum", "exclusiveMinimum"},
    "integer": {"multipleOf", "maximum", "exclusiveMaximum", "minimum", "exclusiveMinimum"},
    "array": {"items", "maxItems", "minItems"},
}

all_type_specific_keywords = set()
for kws in allowed_keywords_by_type.values():
    all_type_specific_keywords |= kws


def check_type_specific_keys(schema, path=""):
    """
    Recursively check a JSON schema fragment for invalid usage of type-specific keywords.

    If the schema defines a "type", then any key in the schema that is a known type-specific keyword
    must be allowed for that type (or one of the types if "type" is a list).

    Args:
        schema (dict): A JSON schema fragment.
        path (str): A string indicating the location of the schema in the overall document.

    Returns:
        List[str]: A list of error messages found in this schema fragment.
    """
    errors = []

    # Ensure we're working with a dict.
    if not isinstance(schema, dict):
        return errors

    if "type" in schema:
        types = schema["type"]
        # Determine allowed keywords based on the type or types.
        if isinstance(types, list):
            allowed = set()
            for t in types:
                allowed |= allowed_keywords_by_type.get(t, set())
        elif isinstance(types, str):
            allowed = allowed_keywords_by_type.get(types, set())
        else:
            allowed = set()

        # Check each key in the schema: if it is a type-specific keyword,
        # then it must be allowed for the given type.
        for key in schema:
            if key in all_type_specific_keywords and key not in allowed:
                errors.append(f"Policy Warning at {path}: '{key}' is not allowed for type {types}.")

    # Recursively check nested schemas.
    for k, v in schema.items():
        if k in ["anyOf", "allOf", "oneOf"]:
            if isinstance(v, list):
                for i, item in enumerate(v):
                    errors.extend(check_type_specific_keys(item, path + f".{k}[{i}]"))
        elif k in ["not", "if", "then", "else"]:
            if isinstance(v, dict):
                errors.extend(check_type_specific_keys(v, path + f".{k}"))
        elif k == "items":
            # "items" can be a schema or an array of schemas.
            if isinstance(v, dict):
                errors.extend(check_type_specific_keys(v, path + ".items"))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    errors.extend(check_type_specific_keys(item, path + f".items[{i}]"))
        elif isinstance(v, dict):
            errors.extend(check_type_specific_keys(v, path + f".{k}"))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    errors.extend(check_type_specific_keys(item, path + f".{k}[{i}]"))

    return errors


def security_policy_type_check(security_policy, available_tools_dict):
    warnings = []
    if security_policy is None:
        return warnings
    for tool_name, policies in security_policy.items():
        if tool_name not in available_tools_dict:
            warnings.append(f"Policy Warning: Tool '{tool_name}' is not available.")
        for priority, effect, policy, fallback in policies:
            for arg_name, restriction in policy.items():
                if arg_name not in available_tools_dict[tool_name]:
                    warnings.append(f"Policy Warning: Argument '{arg_name}' is not available for tool '{tool_name}'.")
                if isinstance(restriction, dict):  # json schema
                    try:
                        validator = validator_for(restriction)
                        validator.check_schema(restriction)
                    except Exception as e:
                        warnings.append(f"Policy Warning: Invalid restriction for argument '{arg_name}' in tool '{tool_name}: {e}'.")
                    errors = check_type_specific_keys(restriction, path=f"{tool_name}.{arg_name}")
                    warnings.extend(errors)
    return warnings


if __name__ == "__main__":
    warnings = security_policy_type_check(
        security_policy={
            "send_money": [
                (1, 0, {"recipient": {"type": "string", "pattern": "UK12345678901234567890"}}, 0),
                (2, 1, {"recipient": {"type": "integer", "pattern": "SE3550000000054910000003"}}, 0),
            ]
        },
        available_tools_dict={
            "send_money": {"recipient": {"description": "IBAN of the recipient", "title": "Recipient", "type": "string"}, "amount": {"description": "Amount of the transaction", "title": "Amount", "type": "number"}, "subject": {"description": "Subject of the transaction", "title": "Subject", "type": "string"}, "date": {"description": "Date of the transaction", "title": "Date", "type": "string"}}
        }
    )
    print(warnings)
