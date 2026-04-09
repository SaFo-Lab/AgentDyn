# tool_registry.py
TOOL_REGISTRY = {}

def register_tool(name: str):
    """register tools"""
    def wrapper(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return wrapper

def get_tool(name: str):
    return TOOL_REGISTRY.get(name)