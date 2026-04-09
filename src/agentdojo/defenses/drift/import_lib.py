import os
import re
import ast
import yaml
import copy
import json

import importlib.resources
import warnings
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from prompts import CONSTRAINTS_BUILD_PROMPT, TOOL_CALLING_PROMPT, INJECTION_DETECTION_PROMPT, ADAPTIVE_ATTACK_PROMPT

from agentdojo.logging import Logger
from collections.abc import Sequence

from agentdojo.ast_utils import (
    ASTParsingError,
    create_python_function_from_tool_call,
    parse_tool_calls_from_python_function,
)
from agentdojo.types import ChatAssistantMessage, ChatMessage
from openai.types.chat import (
    ChatCompletionMessageParam,
)

from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import TaskSuite, model_output_from_messages, functions_stack_trace_from_messages
from collections import defaultdict

from pathlib import Path
from typing_extensions import Self
from typing import TypeVar, Any
from collections.abc import Callable, Sequence
from pydantic import BaseModel
from agentdojo.attacks.attack_registry import ATTACKS, load_attack

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import EmptyEnv, Function, FunctionCall, FunctionsRuntime, TaskEnvironment
# from agentdojo.types import ChatMessage
from agentdojo.yaml_loader import ImportLoader

T = TypeVar("T", bound=BaseModel)
TC = TypeVar("TC", bound=Callable)
BenchmarkVersion = tuple[int, int, int]

IT = TypeVar("IT")
Env = TypeVar("Env", bound=TaskEnvironment)

from agentdojo.agent_pipeline import (
    AgentPipeline,
    InitQuery,
    PromptingLLM,
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.agent_pipeline.llms.prompting_llm import InvalidModelOutputError
from agentdojo.functions_runtime import FunctionsRuntime

from agentdojo.types import ChatMessage, MessageContentBlock, get_text_content_as_str