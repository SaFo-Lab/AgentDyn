from import_lib import *

class DRIFTToolsExecutionLoop(ToolsExecutionLoop):
    """Executes in loop a sequence of pipeline elements related to tool execution until the
    LLM does not return any tool calls.

    Args:
        elements: a sequence of pipeline elements to be executed in loop. One of them should be
            an LLM, and one of them should be a [ToolsExecutor][agentdojo.agent_pipeline.ToolsExecutor] (or
            something that behaves similarly by executing function calls). You can find an example usage
            of this class [here](../../concepts/agent_pipeline.md#combining-pipeline-components).
        max_iters: maximum number of iterations to execute the pipeline elements in loop.
    """

    def __init__(self, elements: Sequence[BasePipelineElement], max_iters: int = 15) -> None:
        self.elements = elements
        self.max_iters = max_iters

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for block in content:
                if isinstance(block, dict):
                    chunks.append(str(block.get("content", "")))
                else:
                    chunks.append(str(block))
            return "\n".join(chunks)
        if content is None:
            return ""
        return str(content)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            raise ValueError("Messages should not be empty when calling ToolsExecutionLoop")

        logger = Logger().get()
        for _ in range(self.max_iters):
            last_message = messages[-1]
            content_text = self._message_text(last_message)
            error_text = str(last_message.get("error", "") or "")
            has_call_error = (
                "[CALL ERROR]" in content_text
                or "[CALL_ERROR]" in content_text
                or "[CALL ERROR]" in error_text
                or "[CALL_ERROR]" in error_text
            )
            if not has_call_error:
                if not last_message["role"] == "assistant":
                    break
                if last_message["tool_calls"] is None:
                    break
                if len(last_message["tool_calls"]) == 0:
                    break
            for element in self.elements:
                query, runtime, env, messages, extra_args = element.query(query, runtime, env, messages, extra_args)
                logger.log(messages)
        return query, runtime, env, messages, extra_args