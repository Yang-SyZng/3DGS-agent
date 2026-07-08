from __future__ import annotations
from typing import Any

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow import AgentStream
import logging

from config import setting

logger = logging.getLogger(__name__)


class BaseFunctionAgent(FunctionAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        if kwargs.get("llm") is None and len(args) < 7:
            kwargs["llm"] = self.build_model()
        kwargs.setdefault("name", "Baser")
        super().__init__(*args, **kwargs)

        logger.info("正在构建 %s Agent...", self.name)

    def build_model(self) -> OpenAILike:
        """Create the CHAT&Function Calling model"""
        logger.info("正在构建LLM...")
        return OpenAILike(
            model=setting.LLM_MODEL_ID,
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=128000,
        )

    def BuildModel(self) -> OpenAILike:
        return self.build_model()

    async def stream_run(self, msg: str) -> None:
        response = self.run(user_msg=msg)
        started = False

        async for event in response.stream_events():
            if isinstance(event, ToolCall):
                logger.info("Tool Call: %s, %s", event.tool_name, event.tool_kwargs)
            elif isinstance(event, ToolCallResult):
                logger.info("Tool Result:\n%s", event.tool_output)
            elif isinstance(event, AgentStream):
                delta = event.delta
                if not started:
                    delta = delta.lstrip()
                    started = True
                print(delta, end="", flush=True)
