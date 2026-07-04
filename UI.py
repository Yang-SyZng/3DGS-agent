from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import chainlit as cl

from agents import ZoteroAgent

logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _get_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return tool_calls

    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    raw_tool_calls = additional_kwargs.get("tool_calls") or []
    parsed_tool_calls = []
    for call in raw_tool_calls:
        function = call.get("function", {})
        parsed_tool_calls.append(
            {
                "name": function.get("name") or call.get("name") or "tool",
                "args": function.get("arguments") or call.get("args") or {},
                "id": call.get("id"),
            }
        )
    return parsed_tool_calls


async def _show_execution_steps(messages: list[Any]) -> None:
    for message in messages:
        for tool_call in _get_tool_calls(message):
            name = tool_call.get("name") or "tool"
            async with cl.Step(name=f"调用工具: {name}", type="tool") as step:
                step.input = tool_call.get("args") or {}
                step.output = "工具调用已发起"

        if getattr(message, "type", None) == "tool":
            name = getattr(message, "name", None) or "tool"
            async with cl.Step(name=f"工具结果: {name}", type="tool") as step:
                step.input = {"tool_call_id": getattr(message, "tool_call_id", None)}
                step.output = _content_to_text(getattr(message, "content", ""))


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content="正在初始化 Zotero Agent...").send()
    agent = await cl.make_async(ZoteroAgent)()
    cl.user_session.set("zotero_agent", agent)
    await cl.Message(content="Zotero Agent 已就绪，可以直接输入要查找或下载的论文。").send()


@cl.on_message
async def main(message: cl.Message) -> None:
    agent = cl.user_session.get("zotero_agent")
    if agent is None:
        await cl.Message(content="Zotero Agent 尚未初始化，正在重新初始化...").send()
        agent = await cl.make_async(ZoteroAgent)()
        cl.user_session.set("zotero_agent", agent)

    progress = cl.Message(content="正在处理请求...")
    await progress.send()

    try:
        response = await cl.make_async(agent.invoke)(
            {"messages": [{"role": "user", "content": message.content}]}
        )
    except Exception as exc:
        logger.exception("Zotero Agent 调用失败")
        progress.content = f"调用 Zotero Agent 失败: {exc}"
        await progress.update()
        return

    messages = response.get("messages", [])
    await _show_execution_steps(messages)

    final_content = ""
    if messages:
        final_content = _content_to_text(getattr(messages[-1], "content", ""))

    progress.content = final_content or "处理完成，但没有返回文本结果。"
    await progress.update()


if __name__ == "__main__":
    port = os.getenv("CHAINLIT_PORT", "8766")
    command = [
        sys.executable,
        "-m",
        "chainlit",
        "run",
        str(Path(__file__).resolve()),
        "-w",
        "--port",
        port,
    ]

    if os.getenv("CHAINLIT_HEADLESS", "").lower() in {"1", "true", "yes"}:
        command.append("--headless")

    subprocess.run(command, check=True)
