from pathlib import Path
import os
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from config.settings import setting

class ArxivMCPClient(BasicMCPClient):
    def __init__(self, 
                command_or_url: str = "uv",
                args: list[str] | None = None,
                env: dict[str, str] | None = None,
                storage_path: str | Path = setting.pdf_save_dir,
                timeout: int = 30,
                sse_read_timeout: int = 300,
                auth=None,
                sampling_callback=None,
                headers=None,
                tool_call_logs_callback=None,
                http_client=None,
            ):
        if args is None:
            args = [
                "tool",
                "run",
                "arxiv-mcp-server",
                "--storage-path",
                str(storage_path),
            ]
        default_env = {
            **os.environ,
            "TESSDATA_PREFIX": "/usr/share/tesseract-ocr/5/tessdata",
        }
        if env is not None:
            default_env.update(env)
        super().__init__(
                        command_or_url,
                        args,
                        env,
                        timeout,
                        sse_read_timeout,
                        auth,
                        sampling_callback,
                        headers,
                        tool_call_logs_callback,
                        http_client
                    )