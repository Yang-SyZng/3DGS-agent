from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from config import setting
from .lazy import LazyToolList

logger = logging.getLogger(__name__)

MinerUBackend = Literal["auto", "pipeline", "vlm-engine", "hybrid-engine"]


def _ensure_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是PDF: {path}")
    return path


class MinerUPDFProcess:
    """
    基于 MinerU CLI 的 PDF 解析工具。
    """

    def __init__(self, output_dir: str | Path | None = None, timeout: int = 1800):
        self.output_dir = Path(output_dir or setting.pdf_process_save_dir)
        self.timeout = timeout

    def _check_mineru_cli(self) -> str:
        executable = shutil.which("mineru")
        if executable is None:
            raise RuntimeError(
                "未找到 MinerU CLI。请先安装 MinerU，并确认 `mineru` 命令在 PATH 中可用。"
            )
        return executable

    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        backend: MinerUBackend,
        extra_args: list[str] | None,
    ) -> list[str]:
        executable = self._check_mineru_cli()
        command = [executable, "-p", str(pdf_path), "-o", str(output_dir)]

        if backend != "auto":
            command.extend(["-b", backend])
        if extra_args:
            command.extend(extra_args)

        return command

    @staticmethod
    def _pick_file(files: list[Path], expected_stem: str) -> Path | None:
        if not files:
            return None

        exact_matches = [path for path in files if path.stem == expected_stem]
        candidates = exact_matches or files
        return max(candidates, key=lambda path: path.stat().st_mtime)

    @staticmethod
    def _replace_path(src: Path, dst: Path) -> None:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))

    def _flatten_mineru_output_dir(self, output_root: Path, pdf_path: Path) -> Path:
        """Move MinerU's method subdirectory contents into the PDF-named directory."""
        pdf_output_dir = output_root / pdf_path.stem
        if not pdf_output_dir.is_dir():
            return output_root

        method_dirs = [
            path
            for path in pdf_output_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        parent_files = [path for path in pdf_output_dir.iterdir() if path.is_file()]

        if len(method_dirs) != 1 or parent_files:
            return pdf_output_dir

        method_dir = method_dirs[0]
        for child in method_dir.iterdir():
            self._replace_path(child, pdf_output_dir / child.name)

        method_dir.rmdir()
        return pdf_output_dir

    def _collect_outputs(self, run_output_dir: Path, pdf_path: Path) -> dict[str, Any]:
        markdown_files = list(run_output_dir.rglob("*.md"))
        json_files = list(run_output_dir.rglob("*.json"))

        markdown_path = self._pick_file(markdown_files, pdf_path.stem)
        json_path = self._pick_file(json_files, pdf_path.stem)

        markdown = markdown_path.read_text(encoding="utf-8") if markdown_path else None
        json_content = None
        if json_path:
            try:
                json_content = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("MinerU JSON 输出无法解析: %s", json_path)

        return {
            "markdown_path": str(markdown_path) if markdown_path else None,
            "json_path": str(json_path) if json_path else None,
            "markdown": markdown,
            "json": json_content,
            "output_files": [str(path) for path in sorted(run_output_dir.rglob("*")) if path.is_file()],
        }

    def parse_pdf(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        backend: MinerUBackend = "auto",
        timeout: int | None = None,
        extra_args: list[str] | None = None,
        flatten_output: bool = True,
    ) -> dict[str, Any]:
        """使用 MinerU 解析 PDF，返回 Markdown、JSON 和输出文件路径。

        Args:
            pdf_path: 本地 PDF 文件路径。
            output_dir: MinerU 输出目录。为空时写入 `database/pdf_ocr_results/`。
            backend: MinerU 后端。`auto` 不传 `-b`，CPU 环境通常使用 `pipeline`。
            timeout: 子进程超时时间，单位秒。
            extra_args: 透传给 MinerU CLI 的额外参数，例如 `["--api-url", "..."]`。
            flatten_output: 是否把 MinerU 生成的 `hybrid_txt`、`pipeline_txt` 等
                方法子目录内容提升到以 PDF 文件名命名的目录下。
        """
        pdf = _ensure_pdf_path(pdf_path)
        run_output_dir = Path(output_dir) if output_dir else self.output_dir
        run_output_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_command(pdf, run_output_dir, backend, extra_args)
        logger.info("正在调用 MinerU 解析 PDF: %s", " ".join(command))

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "MinerU PDF 解析失败，"
                f"exit_code={completed.returncode}, stderr={completed.stderr.strip()}"
            )

        collect_dir = self._flatten_mineru_output_dir(run_output_dir, pdf) if flatten_output else run_output_dir
        outputs = self._collect_outputs(collect_dir, pdf)
        return {
            "pdf_path": str(pdf),
            "output_dir": str(collect_dir),
            "backend": backend,
            "command": command,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            **outputs,
        }

    def extract_markdown(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        backend: MinerUBackend = "auto",
        timeout: int | None = None,
    ) -> str:
        """使用 MinerU 解析 PDF 并直接返回 Markdown 文本。"""
        result = self.parse_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            backend=backend,
            timeout=timeout,
        )
        markdown = result.get("markdown")
        if not markdown:
            raise RuntimeError(f"MinerU 未生成 Markdown 输出: {result['output_dir']}")
        return markdown


_mineru_pdf_process: MinerUPDFProcess | None = None
_mineru_pdf_process_tools: list[Callable[..., Any]] | None = None


def get_MinerUPDFProcess() -> MinerUPDFProcess:
    global _mineru_pdf_process

    if _mineru_pdf_process is None:
        _mineru_pdf_process = MinerUPDFProcess()

    return _mineru_pdf_process


class LazyMinerUPDFProcess:
    @property
    def output_dir(self) -> Path:
        return get_MinerUPDFProcess().output_dir

    @property
    def timeout(self) -> int:
        return get_MinerUPDFProcess().timeout

    def parse_pdf(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        backend: MinerUBackend = "auto",
        timeout: int | None = None,
        extra_args: list[str] | None = None,
        flatten_output: bool = True,
    ) -> dict[str, Any]:
        return get_MinerUPDFProcess().parse_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            backend=backend,
            timeout=timeout,
            extra_args=extra_args,
            flatten_output=flatten_output,
        )

    def extract_markdown(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        backend: MinerUBackend = "auto",
        timeout: int | None = None,
    ) -> str:
        return get_MinerUPDFProcess().extract_markdown(
            pdf_path=pdf_path,
            output_dir=output_dir,
            backend=backend,
            timeout=timeout,
        )

    def __getattr__(self, name: str):
        return getattr(get_MinerUPDFProcess(), name)


mineru_pdf_process = LazyMinerUPDFProcess()


def get_MinerUPDFProcessTools() -> list[Callable[..., Any]]:
    global _mineru_pdf_process_tools

    if _mineru_pdf_process_tools is None:
        process = get_MinerUPDFProcess()
        _mineru_pdf_process_tools = [
            process.parse_pdf,
            process.extract_markdown,
        ]

    return _mineru_pdf_process_tools


MinerUPDFProcessTools = LazyToolList(get_MinerUPDFProcessTools)
