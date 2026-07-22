from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

from config.settings import setting


DEFAULT_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".html"}
SCENARIOS = {
    "single_hop_specific_query_synthesizer": ("single_paper", "单论文问答"),
    "multi_hop_specific_query_synthesizer": ("multi_paper", "多论文综合分析"),
    "multi_hop_abstract_query_synthesizer": ("open_ended", "开放式问答"),
}
ROW_DEFAULTS = {
    "persona_name": "3D Gaussian Splatting Researcher",
    "query_style": "PERFECT_GRAMMAR",
    "query_length": "MEDIUM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 RAGAS 生成 GS 论文测试集")
    parser.add_argument("--input", type=Path, default=Path("database/parser"))
    parser.add_argument("--output", type=Path, default=Path("GS_RAGAS_DATASET/questions.jsonl"))
    # DEFAULT_EXTENSIONS
    parser.add_argument(
        "--extensions",
        nargs="+",
        default={".md"},
        metavar="EXT",
        help="允许的文件后缀，例如：--extensions md txt pdf",
    )
    parser.add_argument("--samples-per-document", type=int, default=5)
    parser.add_argument("--max-documents", type=int, help="仅处理前 N 篇，便于试跑")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    # --local-llm or --no-local-llm
    parser.add_argument("--local-llm", action=argparse.BooleanOptionalAction, default=True, help="是否使用本地 Ollama，默认启用")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.samples_per_document <= 0:
        parser.error("--samples-per-document 必须大于 0")
    args.extensions = normalize_extensions(args.extensions)
    return args


def require_config(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"缺少 {name}")
    return value


def normalize_extensions(extensions: Sequence[str]) -> set[str]:
    normalized = {
        extension.strip().lower()
        if extension.strip().startswith(".")
        else f".{extension.strip().lower()}"
        for extension in extensions
        if extension.strip()
    }
    if not normalized:
        raise ValueError("至少需要指定一种输入文件格式")
    return normalized


def init_model(local_llm: bool) -> tuple[Any, OpenAILikeEmbedding]:
    api_key = require_config(setting.API_KEY, "API_KEY")
    base_url = require_config(setting.BASE_URL, "BASE_URL")
    if local_llm:
        try:
            from tools.llm_local_service.ollama_service import OllamaServer
        except ImportError as exc:
            raise ImportError(
                "本地 LLM 模式需要安装 Ollama 相关依赖"
            ) from exc
        llm = OllamaServer(setting.Local_Model).create_ollama_llm("LLM")
    else:
        llm = OpenAILike(
            model=require_config(setting.LLM_MODEL_ID, "LLM_MODEL_ID"),
            api_base=base_url,
            api_key=api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=128_000,
        )
    embeddings = OpenAILikeEmbedding(
        model_name=require_config(setting.EMBEDDING_MODEL_ID, "EMBEDDING_MODEL_ID"),
        api_base=base_url,
        api_key=api_key,
    )
    return llm, embeddings


def init_generator(local_llm: bool) -> TestsetGenerator:
    llm, embeddings = init_model(local_llm)
    return TestsetGenerator(
        llm=LlamaIndexLLMWrapper(llm),
        embedding_model=LlamaIndexEmbeddingsWrapper(embeddings),
    )


def resolve_input_files(
    paths: Sequence[Path], recursive: bool, extensions: set[str]
) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在：{path}")
        if path.is_file():
            if path.suffix.lower() not in extensions:
                raise ValueError(f"不支持的文件类型：{path}")
            files.append(path)
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        files.extend(
            candidate
            for candidate in iterator
            if candidate.is_file() and candidate.suffix.lower() in extensions
        )

    unique_files = sorted(set(files))
    if not unique_files:
        supported = ", ".join(sorted(extensions))
        raise ValueError(f"未找到支持的文档，支持：{supported}")
    return unique_files


def generate(
    generator: TestsetGenerator,
    inputs: Sequence[Path],
    size: int,
    *,
    extensions: set[str],
    timeout: int,
    max_retries: int,
    max_workers: int,
    seed: int,
    debug: bool,
) -> Any:
    files = resolve_input_files(inputs, recursive=False, extensions=extensions)
    documents = SimpleDirectoryReader(
        input_files=[str(path) for path in files],
        filename_as_id=True,
        raise_on_error=True,
    ).load_data()
    if not documents:
        raise ValueError("文档加载结果为空")

    run_config = RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_workers=max_workers,
        seed=seed,
    )
    return generator.generate_with_llamaindex_docs(
        documents,
        testset_size=size,
        run_config=run_config,
        with_debugging_logs=debug,
        raise_exceptions=True,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_defaults(row: dict[str, Any]) -> dict[str, Any]:
    for key, value in ROW_DEFAULTS.items():
        row.setdefault(key, value)
    return row


def enrich(row: dict[str, Any], primary: str, partner: str, number: int) -> dict[str, Any]:
    synthesizer = str(row.get("synthesizer_name", ""))
    scenario, scenario_name = SCENARIOS.get(synthesizer, ("open_ended", "开放式问答"))
    row["scenario"] = scenario
    row["scenario_name"] = scenario_name
    row["generation_documents"] = [primary]
    row["source_documents"] = [primary] if scenario == "single_paper" else [primary, partner]
    row["sample_id"] = f"{Path(primary).stem}:{number}"
    row.setdefault(
        "task_type",
        {"single_paper": "论文检索", "multi_paper": "方法对比", "open_ended": "技术演进"}[scenario],
    )
    return apply_defaults(row)


def main() -> int:
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    all_paths = resolve_input_files(
        [input_root], recursive=True, extensions=args.extensions
    )
    paths = all_paths[: args.max_documents] if args.max_documents else all_paths
    input_base = input_root if input_root.is_dir() else input_root.parent
    path_positions = {path: index for index, path in enumerate(all_paths)}

    output = args.output.expanduser().resolve()
    rows = read_jsonl(output) if args.resume else []
    for row in rows:
        apply_defaults(row)
    if args.resume and rows:
        write_jsonl(output, rows)
    elif not args.resume:
        write_jsonl(output, [])

    sample_counts: dict[str, int] = {}
    for row in rows:
        for document in set(row.get("generation_documents", [])):
            sample_counts[document] = sample_counts.get(document, 0) + 1

    failures: list[dict[str, str]] = []
    generator = init_generator(args.local_llm)
    print(
        f"发现 {len(paths)} 篇；已有 {len(rows)} 条；使用 RAGAS TestsetGenerator",
        flush=True,
    )
    for index, path in enumerate(paths):
        primary = path.relative_to(input_base).as_posix()
        existing_count = sample_counts.get(primary, 0)
        remaining = (
            args.samples_per_document - existing_count
            if args.resume
            else args.samples_per_document
        )
        if remaining <= 0:
            print(f"[{index + 1}/{len(paths)}] SKIP {primary}", flush=True)
            continue

        primary_position = path_positions[path]
        partner_path = all_paths[(primary_position + 1) % len(all_paths)]
        partner = partner_path.relative_to(input_base).as_posix()
        inputs = [path] if len(all_paths) == 1 else [path, partner_path]
        print(f"[{index + 1}/{len(paths)}] START {primary}: 生成 {remaining} 条", flush=True)
        try:
            dataset = generate(
                generator,
                inputs,
                remaining,
                extensions=args.extensions,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_workers=args.max_workers,
                seed=args.seed,
                debug=args.debug,
            )
            generated = dataset.to_list()[:remaining]
            start = existing_count + 1
            generated_rows = [
                enrich(row, primary, partner, number)
                for number, row in enumerate(generated, start)
            ]
            append_jsonl(output, generated_rows)
            rows.extend(generated_rows)
            sample_counts[primary] = existing_count + len(generated_rows)
            print(
                f"[{index + 1}/{len(paths)}] OK {primary}: +{len(generated)}，累计 {len(rows)}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"document": primary, "error": f"{type(exc).__name__}: {exc}"})
            print(
                f"[{index + 1}/{len(paths)}] FAIL {primary}: {type(exc).__name__}: {exc}",
                flush=True,
            )

    failures_path = output.with_suffix(".failures.json")
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SUMMARY samples={len(rows)} failures={len(failures)} output={output}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
