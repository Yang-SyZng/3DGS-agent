from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_training_examples(
    dataset_path: str | Path,
    mined_negatives: dict[str, list[str]] | None = None,
    *,
    negatives_per_query: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build deterministic query-positive-negative examples from RAGAS rows."""

    if negatives_per_query < 1:
        raise ValueError("negatives_per_query must be positive")
    rows = read_jsonl(dataset_path)
    rng = random.Random(seed)
    candidates = [
        str(context).strip()
        for row in rows
        for context in row.get("reference_contexts", [])
        if str(context).strip()
    ]
    mined_negatives = mined_negatives or {}
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        contexts = [
            str(context).strip()
            for context in row.get("reference_contexts", [])
            if str(context).strip()
        ]
        query = str(row.get("user_input") or "").strip()
        if not query or not contexts:
            continue
        positive = contexts[0]
        sample_id = str(row.get("sample_id") or f"sample:{index}")

        ordered = [
            *mined_negatives.get(sample_id, []),
            *(
                candidates[index + 1 :]
                + candidates[: index + 1]
            ),
        ]
        random_candidates = list(candidates)
        rng.shuffle(random_candidates)
        ordered.extend(random_candidates)

        negatives: list[str] = []
        seen = {positive}
        for candidate in ordered:
            candidate = str(candidate).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            negatives.append(candidate)
            if len(negatives) == negatives_per_query:
                break
        if not negatives:
            continue
        examples.append(
            {
                "sample_id": sample_id,
                "query": query,
                "positive": positive,
                "negatives": negatives,
                "source_documents": row.get("source_documents", []),
                "split": _split_for(sample_id, seed),
            }
        )
    return examples


def _split_for(sample_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build embedding training triples.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mined-negatives", type=Path)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mined = {}
    if args.mined_negatives:
        mined = {
            str(row["sample_id"]): list(row["negatives"])
            for row in read_jsonl(args.mined_negatives)
        }
    examples = build_training_examples(
        args.dataset,
        mined,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
    )
    write_jsonl(args.output, examples)
    print(f"wrote {len(examples)} training examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
