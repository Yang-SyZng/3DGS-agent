from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .dataset import read_jsonl, write_jsonl


async def mine(dataset: Path, output: Path, top_k: int) -> int:
    from rag.embedding import Embedding
    from rag.vector import MilvusHybridClient

    rows = read_jsonl(dataset)
    vector_store = MilvusHybridClient()
    embedding = Embedding()
    mined = []
    for index, row in enumerate(rows):
        query = str(row.get("user_input") or "").strip()
        if not query:
            continue
        positives = {
            str(context).strip() for context in row.get("reference_contexts", [])
        }
        results = await vector_store.search(
            query,
            embed_model=embedding,
            top_k=top_k,
        )
        negatives = []
        for item in results:
            node = getattr(item, "node", item)
            text = str(getattr(node, "text", "")).strip()
            if text and text not in positives:
                negatives.append(text)
        mined.append(
            {
                "sample_id": str(row.get("sample_id") or f"sample:{index}"),
                "negatives": list(dict.fromkeys(negatives)),
            }
        )
    write_jsonl(output, mined)
    return len(mined)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from the current Milvus index."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    count = asyncio.run(mine(args.dataset, args.output, args.top_k))
    print(f"wrote hard negatives for {count} queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

