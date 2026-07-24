from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _paper_ids(client: Any, collection_name: str, row_count: int) -> set[str]:
    paper_ids: set[str] = set()
    if hasattr(client, "query_iterator"):
        iterator = client.query_iterator(
            collection_name=collection_name,
            filter="paper_id != ''",
            output_fields=["paper_id"],
            batch_size=1000,
        )
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                paper_ids.update(
                    str(row["paper_id"]).strip().lower()
                    for row in rows
                    if row.get("paper_id")
                )
        finally:
            iterator.close()
        return paper_ids

    rows = client.query(
        collection_name=collection_name,
        filter="paper_id != ''",
        output_fields=["paper_id"],
        limit=max(row_count, 1),
    )
    return {
        str(row["paper_id"]).strip().lower()
        for row in rows
        if row.get("paper_id")
    }


def collect_index_stats(
    client: Any,
    collection_name: str,
    parser_root: str | Path,
) -> dict[str, Any]:
    """Collect auditable parser and Milvus counts without changing the index."""

    stats = client.get_collection_stats(collection_name=collection_name)
    row_count = int(stats.get("row_count", 0))
    paper_ids = _paper_ids(client, collection_name, row_count)
    parser_path = Path(parser_root)
    parsed_papers = (
        sum(
            1
            for directory in parser_path.iterdir()
            if directory.is_dir() and any(directory.glob("*.md"))
        )
        if parser_path.exists()
        else 0
    )
    return {
        "collection": collection_name,
        "indexed_chunks": row_count,
        "indexed_papers": len(paper_ids),
        "parsed_papers": parsed_papers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report parsed-paper and Milvus index counts."
    )
    parser.add_argument("--uri")
    parser.add_argument("--token")
    parser.add_argument("--collection")
    parser.add_argument("--parser-root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from pymilvus import MilvusClient

    from config import setting

    client = MilvusClient(
        uri=args.uri or setting.Milvus_uri,
        token=args.token if args.token is not None else setting.Milvus_token,
    )
    report = collect_index_stats(
        client,
        args.collection or setting.Milvus_collection_name,
        args.parser_root or setting.pdf_parser_save_dir,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
