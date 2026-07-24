from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import read_jsonl


def retrieval_metrics(
    rankings: dict[str, list[str]],
    positives: dict[str, str],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if not positives:
        raise ValueError("positives cannot be empty")
    result: dict[str, float] = {}
    for k in ks:
        hits = sum(
            positives[query_id] in rankings.get(query_id, [])[:k]
            for query_id in positives
        )
        result[f"recall@{k}"] = hits / len(positives)
    reciprocal_ranks = []
    for query_id, positive_id in positives.items():
        try:
            rank = rankings.get(query_id, []).index(positive_id) + 1
        except ValueError:
            rank = 0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    result["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return result


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(
        mask.sum(1), min=1e-9
    )


def encode(model: Any, tokenizer: Any, texts: list[str], batch_size: int) -> Any:
    import torch
    import torch.nn.functional as functional

    vectors = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            output = model(**batch)
            vectors.append(
                functional.normalize(
                    _mean_pool(output.last_hidden_state, batch["attention_mask"]),
                    dim=-1,
                ).cpu()
            )
    return torch.cat(vectors)


def evaluate_model(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int = 8,
) -> dict[str, float]:
    import torch

    documents = list(
        dict.fromkeys(
            text
            for row in rows
            for text in [row["positive"], *row["negatives"]]
        )
    )
    document_ids = {text: f"d{index}" for index, text in enumerate(documents)}
    query_ids = [str(row["sample_id"]) for row in rows]
    query_vectors = encode(model, tokenizer, [row["query"] for row in rows], batch_size)
    document_vectors = encode(model, tokenizer, documents, batch_size)
    scores = torch.matmul(query_vectors, document_vectors.T)
    rankings = {
        query_id: [
            document_ids[documents[index]]
            for index in torch.argsort(scores[row_index], descending=True).tolist()
        ]
        for row_index, query_id in enumerate(query_ids)
    }
    positives = {
        str(row["sample_id"]): document_ids[row["positive"]] for row in rows
    }
    return retrieval_metrics(rankings, positives)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate embedding Recall@K and MRR.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.cuda() if __import__("torch").cuda.is_available() else model
    rows = [
        row
        for row in read_jsonl(args.data)
        if row.get("split", args.split) == args.split
    ]
    if not rows:
        raise ValueError(f"no examples found for split={args.split}")
    metrics = evaluate_model(model, tokenizer, rows, args.batch_size)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
