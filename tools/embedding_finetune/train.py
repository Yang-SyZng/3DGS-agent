from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from .dataset import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune an embedding model with InfoNCE loss."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    rows = read_jsonl(args.data)
    rows = [row for row in rows if row.get("split", "train") == "train"]
    if not rows:
        raise ValueError("training data is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)

    def embed(texts: list[str]) -> Any:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return functional.normalize(pooled, dim=-1)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in loader:
            query_vectors = embed([row["query"] for row in batch])
            document_groups = [
                [row["positive"], *row["negatives"]] for row in batch
            ]
            width = max(len(group) for group in document_groups)
            document_groups = [
                group + [group[-1]] * (width - len(group))
                for group in document_groups
            ]
            document_vectors = embed(
                [text for group in document_groups for text in group]
            ).view(len(batch), width, -1)
            logits = torch.einsum(
                "bd,bnd->bn", query_vectors, document_vectors
            ) / args.temperature
            labels = torch.zeros(len(batch), dtype=torch.long, device=device)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"epoch={epoch + 1} loss={running_loss / max(len(loader), 1):.6f} "
            f"steps={math.ceil(len(rows) / args.batch_size)}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
