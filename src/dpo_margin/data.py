"""Conversion to TRL's conversational preference format, eval-set construction, and materialization.

TRL 1.17 `DPOTrainer` expects a *preference* dataset, explicit prompt recommended
(https://huggingface.co/docs/trl/v1.17.1/en/dataset_formats):
    {"prompt":   [{"role": "user", "content": ...}, ...],      # shared context, any number of turns
     "chosen":   [{"role": "assistant", "content": ...}],
     "rejected": [{"role": "assistant", "content": ...}]}
The argilla rows are in TRL's *implicit-prompt* form (context repeated inside chosen/rejected), so the
shared context must be moved out, or the prompt would appear twice in the training sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dpo_margin.prepare import pair_sha256, sha256_file

VALID_ROLES = {"system", "user", "assistant"}


def _clean(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def to_dpo_example(row: dict) -> dict:
    """Split [context..., final assistant turn] into prompt / chosen / rejected.

    The full shared context (system, earlier user and assistant turns) is preserved in `prompt`.
    Raises if chosen and rejected do not share exactly the same context, if the context does not end
    with a user turn, or if the final turns are not assistant messages.
    """
    chosen, rejected = row["chosen"], row["rejected"]
    if len(chosen) < 2 or len(rejected) < 2:
        raise ValueError("Expected at least [user, assistant] in chosen/rejected")
    context_c, context_r = _clean(chosen[:-1]), _clean(rejected[:-1])
    if context_c != context_r:
        raise ValueError("chosen and rejected do not share the same context")
    if any(m["role"] not in VALID_ROLES for m in context_c):
        raise ValueError(f"Unknown role in context: {[m['role'] for m in context_c]}")
    if context_c[-1]["role"] != "user":
        raise ValueError("Context must end with a user turn")
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        raise ValueError("Last turn of chosen/rejected must be the assistant response")
    if "prompt" in row and isinstance(row["prompt"], str) and context_c[-1]["content"] != row["prompt"]:
        raise ValueError("Final user turn differs from the row's `prompt` field")
    return {
        "prompt": context_c,
        "chosen": [{"role": "assistant", "content": chosen[-1]["content"]}],
        "rejected": [{"role": "assistant", "content": rejected[-1]["content"]}],
    }


def build_eval_manifest(splits_dir: Path, size: int, seed: int) -> pd.DataFrame:
    """Seeded validation subset, shared by all conditions. Reads validation.csv only."""
    splits_dir = Path(splits_dir)
    sums = dict(line.split("  ", 1)[::-1] for line in (splits_dir / "SHA256SUMS").read_text().splitlines())
    if sha256_file(splits_dir / "validation.csv") != sums.get("validation.csv"):
        raise RuntimeError("validation.csv does not match SHA256SUMS")
    val = pd.read_csv(splits_dir / "validation.csv").sort_values("orig_index").reset_index(drop=True)
    if size > len(val):
        raise ValueError(f"eval size {size} exceeds {len(val)} validation pairs")
    idx = np.sort(np.random.default_rng([seed, 1]).choice(len(val), size=size, replace=False))
    return val.iloc[idx].assign(condition="eval_validation")


def write_jsonl(df: pd.DataFrame, path: Path, fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in df[fields].to_dict(orient="records"):
            f.write(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}) + "\n")
    return sha256_file(path)


def materialize(dataset, manifest: pd.DataFrame):
    """Fetch manifest rows from the raw dataset, verify pair hashes, convert to TRL format."""
    subset = dataset.select(manifest["orig_index"].astype(int).tolist())
    for row, expected in zip(subset, manifest["pair_sha256"]):
        got = pair_sha256(row["prompt"], row["chosen"][-1]["content"], row["rejected"][-1]["content"])
        if got != expected:
            raise RuntimeError("Pair hash mismatch: the dataset revision differs from the manifest")
    return subset.map(to_dpo_example, remove_columns=subset.column_names)
