#!/usr/bin/env python
"""Validate, clean, featurize and split the preference dataset.

Usage: python scripts/prepare_data.py [--config configs/data/prep.yaml]
Outputs (paths from the config): split manifests + SHA256SUMS + prep_metadata.json, the per-row
feature table, and a markdown report of manually inspectable examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dpo_margin import prepare  # noqa: E402
from dpo_margin.data import to_dpo_example  # noqa: E402


def _clip(text: str, n: int = 400) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + f" … [+{len(text) - n} chars]"


def inspection_report(dataset, n: int, seed: int) -> tuple[str, dict]:
    """Seeded sample (one per source first, then random) with per-example structural checks."""
    rng = np.random.default_rng(seed)
    sources = dataset["source"]
    picked = []
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        picked.append(int(rng.choice(idx)))
    taken = set(picked)
    rest = [i for i in range(len(dataset)) if i not in taken]
    picked += [int(i) for i in rng.choice(rest, size=max(0, n - len(picked)), replace=False)]
    picked = sorted(picked[:max(n, len(picked))])

    lines = ["# Example inspection", "",
             f"{len(picked)} examples drawn with seed {seed} (one per source, then uniformly at random).",
             "Checks: user turn identical in chosen/rejected and equal to `prompt`; assistant responses",
             "differ; TRL-formatted example contains the prompt once (only in `prompt`).", ""]
    checks = {"n": len(picked), "shared_prompt": 0, "distinct_responses": 0, "trl_prompt_once": 0}
    for i in picked:
        r = dataset[i]
        shared = r["chosen"][0] == r["rejected"][0] and r["chosen"][0]["content"] == r["prompt"]
        distinct = r["chosen"][-1]["content"].strip() != r["rejected"][-1]["content"].strip()
        trl = to_dpo_example(r)
        once = (all(m["role"] == "assistant" for m in trl["chosen"] + trl["rejected"])
                and len(trl["prompt"]) == 1 and trl["prompt"][0]["content"] == r["prompt"])
        checks["shared_prompt"] += shared
        checks["distinct_responses"] += distinct
        checks["trl_prompt_once"] += once
        lines += [
            f"## Row {i} — source `{r['source']}`", "",
            f"- chosen: `{r['chosen-model']}` rating {r['chosen-rating']} | rejected: `{r['rejected-model']}` "
            f"rating {r['rejected-rating']} | margin {r['chosen-rating'] - r['rejected-rating']}",
            f"- roles chosen={[m['role'] for m in r['chosen']]} rejected={[m['role'] for m in r['rejected']]}",
            f"- shared prompt: {shared} | distinct responses: {distinct} | TRL prompt appears once: {once}", "",
            f"**Prompt:** {_clip(r['prompt'])}", "",
            f"**Chosen:** {_clip(r['chosen'][-1]['content'])}", "",
            f"**Rejected:** {_clip(r['rejected'][-1]['content'])}", "",
        ]
    return "\n".join(lines) + "\n", checks


def full_structure_audit(dataset) -> dict:
    """Structural checks over every raw row (independent of the cleaning reasons)."""
    audit = {"n_rows": len(dataset), "turn_patterns": {}, "user_turn_equals_prompt": 0,
             "chosen_rejected_same_user_turn": 0, "identical_assistant_responses": 0}
    for r in dataset:
        key = f"{[m['role'] for m in r['chosen']]} / {[m['role'] for m in r['rejected']]}"
        audit["turn_patterns"][key] = audit["turn_patterns"].get(key, 0) + 1
        audit["user_turn_equals_prompt"] += r["chosen"][0]["content"] == r["prompt"] == r["rejected"][0]["content"]
        audit["chosen_rejected_same_user_turn"] += r["chosen"][:-1] == r["rejected"][:-1]
        audit["identical_assistant_responses"] += (
            r["chosen"][-1]["content"].strip() == r["rejected"][-1]["content"].strip())
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/prep.yaml")
    args = parser.parse_args()
    cfg = prepare.load_prep_config(args.config)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(cfg["dataset_name"], revision=cfg["dataset_revision"], split="train")
    problems = prepare.check_schema(ds.features)
    print(f"Loaded {len(ds)} rows; columns: {ds.column_names}")
    if problems:
        print("Schema problems:", *problems, sep="\n  ")
        return 1

    audit = full_structure_audit(ds)
    print("Structure audit:", json.dumps(audit))

    report, checks = inspection_report(ds, cfg["inspection"]["n_examples"], cfg["inspection"]["seed"])
    out = ROOT / cfg["outputs"]["inspection_report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Inspected {checks['n']} examples -> {out.relative_to(ROOT)}: {checks}")

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], revision=cfg["tokenizer_revision"])
    df, meta = prepare.prepare(ds, tok, cfg)
    meta = {
        "dataset_name": cfg["dataset_name"], "dataset_revision": cfg["dataset_revision"],
        "tokenizer_name": cfg["tokenizer_name"], "tokenizer_revision": cfg["tokenizer_revision"],
        "schema": {k: str(v) for k, v in ds.features.items()},
        "structure_audit": audit, "inspection_checks": checks, **meta,
    }
    meta = prepare.write_outputs(df, meta, ROOT / cfg["outputs"]["splits_dir"], ROOT / cfg["outputs"]["features_path"])

    print(f"Kept {meta['n_kept']} / {meta['n_raw']}; dropped by reason: {meta['dropped_by_reason']}")
    print(f"Duplicates: {meta['duplicates']}")
    print(f"Split pairs: {meta['split_pairs']}")
    print(f"Train-only thresholds {meta['bucket_thresholds_from_train']}; bucket counts {meta['bucket_counts']}")
    print("Manifest SHA256:", json.dumps(meta["manifest_sha256"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
