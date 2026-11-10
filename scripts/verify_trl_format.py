#!/usr/bin/env python
"""Verify our DPO examples against TRL's documented conversational preference format on a real batch.

Usage: python scripts/verify_trl_format.py [--mode smoke] [--condition E1_random] [--n 8]

Checks (results written to reports/trl_format_check.md):
  1. trl.data_utils.is_conversational() is True for every example.
  2. Our conversion equals TRL's own `extract_prompt` applied to the raw (implicit-prompt) rows.
  3. With the real Qwen2.5 tokenizer, the rendered prompt contains the user turn exactly once and is a
     prefix of prompt+chosen and prompt+rejected.
  4. DPOTrainer's own tokenization (`_prepare_dataset`) keeps the full prompt (no prefix divergence) and
     the completion tokens decode to exactly the assistant text + end-of-turn.
  5. One real collated batch runs a forward/backward DPO step with finite loss (tiny random model that
     shares Qwen2.5's vocabulary; this checks plumbing only, not quality).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from dpo_margin import data, selection  # noqa: E402
from dpo_margin.prepare import load_prep_config  # noqa: E402

DOCS_URL = "https://huggingface.co/docs/trl/v1.17.1/en/dataset_formats"
TINY_MODEL = ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "edca82771e1fad8f0b7ce110ffe770c682043fec")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--condition", default="E1_random")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    import trl
    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer, extract_prompt
    from trl.data_utils import is_conversational

    scfg = yaml.safe_load((ROOT / "configs/data/selection.yaml").read_text())
    pcfg = load_prep_config(ROOT / "configs/data/prep.yaml")
    meta = json.loads((ROOT / pcfg["outputs"]["splits_dir"] / "prep_metadata.json").read_text())
    manifest = selection.read_manifest(ROOT / selection.manifest_dir(scfg["manifest_root"], args.mode)
                                       / f"{args.condition}.jsonl").head(args.n)
    raw = load_dataset(meta["dataset_name"], revision=meta["dataset_revision"], split="train")
    ours = data.materialize(raw, manifest)
    raw_rows = raw.select(manifest["orig_index"].tolist())
    tok = AutoTokenizer.from_pretrained(pcfg["tokenizer_name"], revision=pcfg["tokenizer_revision"])

    results: dict[str, bool] = {}
    results["is_conversational"] = all(is_conversational(ex) for ex in ours)
    results["matches_trl_extract_prompt"] = all(
        extract_prompt({"chosen": r["chosen"], "rejected": r["rejected"]}) ==
        {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
        for r, ex in zip(raw_rows, ours))
    results["no_user_turn_in_completions"] = all(
        all(m["role"] == "assistant" for m in ex["chosen"] + ex["rejected"]) for ex in ours)

    once, prefix = True, True
    for ex in ours:
        p = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        pc = tok.apply_chat_template(ex["prompt"] + ex["chosen"], tokenize=False)
        pr = tok.apply_chat_template(ex["prompt"] + ex["rejected"], tokenize=False)
        n_user = sum(m["role"] == "user" for m in ex["prompt"])
        once &= pc.count("<|im_start|>user") == n_user and pr.count("<|im_start|>user") == n_user
        prefix &= pc.startswith(p) and pr.startswith(p)
    results["user_turns_rendered_once"] = once
    results["prompt_is_prefix_of_completions"] = prefix

    out_dir = ROOT / "results" / "smoke" / "trl_format_check"
    trainer = DPOTrainer(
        model=TINY_MODEL[0],
        args=DPOConfig(output_dir=str(out_dir), use_cpu=True, max_steps=1, per_device_train_batch_size=4,
                       max_length=2048, report_to="none", save_strategy="no", seed=0, logging_steps=1,
                       gradient_checkpointing=False, model_init_kwargs={"revision": TINY_MODEL[1], "dtype": "float32"}),
        train_dataset=Dataset.from_list(list(ours)),
        processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
    )
    tokenized = trainer.train_dataset
    full_prompt, exact_completion = True, True
    for ex, t in zip(ours, tokenized):
        p_str = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        full_prompt &= list(t["prompt_ids"]) == tok(p_str, add_special_tokens=False)["input_ids"]
        exact_completion &= tok.decode(t["chosen_ids"]) == ex["chosen"][0]["content"] + "<|im_end|>\n"
        exact_completion &= tok.decode(t["rejected_ids"]) == ex["rejected"][0]["content"] + "<|im_end|>\n"
    results["trl_keeps_full_prompt"] = full_prompt
    results["trl_completion_tokens_exact"] = exact_completion

    batch = next(iter(trainer.get_train_dataloader()))
    shapes = {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
    trainer.train()
    loss = next(h["loss"] for h in trainer.state.log_history if "loss" in h)
    results["real_batch_step_finite_loss"] = math.isfinite(float(loss))

    first = tokenized[0]
    lines = [
        "# TRL format check", "",
        f"- TRL version: `{trl.__version__}`; documentation checked: <{DOCS_URL}> "
        "(DPOTrainer expects a *preference* dataset, explicit prompt recommended; conversational form uses "
        "`prompt` = list of messages, `chosen`/`rejected` = list with the assistant message).",
        f"- Batch: first {len(ours)} rows of `data/manifests/{args.mode}/{args.condition}.jsonl`; tokenizer "
        f"`{pcfg['tokenizer_name']}` @ `{pcfg['tokenizer_revision'][:12]}`.",
        f"- Collated batch tensor shapes: `{shapes}`; one DPO step loss = {float(loss):.4f} "
        "(tiny random-weight model; plumbing check only).", "",
        "| check | passed |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in results.items()], "",
        "## First example as tokenized by DPOTrainer", "",
        "**prompt_ids decoded:**", "", "```text", tok.decode(first["prompt_ids"])[:1500], "```", "",
        "**chosen_ids decoded (first 400 chars):**", "", "```text", tok.decode(first["chosen_ids"])[:400], "```", "",
        "**rejected_ids decoded (first 400 chars):**", "", "```text", tok.decode(first["rejected_ids"])[:400], "```", "",
    ]
    report = ROOT / "reports" / "trl_format_check.md"
    report.write_text("\n".join(lines) + "\n")
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"batch shapes {shapes}; loss {float(loss):.4f}; report -> {report.relative_to(ROOT)}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
