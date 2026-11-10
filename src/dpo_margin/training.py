"""Support code for scripts/train_dpo.py: data loading, run instrumentation, and correctness checks.

Nothing here changes the DPO objective; the loss is TRL's standard sigmoid DPO loss.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import TrainerCallback

from dpo_margin import data, selection

END_OF_TURN = "<|im_end|>\n"  # Qwen2.5 chat template: every assistant turn ends with this


# --------------------------------------------------------------------------------------------- data
def load_run_data(root: Path, cfg: dict) -> dict:
    """Manifests for this run (checksum- and size-verified) materialized as TRL conversational datasets."""
    from datasets import load_dataset

    dcfg, run_type = cfg["data"], cfg["run_type"]
    scfg = yaml.safe_load((root / dcfg["selection_config"]).read_text())
    mdir = root / selection.manifest_dir(scfg["manifest_root"], run_type)
    manifest_file = mdir / f"{dcfg['condition']}.jsonl"
    if not (mdir / "selection_metadata.json").exists() or not manifest_file.exists():
        raise SystemExit(f"Missing {manifest_file}. Run: python scripts/select_pairs.py --mode {run_type}")
    sel_meta = json.loads((mdir / "selection_metadata.json").read_text())
    eval_file = root / sel_meta["eval_manifest"]["path"]
    for path, digest in [(manifest_file, sel_meta["manifest_sha256"][manifest_file.name]),
                         (eval_file, sel_meta["eval_manifest"]["sha256"])]:
        if data.sha256_file(path) != digest:
            raise SystemExit(f"{path} does not match the SHA-256 in selection_metadata.json")
    train_rows = selection.read_manifest(manifest_file)
    eval_rows = selection.read_manifest(eval_file).head(dcfg.get("eval_limit") or None)
    if len(train_rows) != scfg["sizes"][run_type]:
        raise SystemExit(f"{manifest_file} has {len(train_rows)} rows, expected {scfg['sizes'][run_type]}")

    raw = load_dataset(sel_meta["dataset_name"], revision=sel_meta["dataset_revision"], split="train")
    return {
        "train": data.materialize(raw, train_rows),
        "eval": data.materialize(raw, eval_rows),
        "info": {
            "train_manifest": str(manifest_file.relative_to(root)),
            "train_manifest_sha256": data.sha256_file(manifest_file),
            "n_train_pairs_manifest": len(train_rows),
            "eval_manifest": str(eval_file.relative_to(root)),
            "eval_manifest_sha256": data.sha256_file(eval_file),
            "n_eval_pairs": len(eval_rows),
            "dataset_name": sel_meta["dataset_name"],
            "dataset_revision": sel_meta["dataset_revision"],
            "split_manifest_sha256": sel_meta["split_manifest_sha256"],
            "selection_seed": sel_meta["seed"],
            "eval_seed": scfg["eval"]["seed"],
        },
    }


def length_stats(tokenized, n_before: int, max_length: int | None) -> dict:
    """Truncation/drop statistics of DPOTrainer's tokenized dataset (prompt_ids, chosen_ids, rejected_ids)."""
    p = [len(x) for x in tokenized["prompt_ids"]]
    c = [len(x) for x in tokenized["chosen_ids"]]
    r = [len(x) for x in tokenized["rejected_ids"]]
    limit = max_length or math.inf
    trunc_c = sum(pi + ci > limit for pi, ci in zip(p, c))
    trunc_r = sum(pi + ri > limit for pi, ri in zip(p, r))
    trunc_any = sum((pi + ci > limit) or (pi + ri > limit) for pi, ci, ri in zip(p, c, r))
    kept = [min(pi + ci, limit) + min(pi + ri, limit) for pi, ci, ri in zip(p, c, r)]
    return {
        "n_pairs_in": n_before,
        "n_pairs_after_trl_filter": len(p),
        "n_pairs_dropped_prompt_too_long": n_before - len(p),
        "n_pairs_truncated": trunc_any,
        "n_chosen_truncated": trunc_c,
        "n_rejected_truncated": trunc_r,
        "tokens_per_epoch": int(sum(kept)),
        "max_untruncated_sequence": max(max(pi + ci, pi + ri) for pi, ci, ri in zip(p, c, r)) if p else 0,
    }


# ------------------------------------------------------------------------------------ correctness
def check_completion_masking(trainer, tokenizer, n: int = 4) -> dict:
    """Collate real examples exactly as training does and check what the loss sees.

    TRL builds labels as input_ids with every completion_mask == 0 position set to -100, so the DPO
    log-probabilities are summed over completion tokens only. We verify, per sequence, that the
    masked-in tokens decode to exactly the final assistant message + end-of-turn (or a prefix of it
    when truncated), and that the unmasked prefix is the official chat-template rendering of the prompt.
    """
    n = min(n, len(trainer.train_dataset))
    rows = [trainer.train_dataset[i] for i in range(n)]
    batch = trainer.data_collator(rows)
    ids, cmask, amask = batch["input_ids"], batch["completion_mask"], batch["attention_mask"]
    ok_completion, ok_prompt, ok_padding = True, True, True
    for i in range(n):
        ex = rows[i]  # TRL keeps the conversational columns next to the token ids
        prompt_text = tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        for half, key in ((0, "chosen"), (1, "rejected")):
            j = half * n + i
            m = cmask[j].bool()
            completion = tokenizer.decode(ids[j][m])
            target = ex[key][0]["content"] + END_OF_TURN
            ok_completion &= completion == target or (len(completion) < len(target) and target.startswith(completion))
            # TRL drops pairs whose prompt alone reaches max_length, so every kept prompt is complete.
            first = int(m.nonzero()[0]) if m.any() else int(amask[j].sum())
            ok_prompt &= tokenizer.decode(ids[j][:first]) == prompt_text
            ok_padding &= not (cmask[j].bool() & ~amask[j].bool()).any()
    result = {"n_checked_pairs": n, "completion_tokens_are_assistant_reply": bool(ok_completion),
              "prompt_tokens_excluded_from_loss": bool(ok_prompt), "padding_excluded_from_loss": bool(ok_padding)}
    if not all(v for k, v in result.items() if k != "n_checked_pairs"):
        raise RuntimeError(f"completion-mask check failed: {result}")
    return result


def fingerprint(named_tensors) -> str:
    """SHA-256 over (name, dtype, shape, raw bytes) of tensors, in name order."""
    h = hashlib.sha256()
    for name, t in sorted(named_tensors, key=lambda x: x[0]):
        t = t.detach().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.view(torch.uint8).cpu().numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def base_and_adapter_fingerprints(model) -> dict:
    """Separate fingerprints of the frozen base weights (= DPO reference policy) and the LoRA weights."""
    base, lora = [], []
    for name, p in model.named_parameters():
        (lora if "lora_" in name else base).append((name, p))
    return {"base_weights_sha256": fingerprint(base), "lora_weights_sha256": fingerprint(lora),
            "n_base_params": sum(p.numel() for _, p in base), "n_lora_params": sum(p.numel() for _, p in lora),
            "n_trainable_params": sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)}


@torch.no_grad()
def completion_logps(model, batch: dict, disable_adapter: bool = False) -> list[float]:
    """Summed log-prob of completion tokens per sequence (same quantity TRL's DPO loss uses)."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for row in range(batch["input_ids"].shape[0]):  # one unpadded sequence at a time: full-vocab logits are large
        n = int(batch["attention_mask"][row].sum())
        ids = batch["input_ids"][row, :n][None].to(dev)
        cm = batch["completion_mask"][row, 1:n].to(dev)
        with model.disable_adapter() if disable_adapter else _nullcontext():
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
        out.append(float((lp * cm).sum()))
        del logits, lp
    model.train(was_training)
    return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def probe_batch(trainer, n: int = 2) -> dict:
    """A fixed collated batch from the (held-out) eval set, used for before/after/reload comparisons."""
    ds = trainer.eval_dataset if trainer.eval_dataset is not None else trainer.train_dataset
    return trainer.data_collator([ds[i] for i in range(min(n, len(ds)))])


# ------------------------------------------------------------------------------------ instrumentation
def memory_stats(backend: str) -> dict:
    out = {"process_peak_rss_gb": _peak_rss_gb()}
    if backend == "cuda":
        out.update(gpu_mem_allocated_gb=torch.cuda.memory_allocated() / 1024**3,
                   gpu_mem_peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                   gpu_mem_peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3)
    elif backend == "mps":
        out.update(gpu_mem_allocated_gb=torch.mps.current_allocated_memory() / 1024**3,
                   gpu_mem_driver_gb=torch.mps.driver_allocated_memory() / 1024**3)
    return out


def _peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2  # bytes on macOS, KiB on Linux


class RunLogger(TrainerCallback):
    """Writes history.csv after every log event: TRL metrics + step, wall clock, pairs seen, memory.

    MPS has no peak-memory counter, so peaks are tracked by sampling after every optimizer step.
    """

    def __init__(self, path: Path, backend: str, pairs_per_step: int):
        self.path, self.backend, self.pairs_per_step = path, backend, pairs_per_step
        self.rows: list[dict] = []
        self.t0 = time.time()
        self.peak: dict[str, float] = {}

    def _sample(self):
        for k, v in memory_stats(self.backend).items():
            self.peak[k] = max(self.peak.get(k, 0.0), v)

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()
        if self.backend == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kw):
        self._sample()

    def on_log(self, args, state, control, logs=None, **kw):
        self._sample()
        now = memory_stats(self.backend)
        row = {"step": state.global_step, "epoch": state.epoch, "wall_clock_s": round(time.time() - self.t0, 2),
               "train_pairs_seen": state.global_step * self.pairs_per_step,
               **{k: round(v, 4) for k, v in now.items()},
               **{f"sampled_max_{k}": round(v, 4) for k, v in self.peak.items() if "peak" not in k},
               **(logs or {})}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        losses = [v for k, v in (logs or {}).items() if k in ("loss", "eval_loss")]
        if any(not math.isfinite(float(v)) for v in losses):
            control.should_training_stop = True
            raise FloatingPointError(f"non-finite loss at step {state.global_step}: {logs}")


def sha256_dir(path: Path) -> dict[str, str]:
    return {p.name: data.sha256_file(p) for p in sorted(path.iterdir()) if p.is_file()}
