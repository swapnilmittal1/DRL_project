"""Controlled, source-matched selection of DPO training pairs from the train split.

Design:
  * Pool: data/splits/train.csv only (checksum-verified). Validation/test files are never opened.
  * Every condition draws exactly N pairs. With `match_source`, each condition receives the same
    per-source quota, computed from the train pool's source proportions by largest-remainder
    rounding; E1 (random) is therefore a source-stratified random sample of the whole pool, and
    E2-E4 reproduce exactly the same source composition inside their margin bucket.
  * If any (condition, source) cell has fewer eligible pairs than its quota, selection fails with
    the full feasibility table instead of shrinking or re-allocating the sample.
  * One seed for all conditions; per-source RNG streams are derived from (seed, source name).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from dpo_margin.prepare import assign_bucket, compute_thresholds, sha256_file

MANIFEST_FIELDS = [
    "condition", "row_id", "orig_index", "prompt_sha256", "pair_sha256", "source", "margin", "bucket",
    "chosen_rating", "rejected_rating", "chosen_model", "rejected_model",
    "prompt_tokens", "chosen_tokens", "rejected_tokens", "token_length_diff",
]


class SelectionError(RuntimeError):
    """Selection cannot meet its constraints; the message contains the diagnostic table."""


def _stream(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)])


def load_train_pool(splits_dir: Path) -> pd.DataFrame:
    """Load and validate the train split. Only train.csv (and its checksum) is read."""
    splits_dir = Path(splits_dir)
    sums = dict(line.split("  ", 1)[::-1] for line in (splits_dir / "SHA256SUMS").read_text().splitlines())
    if sha256_file(splits_dir / "train.csv") != sums.get("train.csv"):
        raise SelectionError("train.csv does not match SHA256SUMS; re-run scripts/prepare_data.py")
    pool = pd.read_csv(splits_dir / "train.csv")
    if not pool["prompt_sha256"].is_unique:
        dups = int(pool["prompt_sha256"].duplicated().sum())
        raise SelectionError(f"train pool has {dups} duplicate normalized prompts; duplicates would inflate conditions")
    if not (pool["margin"] > 0).all():
        raise SelectionError("train pool contains non-positive margins")
    return pool.sort_values("orig_index").reset_index(drop=True)


def check_thresholds(pool: pd.DataFrame, quantiles: list[float], expected: list[float]) -> list[float]:
    """Recompute the bucket thresholds from the pool and require agreement with prep metadata."""
    t = compute_thresholds(pool["margin"].to_numpy(), quantiles)
    if t != list(expected):
        raise SelectionError(f"thresholds recomputed from train pool {t} differ from prep metadata {expected}")
    if not (np.asarray(assign_bucket(pool["margin"], t)) == pool["bucket"].to_numpy()).all():
        raise SelectionError("bucket labels in train.csv disagree with the thresholds")
    return t


def source_quotas(pool: pd.DataFrame, n: int) -> dict[str, int]:
    """Largest-remainder allocation of n to sources in proportion to the pool (ties by name)."""
    share = pool["source"].value_counts().sort_index() / len(pool)
    raw = share * n
    base = np.floor(raw).astype(int)
    remainder = (raw - base).sort_values(ascending=False, kind="mergesort")
    for src in remainder.index[: n - int(base.sum())]:
        base[src] += 1
    return {k: int(v) for k, v in base.items()}


def condition_pool(pool: pd.DataFrame, bucket: str | None) -> pd.DataFrame:
    return pool if bucket is None else pool[pool["bucket"] == bucket]


def feasibility(pool: pd.DataFrame, conditions: dict[str, str | None], quotas: dict[str, int]) -> pd.DataFrame:
    rows = []
    for cond, bucket in conditions.items():
        avail = condition_pool(pool, bucket)["source"].value_counts()
        for src, q in quotas.items():
            a = int(avail.get(src, 0))
            rows.append({"condition": cond, "source": src, "required": q, "available": a,
                         "headroom": (a / q) if q else float("inf"), "shortfall": max(0, q - a)})
    return pd.DataFrame(rows)


def _sample(df: pd.DataFrame, k: int, seed: int, key: str) -> pd.DataFrame:
    df = df.sort_values("orig_index")
    idx = _stream(seed, key).choice(len(df), size=k, replace=False)
    return df.iloc[np.sort(idx)]


def select_condition(pool: pd.DataFrame, condition: str, bucket: str | None, n: int, seed: int,
                     quotas: dict[str, int] | None) -> pd.DataFrame:
    cpool = condition_pool(pool, bucket)
    if quotas is None:
        if len(cpool) < n:
            raise SelectionError(f"{condition}: needs {n} pairs but its pool has {len(cpool)}")
        out = _sample(cpool, n, seed, f"{condition}|unmatched")
    else:
        parts = []
        for src, q in sorted(quotas.items()):
            parts.append(_sample(cpool[cpool["source"] == src], q, seed, f"{condition}|{src}"))
        out = pd.concat(parts)
    out = out.sort_values("orig_index").assign(condition=condition)
    if len(out) != n:
        raise SelectionError(f"{condition}: selected {len(out)} pairs, expected {n}")
    if not out["prompt_sha256"].is_unique:
        raise SelectionError(f"{condition}: duplicate prompts in selection")
    return out


def select_all(pool: pd.DataFrame, conditions: dict[str, str | None], n: int, seed: int,
               match_source: bool = True) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    quotas = source_quotas(pool, n) if match_source else None
    feas = feasibility(pool, conditions, quotas or source_quotas(pool, n))
    if match_source and (feas["shortfall"] > 0).any():
        bad = feas[feas["shortfall"] > 0]
        raise SelectionError(
            f"Source-matched selection of {n} pairs per condition is infeasible; refusing to change the sample "
            f"size or quotas.\nCells with too few eligible train pairs:\n{bad.to_string(index=False)}\n"
            f"Options: lower the per-condition size, or disable match_source (records unmatched sources).")
    selected = {c: select_condition(pool, c, b, n, seed, quotas) for c, b in conditions.items()}
    return selected, feas


def unmatched_reference(pool: pd.DataFrame, conditions: dict[str, str | None], n: int, seed: int) -> dict:
    """Simple random samples of the same size from each condition's pool (diagnostics only, never trained on)."""
    out = {}
    for c, b in conditions.items():
        cpool = condition_pool(pool, b)
        out[c] = _sample(cpool, min(n, len(cpool)), seed, f"{c}|unmatched")
    return out


def write_manifest(df: pd.DataFrame, path: Path) -> str:
    from dpo_margin.data import write_jsonl

    return write_jsonl(df, path, MANIFEST_FIELDS)


def manifest_dir(root: str | Path, run_type: str) -> Path:
    return Path(root) / run_type


def eval_manifest_path(root: str | Path, size: int, seed: int) -> Path:
    return Path(root) / "eval" / f"validation_n{size}_s{seed}.jsonl"


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype={"row_id": str, "prompt_sha256": str, "pair_sha256": str})
