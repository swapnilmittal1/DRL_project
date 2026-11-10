"""Cleaning, feature engineering, leakage-safe splitting and margin thresholds.

Pipeline (each dropped row keeps the *first* reason it fails, in DROP_REASONS order):
  structure / ratings / empty text / identical responses / margin -> token lengths -> overlong
  -> duplicate normalized prompts (lowest original index kept)
Kept pairs are split by a salted SHA-256 of the normalized prompt, so the split depends only on the
prompt text and the seed (not on row order), and a prompt can never land in two partitions.
Margin bucket thresholds are quantiles of the *training* partition only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPECTED_FEATURES = {
    "source": "string", "prompt": "string", "chosen": "messages", "chosen-rating": "float64",
    "chosen-model": "string", "rejected": "messages", "rejected-rating": "float64", "rejected-model": "string",
}
DROP_REASONS = (
    "malformed_structure",
    "malformed_rating",
    "empty_prompt",
    "empty_chosen",
    "empty_rejected",
    "identical_responses",
    "non_positive_margin",
    "overlong",
    "duplicate_normalized_prompt",
)
SPLITS = ("train", "validation", "test")
BUCKETS = ("low", "medium", "high")
MANIFEST_COLUMNS = [
    "row_id", "orig_index", "prompt_sha256", "pair_sha256", "source", "margin", "bucket",
    "chosen_rating", "rejected_rating", "chosen_model", "rejected_model",
    "prompt_tokens", "chosen_tokens", "rejected_tokens", "token_length_diff",
]


def load_prep_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    fr = cfg["split"]["fractions"]
    if list(fr) != list(SPLITS) or not math.isclose(sum(fr.values()), 1.0):
        raise ValueError(f"split.fractions must be {SPLITS} summing to 1, got {fr}")
    q = cfg["buckets"]["quantiles"]
    if len(q) != 2 or not 0 < q[0] < q[1] < 1:
        raise ValueError("buckets.quantiles must be two increasing values in (0, 1)")
    return cfg


def normalize_prompt(text: str) -> str:
    """NFKC, lowercase, collapse all whitespace runs, strip."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_sha256(prompt: str, chosen: str, rejected: str) -> str:
    return sha256_text(json.dumps([prompt, chosen, rejected], ensure_ascii=False))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_schema(features) -> list[str]:
    """Compare a datasets.Features object with EXPECTED_FEATURES; return a list of problems."""
    problems = []
    for name, kind in EXPECTED_FEATURES.items():
        if name not in features:
            problems.append(f"missing column {name}")
            continue
        feat = features[name]
        if kind == "messages":
            inner = getattr(feat, "feature", None)
            if not isinstance(inner, dict) or set(inner) != {"role", "content"}:
                problems.append(f"{name}: expected list of {{role, content}}, got {feat}")
        elif getattr(feat, "dtype", None) != kind:
            problems.append(f"{name}: expected {kind}, got {feat}")
    extra = set(features) - set(EXPECTED_FEATURES)
    if extra:
        problems.append(f"unexpected columns {sorted(extra)}")
    return problems


def _valid_rating(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 1.0 <= v <= 5.0


def _structure_ok(row: dict) -> bool:
    ch, rj, prompt = row.get("chosen"), row.get("rejected"), row.get("prompt")
    if not isinstance(prompt, str) or not isinstance(ch, list) or not isinstance(rj, list):
        return False
    if len(ch) != 2 or len(rj) != 2:
        return False
    if [m.get("role") for m in ch] != ["user", "assistant"] or [m.get("role") for m in rj] != ["user", "assistant"]:
        return False
    if not all(isinstance(m.get("content"), str) for m in ch + rj):
        return False
    return ch[0]["content"] == prompt and rj[0]["content"] == prompt


def first_failure(row: dict) -> str | None:
    """Reason a raw row is unusable before tokenization, or None."""
    if not _structure_ok(row):
        return "malformed_structure"
    if not _valid_rating(row.get("chosen-rating")) or not _valid_rating(row.get("rejected-rating")):
        return "malformed_rating"
    if not row["prompt"].strip():
        return "empty_prompt"
    chosen, rejected = row["chosen"][1]["content"], row["rejected"][1]["content"]
    if not chosen.strip():
        return "empty_chosen"
    if not rejected.strip():
        return "empty_rejected"
    if chosen.strip() == rejected.strip():
        return "identical_responses"
    if not row["chosen-rating"] > row["rejected-rating"]:
        return "non_positive_margin"
    return None


def extract_rows(dataset) -> pd.DataFrame:
    """One record per raw example: original fields preserved, texts flattened, first failure recorded."""
    records = []
    for i, row in enumerate(dataset):
        reason = first_failure(row)
        ok_struct = reason != "malformed_structure"
        prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else ""
        chosen = row["chosen"][1]["content"] if ok_struct else ""
        rejected = row["rejected"][1]["content"] if ok_struct else ""
        p_hash = sha256_text(normalize_prompt(prompt))
        pair_hash = pair_sha256(prompt, chosen, rejected)
        records.append({
            "row_id": f"{i:05d}-{pair_hash[:10]}",
            "orig_index": i,
            "source": row.get("source"),
            "chosen_model": row.get("chosen-model"),
            "rejected_model": row.get("rejected-model"),
            "chosen_rating": row.get("chosen-rating"),
            "rejected_rating": row.get("rejected-rating"),
            "prompt": prompt,
            "chosen_text": chosen,
            "rejected_text": rejected,
            "prompt_sha256": p_hash,
            "pair_sha256": pair_hash,
            "drop_reason": reason,
        })
    df = pd.DataFrame.from_records(records)
    ratings_ok = df["drop_reason"].ne("malformed_rating") & df["drop_reason"].ne("malformed_structure")
    df["margin"] = np.where(
        ratings_ok,
        pd.to_numeric(df["chosen_rating"], errors="coerce") - pd.to_numeric(df["rejected_rating"], errors="coerce"),
        np.nan,
    )
    df["prompt_chars"] = df["prompt"].str.len()
    df["chosen_chars"] = df["chosen_text"].str.len()
    df["rejected_chars"] = df["rejected_text"].str.len()
    return df


def _lengths(tokenizer, texts: list[str], batch_size: int = 2048) -> np.ndarray:
    out = []
    for start in range(0, len(texts), batch_size):
        ids = tokenizer(texts[start:start + batch_size], add_special_tokens=False)["input_ids"]
        out.extend(len(x) for x in ids)
    return np.asarray(out, dtype=np.int64)


def add_token_features(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Token lengths for rows without a drop reason, using the policy's chat template.

    prompt_tokens: templated prompt incl. generation prompt (what TRL feeds as the prompt).
    chosen/rejected_tokens: assistant response text alone.
    total_*_tokens: full templated conversation, i.e. the sequence length TRL trains on.
    """
    df = df.copy()
    cols = ["prompt_tokens", "chosen_tokens", "rejected_tokens", "total_chosen_tokens", "total_rejected_tokens"]
    for c in cols:
        df[c] = pd.array([pd.NA] * len(df), dtype="Int64")
    live = df["drop_reason"].isna()
    if not live.any():
        df["token_length_diff"] = pd.array([pd.NA] * len(df), dtype="Int64")
        return df
    sub = df.loc[live]

    def tmpl(prompt, response=None):
        msgs = [{"role": "user", "content": prompt}]
        if response is None:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": response}], tokenize=False)

    df.loc[live, "prompt_tokens"] = _lengths(tokenizer, [tmpl(p) for p in sub["prompt"]])
    df.loc[live, "chosen_tokens"] = _lengths(tokenizer, sub["chosen_text"].tolist())
    df.loc[live, "rejected_tokens"] = _lengths(tokenizer, sub["rejected_text"].tolist())
    df.loc[live, "total_chosen_tokens"] = _lengths(
        tokenizer, [tmpl(p, c) for p, c in zip(sub["prompt"], sub["chosen_text"])])
    df.loc[live, "total_rejected_tokens"] = _lengths(
        tokenizer, [tmpl(p, r) for p, r in zip(sub["prompt"], sub["rejected_text"])])
    df["token_length_diff"] = df["chosen_tokens"] - df["rejected_tokens"]
    return df


def mark_overlong(df: pd.DataFrame, max_total_tokens: int) -> pd.DataFrame:
    df = df.copy()
    live = df["drop_reason"].isna()
    longest = np.maximum(df["total_chosen_tokens"].astype("float"), df["total_rejected_tokens"].astype("float"))
    df.loc[live & (longest > max_total_tokens), "drop_reason"] = "overlong"
    return df


def mark_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Among surviving rows keep the lowest orig_index per normalized prompt."""
    df = df.copy()
    live = df["drop_reason"].isna()
    counts = df.loc[live, "prompt_sha256"].value_counts()
    dup_hashes = counts[counts > 1]
    in_groups = df.loc[live & df["prompt_sha256"].isin(dup_hashes.index)]
    live_sorted = df.loc[live].sort_values("orig_index")
    dup_idx = live_sorted.index[live_sorted.duplicated("prompt_sha256", keep="first")]
    df.loc[dup_idx, "drop_reason"] = "duplicate_normalized_prompt"
    stats = {
        "duplicate_groups": int(len(dup_hashes)),
        "rows_in_duplicate_groups": int(dup_hashes.sum()),
        "rows_dropped": int(len(dup_idx)),
        "groups_differing_only_after_normalization": int(
            in_groups.groupby("prompt_sha256")["prompt"].nunique().gt(1).sum()
        ),
    }
    return df, stats


def split_of(prompt_sha: str, seed: int, fractions: dict[str, float]) -> str:
    """Deterministic split from a salted hash of the normalized prompt."""
    u = int(sha256_text(f"{seed}:{prompt_sha}")[:16], 16) / 2**64
    cum = 0.0
    for name in SPLITS:
        cum += fractions[name]
        if u < cum:
            return name
    return SPLITS[-1]


def compute_thresholds(train_margins: np.ndarray, quantiles: list[float]) -> list[float]:
    """Quantile thresholds of the training margins (numpy 'linear' method)."""
    return [float(x) for x in np.quantile(np.asarray(train_margins, dtype=float), quantiles)]


def assign_bucket(margins, thresholds: list[float]) -> np.ndarray:
    """low: m <= t1; medium: t1 < m <= t2; high: m > t2."""
    m = np.asarray(margins, dtype=float)
    t1, t2 = thresholds
    return np.where(m <= t1, "low", np.where(m <= t2, "medium", "high"))


def prepare(dataset, tokenizer, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Run the full pipeline. Returns the per-row table (all rows, with drop_reason) and metadata."""
    df = extract_rows(dataset)
    df = add_token_features(df, tokenizer)
    df = mark_overlong(df, cfg["max_total_tokens"])
    df, dup_stats = mark_duplicates(df)

    kept = df["drop_reason"].isna()
    df["split"] = None
    df.loc[kept, "split"] = [split_of(h, cfg["split"]["seed"], cfg["split"]["fractions"])
                             for h in df.loc[kept, "prompt_sha256"]]
    train = kept & df["split"].eq("train")
    thresholds = compute_thresholds(df.loc[train, "margin"].to_numpy(), cfg["buckets"]["quantiles"])
    df["bucket"] = None
    df.loc[kept, "bucket"] = assign_bucket(df.loc[kept, "margin"], thresholds)

    reasons = df["drop_reason"].value_counts()
    meta = {
        "n_raw": int(len(df)),
        "n_kept": int(kept.sum()),
        "n_dropped": int((~kept).sum()),
        "dropped_by_reason": {r: int(reasons.get(r, 0)) for r in DROP_REASONS},
        "duplicates": dup_stats,
        "max_total_tokens": cfg["max_total_tokens"],
        "split_seed": cfg["split"]["seed"],
        "split_fractions": cfg["split"]["fractions"],
        "split_pairs": {s: int((kept & df["split"].eq(s)).sum()) for s in SPLITS},
        "bucket_quantiles": cfg["buckets"]["quantiles"],
        "bucket_thresholds_from_train": thresholds,
        "bucket_rule": "low: margin <= t1; medium: t1 < margin <= t2; high: margin > t2",
        "bucket_counts": {
            s: {b: int((kept & df["split"].eq(s) & df["bucket"].eq(b)).sum()) for b in BUCKETS} for s in SPLITS
        },
    }
    return df, meta


def write_outputs(df: pd.DataFrame, meta: dict, splits_dir: Path, features_path: Path) -> dict:
    """Write split manifests, dropped list, checksums, metadata and the feature table."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    kept = df["drop_reason"].isna()
    files = {}
    for s in SPLITS:
        part = df.loc[kept & df["split"].eq(s), MANIFEST_COLUMNS].sort_values("orig_index")
        path = splits_dir / f"{s}.csv"
        part.to_csv(path, index=False, lineterminator="\n")
        files[path.name] = sha256_file(path)
    dropped = df.loc[~kept, ["row_id", "orig_index", "source", "drop_reason"]].sort_values("orig_index")
    path = splits_dir / "dropped.csv"
    dropped.to_csv(path, index=False, lineterminator="\n")
    files[path.name] = sha256_file(path)

    (splits_dir / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(files.items())))
    meta = {**meta, "manifest_sha256": files}
    (splits_dir / "prep_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    text_cols = ["prompt", "chosen_text", "rejected_text"]
    df.drop(columns=text_cols).to_parquet(features_path, index=False)
    return meta


def verify_checksums(splits_dir: Path) -> list[str]:
    """Return names of manifests whose SHA-256 does not match SHA256SUMS."""
    bad = []
    for line in (splits_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256_file(splits_dir / name) != digest:
            bad.append(name)
    return bad
