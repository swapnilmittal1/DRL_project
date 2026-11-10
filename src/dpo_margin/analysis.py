"""Descriptive statistics for the EDA summary. Everything here is computed from the feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from dpo_margin.prepare import BUCKETS, DROP_REASONS, SPLITS

LENGTH_COLS = ["prompt_tokens", "chosen_tokens", "rejected_tokens", "token_length_diff",
               "total_chosen_tokens", "total_rejected_tokens"]


def describe(x) -> dict:
    x = pd.Series(x, dtype="float64").dropna()
    if x.empty:
        return {"n": 0}
    q = x.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "p01": float(q[0.01]), "p25": float(q[0.25]), "median": float(q[0.5]),
            "p75": float(q[0.75]), "p99": float(q[0.99]), "max": float(x.max())}


def value_counts(x) -> dict:
    return {f"{k:g}" if isinstance(k, float) else str(k): int(v)
            for k, v in pd.Series(x).value_counts().sort_index().items()}


def spearman(a, b) -> dict:
    a, b = pd.Series(a, dtype="float64"), pd.Series(b, dtype="float64")
    ok = a.notna() & b.notna()
    res = stats.spearmanr(a[ok], b[ok])
    return {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": int(ok.sum())}


def cramers_v(a, b) -> dict:
    table = pd.crosstab(pd.Series(a), pd.Series(b))
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p),
            "cramers_v": float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0, "n": int(n)}


def threshold_sensitivity(train_margins, quantiles) -> dict:
    """Thresholds under different numpy quantile conventions, and the train ECDF at each margin value."""
    m = np.asarray(train_margins, dtype=float)
    methods = {meth: [float(v) for v in np.quantile(m, quantiles, method=meth)]
               for meth in ("linear", "lower", "higher", "nearest", "midpoint")}
    values = np.unique(m)
    ecdf = {f"{v:g}": float((m <= v).mean()) for v in values}
    return {"by_quantile_method": methods, "train_ecdf_at_value": ecdf}


def compute_summary(df: pd.DataFrame, meta: dict) -> dict:
    kept = df[df["drop_reason"].isna()].copy()
    train = kept[kept["split"] == "train"]
    n_raw = len(df)

    reasons = {r: int((df["drop_reason"] == r).sum()) for r in DROP_REASONS}
    counts = {
        "raw": n_raw, "kept": int(len(kept)), "dropped": int(n_raw - len(kept)),
        "dropped_by_reason": reasons,
        "filter_rate_by_reason": {r: c / n_raw for r, c in reasons.items()},
        "overall_filter_rate": (n_raw - len(kept)) / n_raw,
        "dropped_by_reason_and_source": {
            r: value_counts(df.loc[df["drop_reason"] == r, "source"]) for r, c in reasons.items() if c
        },
        "raw_by_source": value_counts(df["source"]),
        "kept_by_source": value_counts(kept["source"]),
    }

    prompts_by_split = {s: set(kept.loc[kept["split"] == s, "prompt_sha256"]) for s in SPLITS}
    splits = {
        "pairs": {s: int((kept["split"] == s).sum()) for s in SPLITS},
        "unique_prompts": {s: len(prompts_by_split[s]) for s in SPLITS},
        "fraction": {s: float((kept["split"] == s).mean()) for s in SPLITS},
        "prompt_overlap": {f"{a}&{b}": len(prompts_by_split[a] & prompts_by_split[b])
                           for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]},
        "by_source": {s: value_counts(kept.loc[kept["split"] == s, "source"]) for s in SPLITS},
        "manifest_sha256": meta["manifest_sha256"],
    }

    ratings = {
        "chosen": describe(kept["chosen_rating"]), "rejected": describe(kept["rejected_rating"]),
        "chosen_value_counts": value_counts(kept["chosen_rating"]),
        "rejected_value_counts": value_counts(kept["rejected_rating"]),
        "spearman_chosen_vs_rejected": spearman(kept["chosen_rating"], kept["rejected_rating"]),
        "pearson_chosen_vs_rejected": float(np.corrcoef(kept["chosen_rating"], kept["rejected_rating"])[0, 1]),
    }

    margin = {
        "all_kept": describe(kept["margin"]), "all_kept_value_counts": value_counts(kept["margin"]),
        "train": describe(train["margin"]),
        "by_split": {s: describe(kept.loc[kept["split"] == s, "margin"]) for s in SPLITS},
    }

    thresholds = {
        "quantiles": meta["bucket_quantiles"],
        "values_from_train": meta["bucket_thresholds_from_train"],
        "rule": meta["bucket_rule"],
        "bucket_counts": meta["bucket_counts"],
        "bucket_fraction_train": {b: float((train["bucket"] == b).mean()) for b in BUCKETS},
        "bucket_margin_values_train": {b: sorted(float(v) for v in train.loc[train["bucket"] == b, "margin"].unique())
                                       for b in BUCKETS},
        "sensitivity": threshold_sensitivity(train["margin"], meta["bucket_quantiles"]),
    }

    lengths = {c: describe(kept[c]) for c in LENGTH_COLS}
    lengths["fraction_chosen_longer"] = float((kept["token_length_diff"] > 0).mean())
    lengths["fraction_equal_length"] = float((kept["token_length_diff"] == 0).mean())
    short = kept[(kept["chosen_tokens"] <= 2) | (kept["rejected_tokens"] <= 2)]
    lengths["pairs_with_response_le_2_tokens"] = {"n": int(len(short)), "by_source": value_counts(short["source"])}

    by_source = {}
    for src, g in kept.groupby("source"):
        tg = g[g["split"] == "train"]
        by_source[src] = {
            "n_kept": int(len(g)), "share_of_kept": float(len(g) / len(kept)),
            "margin": describe(g["margin"]),
            "chosen_rating_mean": float(g["chosen_rating"].mean()),
            "rejected_rating_mean": float(g["rejected_rating"].mean()),
            "chosen_tokens_mean": float(g["chosen_tokens"].mean()),
            "rejected_tokens_mean": float(g["rejected_tokens"].mean()),
            "token_length_diff_mean": float(g["token_length_diff"].mean()),
            "train_bucket_share": {b: float((tg["bucket"] == b).mean()) for b in BUCKETS},
        }

    models = sorted(set(kept["chosen_model"]) | set(kept["rejected_model"]))
    cm, rm = kept["chosen_model"].value_counts(), kept["rejected_model"].value_counts()
    model_stats = {
        "chosen_counts": {k: int(v) for k, v in cm.items()},
        "rejected_counts": {k: int(v) for k, v in rm.items()},
        "chosen_share_when_present": {m: float(cm.get(m, 0) / (cm.get(m, 0) + rm.get(m, 0))) for m in models},
        "same_model_pairs": int((kept["chosen_model"] == kept["rejected_model"]).sum()),
        "n_models": len(models),
    }

    by_bucket = {}
    for b in BUCKETS:
        g = train[train["bucket"] == b]
        by_bucket[b] = {
            "n": int(len(g)), "margin_mean": float(g["margin"].mean()),
            "chosen_rating_mean": float(g["chosen_rating"].mean()),
            "rejected_rating_mean": float(g["rejected_rating"].mean()),
            "prompt_tokens_mean": float(g["prompt_tokens"].mean()),
            "chosen_tokens_mean": float(g["chosen_tokens"].mean()),
            "rejected_tokens_mean": float(g["rejected_tokens"].mean()),
            "token_length_diff_mean": float(g["token_length_diff"].mean()),
            "token_length_diff_median": float(g["token_length_diff"].median()),
            "fraction_chosen_longer": float((g["token_length_diff"] > 0).mean()),
            "source_share": {k: float(v) for k, v in g["source"].value_counts(normalize=True).sort_index().items()},
            "top_chosen_models": {k: int(v) for k, v in g["chosen_model"].value_counts().head(5).items()},
            "top_rejected_models": {k: int(v) for k, v in g["rejected_model"].value_counts().head(5).items()},
        }
    confounders = {
        "scope": "train split; associations are descriptive, not causal",
        "spearman_margin_vs": {c: spearman(train["margin"], train[c]) for c in
                               ["token_length_diff", "chosen_tokens", "rejected_tokens", "prompt_tokens",
                                "chosen_rating", "rejected_rating"]},
        "by_bucket": by_bucket,
        "bucket_vs_source": cramers_v(train["bucket"], train["source"]),
        "bucket_vs_chosen_model": cramers_v(train["bucket"], train["chosen_model"]),
        "bucket_vs_rejected_model": cramers_v(train["bucket"], train["rejected_model"]),
    }

    return {
        "provenance": {k: meta[k] for k in ("dataset_name", "dataset_revision", "tokenizer_name",
                                            "tokenizer_revision", "max_total_tokens", "split_seed")},
        "schema": meta["schema"],
        "structure_audit": meta["structure_audit"],
        "inspection_checks": meta["inspection_checks"],
        "counts": counts,
        "duplicates": meta["duplicates"],
        "splits": splits,
        "ratings": ratings,
        "margin": margin,
        "thresholds": thresholds,
        "token_lengths": lengths,
        "by_source": by_source,
        "models": model_stats,
        "confounders": confounders,
    }
