"""Balance diagnostics for selected conditions (descriptive; computed from the manifests)."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def tvd(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance between two categorical samples."""
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    keys = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(keys, fill_value=0) - pb.reindex(keys, fill_value=0)).abs().sum())


def ks(a, b) -> tuple[float, float]:
    r = stats.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def margin_rule(bucket: str | None, t: list[float]) -> str:
    return {None: "margin > 0", "low": f"0 < margin <= {t[0]:g}", "medium": f"{t[0]:g} < margin <= {t[1]:g}",
            "high": f"margin > {t[1]:g}"}[bucket]


def summarize(selected: dict, unmatched: dict, pool: pd.DataFrame, conditions: dict, feas: pd.DataFrame,
              thresholds: list[float], n_target: int, reference: str = "E1_random") -> pd.DataFrame:
    ref = selected[reference]
    rows = []
    for cond, df in selected.items():
        bucket = conditions[cond]
        bpool = pool if bucket is None else pool[pool["bucket"] == bucket]
        um = unmatched[cond]
        f = feas[feas["condition"] == cond]
        ks_pool = ks(df["margin"], bpool["margin"])
        ks_um = ks(df["margin"], um["margin"])
        ks_c = ks(df["chosen_tokens"], ref["chosen_tokens"])
        ks_r = ks(df["rejected_tokens"], ref["rejected_tokens"])
        rows.append({
            "condition": cond,
            "margin_rule": margin_rule(bucket, thresholds),
            "n_target": n_target,
            "n_selected": len(df),
            "n_unique_prompts": int(df["prompt_sha256"].nunique()),
            "eligible_pool_size": len(bpool),
            "min_source_headroom": float(f["headroom"].min()),
            "total_shortfall": int(f["shortfall"].sum()),
            "margin_mean": df["margin"].mean(),
            "margin_std": df["margin"].std(),
            "margin_min": df["margin"].min(),
            "margin_median": df["margin"].median(),
            "margin_max": df["margin"].max(),
            "margin_mean_unmatched_sample": um["margin"].mean(),
            "margin_mean_eligible_pool": bpool["margin"].mean(),
            "margin_ks_vs_eligible_pool": ks_pool[0],
            "margin_ks_vs_eligible_pool_p": ks_pool[1],
            "margin_ks_vs_unmatched_sample": ks_um[0],
            "source_tvd_vs_train_pool": tvd(df["source"], pool["source"]),
            "source_tvd_unmatched_vs_train_pool": tvd(um["source"], pool["source"]),
            "chosen_rating_mean": df["chosen_rating"].mean(),
            "rejected_rating_mean": df["rejected_rating"].mean(),
            "prompt_tokens_mean": df["prompt_tokens"].mean(),
            "chosen_tokens_mean": df["chosen_tokens"].mean(),
            "chosen_tokens_median": df["chosen_tokens"].median(),
            "rejected_tokens_mean": df["rejected_tokens"].mean(),
            "rejected_tokens_median": df["rejected_tokens"].median(),
            "length_diff_mean": df["token_length_diff"].mean(),
            "length_diff_median": df["token_length_diff"].median(),
            "frac_chosen_longer": (df["token_length_diff"] > 0).mean(),
            f"chosen_tokens_ks_vs_{reference}": ks_c[0],
            f"rejected_tokens_ks_vs_{reference}": ks_r[0],
            f"chosen_model_tvd_vs_{reference}": tvd(df["chosen_model"], ref["chosen_model"]),
            f"rejected_model_tvd_vs_{reference}": tvd(df["rejected_model"], ref["rejected_model"]),
            f"overlap_with_{reference}": int(df["row_id"].isin(ref["row_id"]).sum()) if cond != reference else len(df),
        })
    return pd.DataFrame(rows)


def overlap_matrix(selected: dict) -> pd.DataFrame:
    names = list(selected)
    return pd.DataFrame([[int(selected[a]["row_id"].isin(selected[b]["row_id"]).sum()) for b in names] for a in names],
                        index=names, columns=names)
