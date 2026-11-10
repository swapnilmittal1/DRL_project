#!/usr/bin/env python
"""Exploratory data analysis of the prepared preference data.

Usage: python scripts/eda.py [--config configs/data/prep.yaml]
Reads the outputs of scripts/prepare_data.py; writes figures/eda/*.{png,pdf},
results/eda/analysis_summary.json and reports/eda_report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from dpo_margin.analysis import compute_summary  # noqa: E402
from dpo_margin.eda_report import render_report  # noqa: E402
from dpo_margin.prepare import BUCKETS, SPLITS, load_prep_config  # noqa: E402

FIG_DIR = ROOT / "figures" / "eda"
BUCKET_COLORS = dict(zip(BUCKETS, sns.color_palette("colorblind", 3)))
SPLIT_COLORS = dict(zip(SPLITS, ["#4c72b0", "#dd8452", "#55a868"]))


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def fig_margin_hist(kept: pd.DataFrame) -> None:
    counts = kept["margin"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(counts.index, counts.values, width=0.2, color=sns.color_palette("colorblind")[0], edgecolor="black", lw=0.4)
    ax.set_xlabel("Preference margin (chosen rating − rejected rating)")
    ax.set_ylabel("Number of pairs")
    ax.set_xticks(counts.index[::2])
    m = kept["margin"]
    ax.set_title(f"Preference margins of cleaned pairs (n = {len(m):,}; mean {m.mean():.2f}, median {m.median():.2f})")
    save(fig, "fig1_margin_histogram")


def fig_chosen_vs_rejected(kept: pd.DataFrame) -> None:
    grid = kept.groupby(["rejected_rating", "chosen_rating"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(grid["rejected_rating"], grid["chosen_rating"], s=12 + 280 * np.sqrt(grid["n"] / grid["n"].max()),
                    c=np.log10(grid["n"]), cmap="viridis", edgecolor="black", lw=0.3)
    ax.plot([1, 5], [1, 5], ls="--", color="grey", lw=1, label="chosen = rejected")
    ax.set_xlim(0.8, 5.2), ax.set_ylim(0.8, 5.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Rejected rating")
    ax.set_ylabel("Chosen rating")
    ax.set_title("Chosen vs. rejected rating\n(marker area and colour = number of pairs)")
    ax.legend(loc="lower right", frameon=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("log10(number of pairs)")
    save(fig, "fig2_chosen_vs_rejected")


def fig_margin_by_source(kept: pd.DataFrame) -> None:
    order = kept.groupby("source")["margin"].median().sort_values().index.tolist()
    n = kept["source"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.boxplot(data=kept, y="source", x="margin", order=order, ax=ax, color="#cfd8e3", fliersize=1.5, linewidth=0.8,
                showmeans=True, meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black",
                                           "markersize": 3.5})
    ax.set_yticks(range(len(order)), [f"{s} (n={n[s]:,})" for s in order])
    ax.set_xlabel("Preference margin")
    ax.set_ylabel("")
    ax.set_title("Margin distribution by source (black diamond = mean)")
    save(fig, "fig3_margin_by_source")


def fig_length_vs_margin(kept: pd.DataFrame, rho: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    order = sorted(kept["margin"].unique())
    data = kept.assign(margin_label=kept["margin"].map(lambda v: f"{v:g}"), diff=kept["token_length_diff"].astype(float))
    sns.boxplot(data=data, x="margin_label", y="diff", order=[f"{v:g}" for v in order], ax=ax, color="#cfd8e3",
                showfliers=False, linewidth=0.8)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Preference margin")
    ax.set_ylabel("Chosen − rejected response tokens")
    ax.set_title(f"Response-length difference vs. margin (outliers not drawn)\n"
                 f"Spearman ρ = {rho['rho']:.3f} (all cleaned pairs, n = {rho['n']:,})")
    save(fig, "fig4_length_diff_vs_margin")


def fig_examples_per_source(kept: pd.DataFrame, raw: pd.DataFrame) -> None:
    tab = pd.crosstab(kept["source"], kept["split"]).reindex(columns=list(SPLITS), fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    left = np.zeros(len(tab))
    for s in SPLITS:
        ax.barh(tab.index, tab[s], left=left, color=SPLIT_COLORS[s], label=s, edgecolor="black", lw=0.3)
        left += tab[s].to_numpy()
    raw_n = raw["source"].value_counts()
    for i, src in enumerate(tab.index):
        ax.text(left[i] * 1.01 + 150, i, f"{int(left[i]):,} kept / {raw_n[src]:,} raw", va="center", fontsize=7)
    ax.set_xlim(0, left.max() * 1.35)
    ax.set_xlabel("Number of pairs")
    ax.set_title("Available pairs per source after cleaning, by split")
    ax.legend(title="split", loc="lower right")
    save(fig, "fig5_examples_per_source")


def fig_thresholds(train: pd.DataFrame, thresholds: list[float]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    counts = train.groupby(["margin", "bucket"]).size().reset_index(name="n")
    for b in BUCKETS:
        c = counts[counts["bucket"] == b]
        n_b = int((train["bucket"] == b).sum())
        a1.bar(c["margin"], c["n"], width=0.2, color=BUCKET_COLORS[b], edgecolor="black", lw=0.4,
               label=f"{b} (n={n_b:,}, {n_b / len(train):.1%})")
    for t in thresholds:
        a1.axvline(t + 0.125, color="black", ls="--", lw=1)
    a1.set_xlabel("Preference margin")
    a1.set_ylabel("Train pairs")
    a1.set_title(f"Train-split margin buckets (thresholds t1={thresholds[0]:g}, t2={thresholds[1]:g})")
    a1.legend(title="bucket")

    m = np.sort(train["margin"].to_numpy())
    a2.step(m, np.arange(1, len(m) + 1) / len(m), where="post", color="black", lw=1.2)
    for q in (1 / 3, 2 / 3):
        a2.axhline(q, color="grey", ls=":", lw=1)
    for t in thresholds:
        a2.axvline(t, color="black", ls="--", lw=1)
    a2.set_xlabel("Preference margin")
    a2.set_ylabel("Empirical CDF (train)")
    a2.set_title("ECDF with 1/3 and 2/3 levels (dotted)")
    save(fig, "fig6_margin_thresholds")


def fig_models(kept: pd.DataFrame) -> None:
    cm, rm = kept["chosen_model"].value_counts(), kept["rejected_model"].value_counts()
    models = (cm.add(rm, fill_value=0)).sort_values().index
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    a1.barh(models, cm.reindex(models, fill_value=0), color=sns.color_palette("colorblind")[2], edgecolor="black", lw=0.3)
    a2.barh(models, rm.reindex(models, fill_value=0), color=sns.color_palette("colorblind")[3], edgecolor="black", lw=0.3)
    a1.set_title("Chosen-response model")
    a2.set_title("Rejected-response model")
    for a in (a1, a2):
        a.set_xlabel("Number of cleaned pairs")
    fig.suptitle("Model identity frequencies (cleaned pairs)", y=1.0)
    save(fig, "fig7_model_frequencies")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/prep.yaml")
    args = parser.parse_args()
    cfg = load_prep_config(args.config)

    df = pd.read_parquet(ROOT / cfg["outputs"]["features_path"])
    meta = json.loads((ROOT / cfg["outputs"]["splits_dir"] / "prep_metadata.json").read_text())
    summary = compute_summary(df, meta)

    out = ROOT / "results" / "eda" / "analysis_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({"pdf.fonttype": 42, "svg.hashsalt": "eda", "axes.titlesize": 9.5})
    kept = df[df["drop_reason"].isna()]
    train = kept[kept["split"] == "train"]
    fig_margin_hist(kept)
    fig_chosen_vs_rejected(kept)
    fig_margin_by_source(kept)
    from dpo_margin.analysis import spearman
    fig_length_vs_margin(kept, spearman(kept["margin"], kept["token_length_diff"]))
    fig_examples_per_source(kept, df)
    fig_thresholds(train, summary["thresholds"]["values_from_train"])
    fig_models(kept)
    print(f"Wrote 7 figures (PNG + PDF) to {FIG_DIR.relative_to(ROOT)}")

    report = ROOT / "reports" / "eda_report.md"
    report.write_text(render_report(summary))
    print(f"Wrote {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
