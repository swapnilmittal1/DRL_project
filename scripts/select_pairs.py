#!/usr/bin/env python
"""Select source-matched training pairs for E1-E4 from the train split and report balance.

Usage: python scripts/select_pairs.py --mode smoke|pilot|final
Writes:
  data/manifests/<mode>/<condition>.jsonl, selection_metadata.json
  data/manifests/eval/validation_n<size>_s<seed>.jsonl      (shared eval subset, validation split)
  results/selection/<mode>/selected_pairs_summary.csv, source_counts.csv, overlap_matrix.csv
  figures/selection/<mode>/{dataset_balance,margin_comparison,source_distribution,response_length_comparison}.{png,pdf}
  reports/selection/<mode>/sample_examples.md, selection_report.md
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
import yaml  # noqa: E402

from dpo_margin import data, selection  # noqa: E402
from dpo_margin.prepare import load_prep_config  # noqa: E402
from dpo_margin.selection_diagnostics import overlap_matrix, summarize  # noqa: E402

COLORS = dict(zip(["E1_random", "E2_high_margin", "E3_medium_margin", "E4_low_margin"],
                  ["#7f7f7f", "#0173b2", "#de8f05", "#029e73"]))


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def fig_balance(selected, feas, n, out):
    conds = list(selected)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.45})
    counts = [len(selected[c]) for c in conds]
    a1.bar(conds, counts, color=[COLORS[c] for c in conds], edgecolor="black", lw=0.4)
    a1.axhline(n, color="black", ls="--", lw=1, label=f"target = {n:,}")
    for i, c in enumerate(conds):
        a1.text(i, counts[i] * 0.5, f"{counts[i]:,} pairs\n{selected[c]['prompt_sha256'].nunique():,} prompts",
                ha="center", va="center", fontsize=7, color="white")
    a1.set_ylabel("Selected pairs")
    a1.set_title("Selected pairs per condition")
    a1.tick_params(axis="x", rotation=20)
    a1.legend(loc="lower right")
    head = feas.pivot(index="source", columns="condition", values="headroom")[conds]
    avail = feas.pivot(index="source", columns="condition", values="available")[conds]
    req = feas.pivot(index="source", columns="condition", values="required")[conds]
    annot = avail.astype(int).astype(str) + "/" + req.astype(int).astype(str)
    sns.heatmap(np.log10(head), annot=annot, fmt="", cmap="RdYlGn", vmin=-1.5, vmax=1.5, ax=a2,
                cbar_kws={"label": "log10(available / required); < 0 = shortfall"},
                annot_kws={"fontsize": 7}, linewidths=0.5)
    a2.set_title("Eligible train pairs / source quota (red = shortfall)")
    a2.set_xlabel("")
    a2.set_ylabel("")
    a2.tick_params(axis="x", rotation=20)
    save(fig, out, "dataset_balance")


def fig_margin(selected, pool, conditions, out):
    conds = list(selected)
    values = np.sort(pool["margin"].unique())
    fig, axes = plt.subplots(len(conds), 1, figsize=(7, 1.9 * len(conds)), sharex=True)
    for ax, c in zip(axes, conds):
        b = conditions[c]
        bpool = pool if b is None else pool[pool["bucket"] == b]
        sel = selected[c]["margin"].value_counts(normalize=True).reindex(values, fill_value=0)
        ref = bpool["margin"].value_counts(normalize=True).reindex(values, fill_value=0)
        ax.bar(values, sel.values, width=0.2, color=COLORS[c], edgecolor="black", lw=0.3, label="selected (source-matched)")
        ax.plot(values, ref.values, "k_", ms=12, mew=1.5, label="eligible pool (unmatched)")
        ax.set_ylabel("fraction")
        ax.set_title(f"{c}: mean margin {selected[c]['margin'].mean():.3f} (pool {bpool['margin'].mean():.3f})",
                     fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Preference margin (chosen − rejected rating)")
    axes[-1].set_xticks(values[::2])
    save(fig, out, "margin_comparison")


def fig_sources(selected, pool, conditions, out):
    sources = sorted(pool["source"].unique())
    before = {"train pool": pool}
    for c, b in conditions.items():
        if b is not None:
            before[f"{b} bucket"] = pool[pool["bucket"] == b]
    pal = dict(zip(sources, sns.color_palette("tab10", len(sources))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, groups, title in [(axes[0], before, "Before matching: eligible pools"),
                              (axes[1], selected, "After matching: selected conditions")]:
        names = list(groups)
        bottom = np.zeros(len(names))
        for s in sources:
            vals = np.array([(groups[g]["source"] == s).mean() for g in names])
            ax.bar(names, vals, bottom=bottom, color=pal[s], edgecolor="white", lw=0.3, label=s)
            bottom += vals
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Share of pairs")
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title="source")
    save(fig, out, "source_distribution")


def fig_lengths(selected, out):
    conds = list(selected)
    long = pd.concat(selected.values(), ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, label in zip(axes, ["chosen_tokens", "rejected_tokens", "token_length_diff"],
                              ["Chosen response tokens", "Rejected response tokens", "Chosen − rejected tokens"]):
        sns.boxplot(data=long, x="condition", y=col, order=conds, hue="condition", palette=COLORS, ax=ax,
                    showfliers=False, linewidth=0.8, legend=False,
                    showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black",
                                               "markersize": 3.5})
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("tokens")
        ax.tick_params(axis="x", rotation=25)
        if col == "token_length_diff":
            ax.axhline(0, color="grey", ls="--", lw=0.8)
    fig.suptitle("Response lengths by condition (Qwen tokens; boxes IQR, whiskers 1.5 IQR, outliers hidden, white diamond = mean)",
                 fontsize=9, y=1.02)
    save(fig, out, "response_length_comparison")


def _clip(t: str, n: int = 300) -> str:
    t = t.replace("\n", " ⏎ ")
    return t if len(t) <= n else t[:n] + f" … [+{len(t) - n} chars]"


def sample_examples(selected, raw, seed, k=2) -> str:
    lines = ["# Sample selected examples", "",
             f"{k} pairs per condition, chosen with seed {seed} from each manifest. Shown in the exact TRL "
             "conversational preference format passed to `DPOTrainer` (texts truncated for display).", ""]
    for c, df in selected.items():
        rng = np.random.default_rng([seed, 99])
        pick = df.iloc[np.sort(rng.choice(len(df), size=min(k, len(df)), replace=False))]
        lines += [f"## {c}", ""]
        trl_rows = data.materialize(raw, pick)
        for (_, meta), ex in zip(pick.iterrows(), trl_rows):
            lines += [f"### `{meta['row_id']}` — source `{meta['source']}`, margin {meta['margin']:g}", "",
                      f"- chosen `{meta['chosen_model']}` ({meta['chosen_rating']:g}) vs rejected "
                      f"`{meta['rejected_model']}` ({meta['rejected_rating']:g}); tokens chosen "
                      f"{meta['chosen_tokens']}, rejected {meta['rejected_tokens']}", "", "```json",
                      json.dumps({"prompt": [{**m, "content": _clip(m["content"])} for m in ex["prompt"]],
                                  "chosen": [{**m, "content": _clip(m["content"])} for m in ex["chosen"]],
                                  "rejected": [{**m, "content": _clip(m["content"])} for m in ex["rejected"]]},
                                 indent=1, ensure_ascii=False), "```", ""]
    return "\n".join(lines) + "\n"


def selection_report(mode, n, summary: pd.DataFrame, quotas, overlaps: pd.DataFrame, thresholds) -> str:
    s = summary.set_index("condition")
    L = [f"# Selection report — `{mode}` mode ({n:,} pairs per condition)", "",
         "Generated by `scripts/select_pairs.py`; all values computed from the manifests. Descriptive only.", "",
         f"Train-only thresholds: t1 = {thresholds[0]:g}, t2 = {thresholds[1]:g}.", "",
         "## Source quotas (identical for every condition)", "", "| source | pairs |", "|---|---:|"]
    L += [f"| {k} | {v:,} |" for k, v in quotas.items()]
    L += ["", "## Matching outcome", "",
          "| condition | rule | selected | unique prompts | min headroom | shortfall | source TVD vs pool (matched / unmatched) "
          "| mean margin (selected / unmatched / pool) | margin KS vs pool |", "|---|---|---:|---:|---:|---:|---|---|---:|"]
    for c, r in s.iterrows():
        L.append(f"| {c} | {r['margin_rule']} | {r['n_selected']:,} | {r['n_unique_prompts']:,} | "
                 f"{r['min_source_headroom']:.2f} | {r['total_shortfall']} | {r['source_tvd_vs_train_pool']:.3f} / "
                 f"{r['source_tvd_unmatched_vs_train_pool']:.3f} | {r['margin_mean']:.3f} / "
                 f"{r['margin_mean_unmatched_sample']:.3f} / {r['margin_mean_eligible_pool']:.3f} | "
                 f"{r['margin_ks_vs_eligible_pool']:.3f} |")
    L += ["", "## Residual differences relative to E1_random", "",
          "| condition | chosen rating | rejected rating | prompt tok | chosen tok (mean/median) | rejected tok (mean/median) "
          "| diff (mean/median) | chosen longer | KS chosen tok | KS rejected tok | TVD chosen model | TVD rejected model |",
          "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for c, r in s.iterrows():
        L.append(f"| {c} | {r['chosen_rating_mean']:.2f} | {r['rejected_rating_mean']:.2f} | {r['prompt_tokens_mean']:.0f} | "
                 f"{r['chosen_tokens_mean']:.0f} / {r['chosen_tokens_median']:.0f} | "
                 f"{r['rejected_tokens_mean']:.0f} / {r['rejected_tokens_median']:.0f} | "
                 f"{r['length_diff_mean']:.0f} / {r['length_diff_median']:.0f} | {100 * r['frac_chosen_longer']:.1f}% | "
                 f"{r['chosen_tokens_ks_vs_E1_random']:.3f} | {r['rejected_tokens_ks_vs_E1_random']:.3f} | "
                 f"{r['chosen_model_tvd_vs_E1_random']:.3f} | {r['rejected_model_tvd_vs_E1_random']:.3f} |")
    L += ["", "## Row overlap between conditions", "", "| | " + " | ".join(overlaps.columns) + " |",
          "|---|" + "---:|" * len(overlaps.columns)]
    L += [f"| {i} | " + " | ".join(f"{v:,}" for v in row) + " |" for i, row in overlaps.iterrows()]
    L += ["", "E2–E4 are disjoint by construction (non-overlapping margin ranges); E1 overlaps each bucket by chance.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["smoke", "pilot", "final"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/selection.yaml")
    parser.add_argument("--prep-config", type=Path, default=ROOT / "configs/data/prep.yaml")
    parser.add_argument("--no-examples", action="store_true", help="skip sample_examples.md (no dataset load)")
    args = parser.parse_args()

    scfg = yaml.safe_load(args.config.read_text())
    pcfg = load_prep_config(args.prep_config)
    splits_dir = ROOT / pcfg["outputs"]["splits_dir"]
    prep_meta = json.loads((splits_dir / "prep_metadata.json").read_text())
    n, seed, conditions = scfg["sizes"][args.mode], scfg["seed"], scfg["conditions"]

    pool = selection.load_train_pool(splits_dir)
    thresholds = selection.check_thresholds(pool, prep_meta["bucket_quantiles"], prep_meta["bucket_thresholds_from_train"])
    selected, feas = selection.select_all(pool, conditions, n, seed, scfg["match_source"])
    unmatched = selection.unmatched_reference(pool, conditions, n, seed)
    quotas = selection.source_quotas(pool, n)

    mdir = ROOT / selection.manifest_dir(scfg["manifest_root"], args.mode)
    files = {f"{c}.jsonl": selection.write_manifest(df, mdir / f"{c}.jsonl") for c, df in selected.items()}
    ev = data.build_eval_manifest(splits_dir, scfg["eval"]["size"], scfg["eval"]["seed"])
    ev_path = ROOT / selection.eval_manifest_path(scfg["manifest_root"], scfg["eval"]["size"], scfg["eval"]["seed"])
    ev_sha = data.write_jsonl(ev, ev_path, [f for f in selection.MANIFEST_FIELDS])
    meta = {
        "mode": args.mode, "n_per_condition": n, "seed": seed, "match_source": scfg["match_source"],
        "conditions": conditions, "bucket_quantiles": prep_meta["bucket_quantiles"], "thresholds": thresholds,
        "source_quotas": quotas, "train_pool_size": len(pool),
        "split_manifest_sha256": prep_meta["manifest_sha256"],
        "dataset_name": prep_meta["dataset_name"], "dataset_revision": prep_meta["dataset_revision"],
        "manifest_sha256": files,
        "eval_manifest": {"path": str(ev_path.relative_to(ROOT)), "sha256": ev_sha, **scfg["eval"]},
        "feasibility": feas.to_dict(orient="records"),
    }
    (mdir / "selection_metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")

    rdir = ROOT / "results" / "selection" / args.mode
    rdir.mkdir(parents=True, exist_ok=True)
    summary = summarize(selected, unmatched, pool, conditions, feas, thresholds, n)
    summary.to_csv(rdir / "selected_pairs_summary.csv", index=False, float_format="%.6g", lineterminator="\n")
    allsel = pd.concat(selected.values(), ignore_index=True)
    pd.crosstab(allsel["source"], allsel["condition"]).to_csv(rdir / "source_counts.csv", lineterminator="\n")
    overlaps = overlap_matrix(selected)
    overlaps.to_csv(rdir / "overlap_matrix.csv", lineterminator="\n")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update({"pdf.fonttype": 42, "axes.titlesize": 9.5})
    fdir = ROOT / "figures" / "selection" / args.mode
    fig_balance(selected, feas, n, fdir)
    fig_margin(selected, pool, conditions, fdir)
    fig_sources(selected, pool, conditions, fdir)
    fig_lengths(selected, fdir)

    rep = ROOT / "reports" / "selection" / args.mode
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "selection_report.md").write_text(selection_report(args.mode, n, summary, quotas, overlaps, thresholds))
    if not args.no_examples:
        from datasets import load_dataset

        raw = load_dataset(prep_meta["dataset_name"], revision=prep_meta["dataset_revision"], split="train")
        (rep / "sample_examples.md").write_text(sample_examples(selected, raw, seed))

    print(f"[{args.mode}] {n} pairs x {len(selected)} conditions from a train pool of {len(pool):,}")
    print(summary[["condition", "n_selected", "n_unique_prompts", "min_source_headroom", "total_shortfall",
                   "margin_mean", "margin_mean_unmatched_sample", "source_tvd_vs_train_pool",
                   "source_tvd_unmatched_vs_train_pool"]].to_string(index=False))
    print(f"Manifests: {mdir.relative_to(ROOT)}  Summary: {rdir.relative_to(ROOT)}  Figures: {fdir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
