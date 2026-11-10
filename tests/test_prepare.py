import json

import numpy as np
import pandas as pd
import pytest
from conftest import SPECIAL_ROWS, build_rows

from datasets import Dataset

from dpo_margin import prepare
from dpo_margin.selection import select_condition


@pytest.fixture
def prepared(raw_dataset, tokenizer, prep_cfg):
    return prepare.prepare(raw_dataset, tokenizer, prep_cfg)


def test_each_failure_mode_gets_its_reason(prepared):
    df, _ = prepared
    for i, (_, expected) in enumerate(SPECIAL_ROWS):
        got = df.loc[df["orig_index"] == i, "drop_reason"].item()
        assert (got if isinstance(got, str) else None) == expected, f"row {i}"


def test_drop_counts_recorded_for_every_reason(prepared):
    df, meta = prepared
    assert set(meta["dropped_by_reason"]) == set(prepare.DROP_REASONS)
    assert sum(meta["dropped_by_reason"].values()) == meta["n_dropped"] == df["drop_reason"].notna().sum()
    assert meta["n_kept"] + meta["n_dropped"] == meta["n_raw"] == len(df)


def test_all_kept_margins_positive(prepared):
    df, _ = prepared
    kept = df[df["drop_reason"].isna()]
    assert len(kept) > 0 and (kept["margin"] > 0).all()


@pytest.mark.parametrize("bucket", [None, "low", "medium", "high"])
def test_all_selected_margins_positive(prepared, bucket):
    df, _ = prepared
    train = df[df["drop_reason"].isna() & (df["split"] == "train")]
    picked = select_condition(train, "c", bucket, 20, seed=0, quotas=None)
    assert (picked["margin"] > 0).all()


def test_no_empty_responses_survive(prepared, raw_dataset):
    df, _ = prepared
    for i in df.loc[df["drop_reason"].isna(), "orig_index"]:
        row = raw_dataset[int(i)]
        assert row["chosen"][-1]["content"].strip() and row["rejected"][-1]["content"].strip()
        assert row["prompt"].strip()


def test_score_and_length_calculations(prepared, raw_dataset, tokenizer):
    df, _ = prepared
    kept = df[df["drop_reason"].isna()]
    for _, r in kept.iterrows():
        raw = raw_dataset[int(r["orig_index"])]
        assert r["margin"] == raw["chosen-rating"] - raw["rejected-rating"]
        assert r["chosen_rating"] == raw["chosen-rating"] and r["rejected_rating"] == raw["rejected-rating"]
        assert r["chosen_tokens"] == len(raw["chosen"][-1]["content"].split())
        assert r["rejected_tokens"] == len(raw["rejected"][-1]["content"].split())
        assert r["token_length_diff"] == r["chosen_tokens"] - r["rejected_tokens"]
        prompt_str = tokenizer.apply_chat_template([raw["chosen"][0]], add_generation_prompt=True)
        assert r["prompt_tokens"] == len(prompt_str.split())
        assert r["source"] == raw["source"] and r["chosen_model"] == raw["chosen-model"]
        assert r["rejected_model"] == raw["rejected-model"]


def test_margin_is_exact_on_quarter_grid():
    assert prepare.compute_thresholds(np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0]), [0.5, 0.9]) == \
        [float(x) for x in np.quantile([0.25, 0.5, 0.75, 1.0, 2.0, 3.0], [0.5, 0.9])]
    np.testing.assert_array_equal(prepare.assign_bucket([0.25, 0.75, 1.0, 1.5, 1.75], [0.75, 1.5]),
                                  ["low", "low", "medium", "medium", "high"])


def test_splits_disjoint_by_prompt(prepared):
    df, _ = prepared
    kept = df[df["drop_reason"].isna()]
    sets = {s: set(kept.loc[kept["split"] == s, "prompt_sha256"]) for s in prepare.SPLITS}
    assert not (sets["train"] & sets["validation"]) and not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])
    assert kept["split"].isin(prepare.SPLITS).all()
    assert df.loc[df["drop_reason"].notna(), "split"].isna().all()


def test_duplicate_prompts_detected_after_normalization():
    assert prepare.normalize_prompt("  Duplicate \n  PROMPT\t") == "duplicate prompt"
    assert prepare.sha256_text(prepare.normalize_prompt("A  b")) == prepare.sha256_text(prepare.normalize_prompt("a b"))


def test_thresholds_use_train_only(prepared):
    df, meta = prepared
    kept = df[df["drop_reason"].isna()]
    train = kept[kept["split"] == "train"]
    assert meta["bucket_thresholds_from_train"] == prepare.compute_thresholds(train["margin"].to_numpy(), [1 / 3, 2 / 3])
    np.testing.assert_array_equal(kept["bucket"], prepare.assign_bucket(kept["margin"], meta["bucket_thresholds_from_train"]))


def test_thresholds_unchanged_when_heldout_margins_change(raw_dataset, tokenizer, prep_cfg):
    df, meta = prepare.prepare(raw_dataset, tokenizer, prep_cfg)
    heldout = df.loc[df["drop_reason"].isna() & df["split"].ne("train"), "orig_index"].tolist()
    rows = build_rows()
    for i in heldout:
        rows[i]["chosen-rating"], rows[i]["rejected-rating"] = 5.0, 1.0
    df2, meta2 = prepare.prepare(Dataset.from_list(rows), tokenizer, prep_cfg)
    assert meta2["bucket_thresholds_from_train"] == meta["bucket_thresholds_from_train"]
    assert df2.loc[df2["orig_index"].isin(heldout), "margin"].eq(4.0).all()


def test_split_is_independent_of_row_order(raw_dataset, tokenizer, prep_cfg):
    df, _ = prepare.prepare(raw_dataset, tokenizer, prep_cfg)
    shuffled = raw_dataset.shuffle(seed=123)
    df2, _ = prepare.prepare(shuffled, tokenizer, prep_cfg)
    a = df[df["drop_reason"].isna()].set_index("pair_sha256")["split"]
    b = df2[df2["drop_reason"].isna()].set_index("pair_sha256")["split"]
    assert a.sort_index().equals(b.reindex(a.index).sort_index())


def test_split_fractions_roughly_respected(prepared):
    df, _ = prepared
    frac = df.loc[df["drop_reason"].isna(), "split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.08


def test_outputs_byte_identical_across_runs(raw_dataset, tokenizer, prep_cfg, tmp_path):
    outs = []
    for k in range(2):
        df, meta = prepare.prepare(raw_dataset, tokenizer, prep_cfg)
        d = tmp_path / f"run{k}"
        prepare.write_outputs(df, meta, d / "splits", d / "features.parquet")
        outs.append(d)
    for name in ["train.csv", "validation.csv", "test.csv", "dropped.csv", "SHA256SUMS", "prep_metadata.json"]:
        assert (outs[0] / "splits" / name).read_bytes() == (outs[1] / "splits" / name).read_bytes(), name
    pd.testing.assert_frame_equal(pd.read_parquet(outs[0] / "features.parquet"),
                                  pd.read_parquet(outs[1] / "features.parquet"))


def test_different_seed_changes_split(raw_dataset, tokenizer, prep_cfg):
    a, _ = prepare.prepare(raw_dataset, tokenizer, prep_cfg)
    b, _ = prepare.prepare(raw_dataset, tokenizer, {**prep_cfg, "split": {**prep_cfg["split"], "seed": 1}})
    assert not a["split"].equals(b["split"])


def test_checksums_written_and_verified(prepared, tmp_path):
    df, meta = prepared
    meta = prepare.write_outputs(df, meta, tmp_path / "splits", tmp_path / "f.parquet")
    assert prepare.verify_checksums(tmp_path / "splits") == []
    stored = json.loads((tmp_path / "splits" / "prep_metadata.json").read_text())["manifest_sha256"]
    assert stored == meta["manifest_sha256"] and set(stored) == {"train.csv", "validation.csv", "test.csv", "dropped.csv"}
    with open(tmp_path / "splits" / "train.csv", "a") as f:
        f.write("tampered\n")
    assert prepare.verify_checksums(tmp_path / "splits") == ["train.csv"]


def test_row_ids_unique_and_stable(prepared):
    df, _ = prepared
    assert df["row_id"].is_unique
    assert all(r.startswith(f"{i:05d}-") for r, i in zip(df["row_id"], df["orig_index"]))
