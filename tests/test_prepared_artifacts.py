"""Checks on the real prepared outputs (data/splits). Skipped until scripts/prepare_data.py has run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dpo_margin import prepare

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
FEATURES = ROOT / "data" / "processed" / "pairs_features.parquet"

pytestmark = pytest.mark.skipif(not (SPLITS / "prep_metadata.json").exists(), reason="run prepare_data.py first")


@pytest.fixture(scope="module")
def meta():
    return json.loads((SPLITS / "prep_metadata.json").read_text())


@pytest.fixture(scope="module")
def splits():
    return {s: pd.read_csv(SPLITS / f"{s}.csv") for s in prepare.SPLITS}


def test_checksums_match():
    assert prepare.verify_checksums(SPLITS) == []


def test_counts_consistent(meta, splits):
    assert {s: len(df) for s, df in splits.items()} == meta["split_pairs"]
    dropped = pd.read_csv(SPLITS / "dropped.csv")
    assert len(dropped) == meta["n_dropped"] and sum(len(d) for d in splits.values()) == meta["n_kept"]
    assert dropped["drop_reason"].value_counts().to_dict() == {k: v for k, v in meta["dropped_by_reason"].items() if v}


def test_real_splits_disjoint_by_prompt(splits):
    sets = {s: set(df["prompt_sha256"]) for s, df in splits.items()}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])
    all_ids = pd.concat(splits.values())["row_id"]
    assert all_ids.is_unique


def test_real_margins_positive_and_buckets_match_train_thresholds(meta, splits):
    t = meta["bucket_thresholds_from_train"]
    assert t == prepare.compute_thresholds(splits["train"]["margin"].to_numpy(), meta["bucket_quantiles"])
    for df in splits.values():
        assert (df["margin"] > 0).all()
        np.testing.assert_array_equal(df["bucket"], prepare.assign_bucket(df["margin"], t))


@pytest.mark.skipif(not FEATURES.exists(), reason="feature table not generated")
def test_real_features_consistent():
    df = pd.read_parquet(FEATURES)
    kept = df[df["drop_reason"].isna()]
    np.testing.assert_allclose(kept["margin"], kept["chosen_rating"] - kept["rejected_rating"])
    assert (kept["chosen_tokens"] > 0).all() and (kept["rejected_tokens"] > 0).all()
    assert (kept["token_length_diff"] == kept["chosen_tokens"] - kept["rejected_tokens"]).all()
    assert (np.maximum(kept["total_chosen_tokens"], kept["total_rejected_tokens"]) <= 2048).all()
    assert kept["prompt_sha256"].is_unique


@pytest.mark.network
@pytest.mark.skipif(not FEATURES.exists(), reason="feature table not generated")
def test_real_kept_responses_non_empty(meta):
    from datasets import load_dataset

    ds = load_dataset(meta["dataset_name"], revision=meta["dataset_revision"], split="train")
    kept = pd.concat(pd.read_csv(SPLITS / f"{s}.csv") for s in prepare.SPLITS)["orig_index"].tolist()
    sub = ds.select(kept)
    for ch, rj in zip(sub["chosen"], sub["rejected"]):
        assert ch[-1]["content"].strip() and rj[-1]["content"].strip()
        assert ch[-1]["content"].strip() != rj[-1]["content"].strip()
