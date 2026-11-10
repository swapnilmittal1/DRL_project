"""Checks on the real selection manifests. Each mode is skipped until select_pairs.py has produced it."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dpo_margin import selection
from dpo_margin.prepare import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCFG = yaml.safe_load((ROOT / "configs/data/selection.yaml").read_text())
SPLITS = ROOT / "data" / "splits"
MODES = list(SCFG["sizes"])


def _mode_dir(mode):
    return ROOT / selection.manifest_dir(SCFG["manifest_root"], mode)


def _load(mode):
    d = _mode_dir(mode)
    if not (d / "selection_metadata.json").exists():
        pytest.skip(f"run scripts/select_pairs.py --mode {mode}")
    meta = json.loads((d / "selection_metadata.json").read_text())
    return meta, {c: selection.read_manifest(d / f"{c}.jsonl") for c in SCFG["conditions"]}


@pytest.fixture(scope="module")
def splits():
    if not (SPLITS / "train.csv").exists():
        pytest.skip("run scripts/prepare_data.py")
    return {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ("train", "validation", "test")}


@pytest.mark.parametrize("mode", MODES)
def test_required_number_of_examples(mode):
    meta, sel = _load(mode)
    n = SCFG["sizes"][mode]
    assert meta["n_per_condition"] == n
    for c, df in sel.items():
        assert len(df) == n, c
        assert df["prompt_sha256"].is_unique, f"{c}: duplicate prompts"
        assert (df["condition"] == c).all()


@pytest.mark.parametrize("mode", MODES)
def test_no_cross_split_contamination(mode, splits):
    _, sel = _load(mode)
    train_ids = set(splits["train"]["row_id"])
    held_ids = set(splits["validation"]["row_id"]) | set(splits["test"]["row_id"])
    held_prompts = set(splits["validation"]["prompt_sha256"]) | set(splits["test"]["prompt_sha256"])
    for c, df in sel.items():
        assert set(df["row_id"]) <= train_ids, c
        assert not set(df["row_id"]) & held_ids, c
        assert not set(df["prompt_sha256"]) & held_prompts, c


@pytest.mark.parametrize("mode", MODES)
def test_manifest_rows_match_train_split(mode, splits):
    _, sel = _load(mode)
    train = splits["train"].set_index("row_id")
    for c, df in sel.items():
        ref = train.loc[df["row_id"]]
        for col in ("orig_index", "prompt_sha256", "pair_sha256", "source", "margin", "bucket"):
            assert (ref[col].to_numpy() == df[col].to_numpy()).all(), (c, col)


@pytest.mark.parametrize("mode", MODES)
def test_equal_source_counts_and_margin_rules(mode):
    meta, sel = _load(mode)
    t1, t2 = meta["thresholds"]
    counts = [df["source"].value_counts().sort_index().to_dict() for df in sel.values()]
    assert all(c == counts[0] for c in counts)
    assert counts[0] == dict(sorted(meta["source_quotas"].items()))
    assert (sel["E1_random"]["margin"] > 0).all()
    assert (sel["E2_high_margin"]["margin"] > t2).all()
    assert ((sel["E3_medium_margin"]["margin"] > t1) & (sel["E3_medium_margin"]["margin"] <= t2)).all()
    assert ((sel["E4_low_margin"]["margin"] > 0) & (sel["E4_low_margin"]["margin"] <= t1)).all()


@pytest.mark.parametrize("mode", MODES)
def test_checksums_and_shared_provenance(mode):
    meta, _ = _load(mode)
    d = _mode_dir(mode)
    for name, digest in meta["manifest_sha256"].items():
        assert sha256_file(d / name) == digest
    assert sha256_file(ROOT / meta["eval_manifest"]["path"]) == meta["eval_manifest"]["sha256"]
    prep = json.loads((SPLITS / "prep_metadata.json").read_text())
    assert meta["split_manifest_sha256"] == prep["manifest_sha256"]
    assert meta["thresholds"] == prep["bucket_thresholds_from_train"]


def test_eval_manifest_is_validation_only(splits):
    path = ROOT / selection.eval_manifest_path(SCFG["manifest_root"], SCFG["eval"]["size"], SCFG["eval"]["seed"])
    if not path.exists():
        pytest.skip("eval manifest not built")
    ev = selection.read_manifest(path)
    assert len(ev) == SCFG["eval"]["size"]
    assert set(ev["row_id"]) <= set(splits["validation"]["row_id"])
