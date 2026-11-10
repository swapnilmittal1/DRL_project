import numpy as np
import pandas as pd
import pytest

from dpo_margin import selection
from dpo_margin.prepare import assign_bucket, compute_thresholds, sha256_file

CONDITIONS = {"E1_random": None, "E2_high_margin": "high", "E3_medium_margin": "medium", "E4_low_margin": "low"}
GRID = np.arange(0.25, 4.01, 0.25)


def make_pool(n=8000, seed=0) -> pd.DataFrame:
    """Synthetic train pool where margin depends on source (so matching has something to correct)."""
    rng = np.random.default_rng(seed)
    sources = rng.choice(["alpha", "beta", "gamma"], size=n, p=[0.5, 0.3, 0.2])
    skew = {"alpha": 1.6, "beta": 1.0, "gamma": 0.6}
    margins = np.array([GRID[min(len(GRID) - 1, int(rng.exponential(skew[s] * 3)))] for s in sources])
    chosen = np.minimum(5.0, 1.0 + margins + 0.25 * rng.integers(0, 3, n))
    df = pd.DataFrame({
        "row_id": [f"{i:05d}-{i:010x}" for i in range(n)], "orig_index": np.arange(n),
        "prompt_sha256": [f"p{i:063d}" for i in range(n)], "pair_sha256": [f"q{i:063d}" for i in range(n)],
        "source": sources, "margin": margins, "chosen_rating": chosen, "rejected_rating": chosen - margins,
        "chosen_model": rng.choice(["m1", "m2", "m3"], n), "rejected_model": rng.choice(["m4", "m5"], n),
        "prompt_tokens": rng.integers(30, 400, n), "chosen_tokens": rng.integers(1, 800, n),
        "rejected_tokens": rng.integers(1, 800, n),
    })
    df["token_length_diff"] = df["chosen_tokens"] - df["rejected_tokens"]
    df["bucket"] = assign_bucket(df["margin"], compute_thresholds(df["margin"].to_numpy(), [1 / 3, 2 / 3]))
    return df


@pytest.fixture
def splits_dir(tmp_path):
    d = tmp_path / "splits"
    d.mkdir()
    make_pool().to_csv(d / "train.csv", index=False)
    (d / "validation.csv").write_text("\x00 not a csv \x00")  # selection must never read these
    (d / "test.csv").write_text("\x00 not a csv \x00")
    (d / "SHA256SUMS").write_text(f"{sha256_file(d / 'train.csv')}  train.csv\n")
    return d


@pytest.fixture
def pool(splits_dir):
    return selection.load_train_pool(splits_dir)


@pytest.mark.parametrize("n", [100, 500])
def test_every_condition_has_exactly_n(pool, n):
    sel, feas = selection.select_all(pool, CONDITIONS, n, seed=0)
    assert set(sel) == set(CONDITIONS)
    for c, df in sel.items():
        assert len(df) == n, c
        assert df["prompt_sha256"].is_unique and df["row_id"].is_unique
    assert feas["shortfall"].sum() == 0


def test_source_counts_identical_across_conditions_and_match_pool(pool):
    n = 500
    sel, _ = selection.select_all(pool, CONDITIONS, n, seed=0)
    quotas = selection.source_quotas(pool, n)
    assert sum(quotas.values()) == n
    expected = pool["source"].value_counts(normalize=True) * n
    assert all(abs(quotas[s] - expected[s]) < 1 for s in quotas)
    for df in sel.values():
        assert df["source"].value_counts().sort_index().to_dict() == dict(sorted(quotas.items()))


def test_margin_rules(pool):
    sel, _ = selection.select_all(pool, CONDITIONS, 300, seed=0)
    t = compute_thresholds(pool["margin"].to_numpy(), [1 / 3, 2 / 3])
    assert (sel["E1_random"]["margin"] > 0).all()
    assert (sel["E2_high_margin"]["margin"] > t[1]).all()
    assert sel["E3_medium_margin"]["margin"].between(t[0], t[1], inclusive="right").all()
    assert ((sel["E4_low_margin"]["margin"] > 0) & (sel["E4_low_margin"]["margin"] <= t[0])).all()


def test_margin_conditions_disjoint(pool):
    sel, _ = selection.select_all(pool, CONDITIONS, 300, seed=0)
    ids = {c: set(df["row_id"]) for c, df in sel.items()}
    for a in ("E2_high_margin", "E3_medium_margin", "E4_low_margin"):
        for b in ("E2_high_margin", "E3_medium_margin", "E4_low_margin"):
            if a < b:
                assert not ids[a] & ids[b]


def test_reproducible_with_same_seed_and_changes_with_seed(pool):
    a, _ = selection.select_all(pool, CONDITIONS, 200, seed=0)
    b, _ = selection.select_all(pool, CONDITIONS, 200, seed=0)
    c, _ = selection.select_all(pool, CONDITIONS, 200, seed=1)
    for k in CONDITIONS:
        pd.testing.assert_frame_equal(a[k], b[k])
        assert a[k]["row_id"].tolist() != c[k]["row_id"].tolist()


def test_infeasible_fails_with_diagnostic_and_no_silent_resize(pool):
    n_high = int((pool["bucket"] == "high").sum())
    with pytest.raises(selection.SelectionError) as err:
        selection.select_all(pool, CONDITIONS, n_high + 50, seed=0)
    msg = str(err.value)
    assert "infeasible" in msg and "required" in msg and "available" in msg and "E2_high_margin" in msg


def test_unmatched_mode_also_refuses_to_shrink(pool):
    with pytest.raises(selection.SelectionError, match="needs"):
        selection.select_condition(pool, "E2_high_margin", "high", len(pool), seed=0, quotas=None)


def test_duplicate_prompts_rejected(splits_dir):
    df = pd.read_csv(splits_dir / "train.csv")
    df.loc[1, "prompt_sha256"] = df.loc[0, "prompt_sha256"]
    df.to_csv(splits_dir / "train.csv", index=False)
    (splits_dir / "SHA256SUMS").write_text(f"{sha256_file(splits_dir / 'train.csv')}  train.csv\n")
    with pytest.raises(selection.SelectionError, match="duplicate"):
        selection.load_train_pool(splits_dir)


def test_tampered_train_split_rejected(splits_dir):
    with open(splits_dir / "train.csv", "a") as f:
        f.write("x\n")
    with pytest.raises(selection.SelectionError, match="SHA256SUMS"):
        selection.load_train_pool(splits_dir)


def test_threshold_mismatch_rejected(pool):
    with pytest.raises(selection.SelectionError, match="differ"):
        t = compute_thresholds(pool["margin"].to_numpy(), [1 / 3, 2 / 3])
        selection.check_thresholds(pool, [1 / 3, 2 / 3], [t[0], t[1] + 0.25])


def test_matching_corrects_source_skew(pool):
    from dpo_margin.selection_diagnostics import tvd

    n = 400
    sel, _ = selection.select_all(pool, CONDITIONS, n, seed=0)
    um = selection.unmatched_reference(pool, CONDITIONS, n, seed=0)
    assert tvd(um["E4_low_margin"]["source"], pool["source"]) > 0.05
    assert tvd(sel["E4_low_margin"]["source"], pool["source"]) < 0.01


def test_manifest_roundtrip(pool, tmp_path):
    sel, _ = selection.select_all(pool, CONDITIONS, 100, seed=0)
    path = tmp_path / "E1_random.jsonl"
    digest = selection.write_manifest(sel["E1_random"], path)
    assert digest == sha256_file(path)
    back = selection.read_manifest(path)
    assert list(back.columns) == selection.MANIFEST_FIELDS and len(back) == 100
    assert back["row_id"].tolist() == sel["E1_random"]["row_id"].tolist()
