import pandas as pd
import pytest
from conftest import make_row

from dpo_margin import data, prepare


def msg(role, content):
    return {"role": role, "content": content}


def test_single_turn_prompt_only_in_prompt():
    ex = data.to_dpo_example(make_row("hi", "good", "bad", 5.0, 1.0))
    assert ex == {"prompt": [msg("user", "hi")], "chosen": [msg("assistant", "good")],
                  "rejected": [msg("assistant", "bad")]}


def test_multi_turn_context_preserved():
    ctx = [msg("system", "be brief"), msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")]
    row = {"chosen": ctx + [msg("assistant", "good")], "rejected": ctx + [msg("assistant", "bad")]}
    ex = data.to_dpo_example(row)
    assert ex["prompt"] == ctx
    assert ex["chosen"] == [msg("assistant", "good")] and ex["rejected"] == [msg("assistant", "bad")]


@pytest.mark.parametrize("breaker", [
    lambda c, r: r.__setitem__(1, msg("assistant", "different a1")),   # different context
    lambda c, r: r.insert(0, msg("system", "extra")),                   # different length
])
def test_context_mismatch_rejected(breaker):
    ctx = [msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")]
    chosen, rejected = ctx + [msg("assistant", "good")], list(ctx) + [msg("assistant", "bad")]
    breaker(chosen, rejected)
    with pytest.raises(ValueError, match="context"):
        data.to_dpo_example({"chosen": chosen, "rejected": rejected})


def test_context_must_end_with_user_and_completion_must_be_assistant():
    with pytest.raises(ValueError, match="end with a user"):
        data.to_dpo_example({"chosen": [msg("assistant", "x"), msg("assistant", "y")],
                             "rejected": [msg("assistant", "x"), msg("assistant", "z")]})
    with pytest.raises(ValueError, match="assistant response"):
        data.to_dpo_example({"chosen": [msg("user", "x"), msg("user", "y")],
                             "rejected": [msg("user", "x"), msg("assistant", "z")]})


def test_prompt_field_must_match_final_user_turn():
    row = make_row("hi", "good", "bad", 5.0, 1.0)
    row["prompt"] = "something else"
    with pytest.raises(ValueError, match="prompt"):
        data.to_dpo_example(row)


@pytest.mark.parametrize("ctx", [
    [msg("user", "q")],
    [msg("system", "s"), msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")],
])
def test_matches_trl_extract_prompt_and_is_conversational(ctx):
    from trl import extract_prompt
    from trl.data_utils import is_conversational

    row = {"chosen": ctx + [msg("assistant", "good")], "rejected": ctx + [msg("assistant", "bad")]}
    ours = data.to_dpo_example(row)
    assert ours == extract_prompt(row)
    assert is_conversational(ours)


def test_eval_manifest_reads_only_validation(tmp_path, raw_dataset, tokenizer, prep_cfg):
    df, meta = prepare.prepare(raw_dataset, tokenizer, prep_cfg)
    prepare.write_outputs(df, meta, tmp_path / "splits", tmp_path / "f.parquet")
    (tmp_path / "splits" / "train.csv").write_text("\x00 garbage")  # must not be read
    ev = data.build_eval_manifest(tmp_path / "splits", 10, seed=0)
    val = pd.read_csv(tmp_path / "splits" / "validation.csv")
    assert len(ev) == 10 and set(ev["row_id"]) <= set(val["row_id"])
    assert ev["row_id"].tolist() == data.build_eval_manifest(tmp_path / "splits", 10, seed=0)["row_id"].tolist()


def test_materialize_verifies_pair_hashes(raw_dataset, tokenizer, prep_cfg):
    df, _ = prepare.prepare(raw_dataset, tokenizer, prep_cfg)
    kept = df[df["drop_reason"].isna()].head(5)
    ds = data.materialize(raw_dataset, kept)
    assert ds.column_names == ["prompt", "chosen", "rejected"] and len(ds) == 5
    bad = kept.copy()
    bad.iloc[0, bad.columns.get_loc("pair_sha256")] = "0" * 64
    with pytest.raises(RuntimeError, match="hash"):
        data.materialize(raw_dataset, bad)
