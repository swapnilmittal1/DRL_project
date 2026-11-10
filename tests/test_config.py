from pathlib import Path

import pytest

from dpo_margin.config import apply_overrides, deep_merge, load_config
from dpo_margin.prepare import load_prep_config

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
RUN_CONFIGS = [*EXPERIMENTS, *sorted((ROOT / "configs" / "smoke").glob("*.yaml"))]


def test_all_run_configs_load():
    for path in RUN_CONFIGS:
        load_config(path)


def test_base_template_refuses_to_train_without_condition():
    with pytest.raises(ValueError, match="condition"):
        load_config(ROOT / "configs" / "base.yaml")


def test_prep_config_loads():
    load_prep_config(ROOT / "configs" / "data" / "prep.yaml")


def test_experiments_differ_only_in_id_condition_and_train_flag():
    cfgs = [load_config(p) for p in EXPERIMENTS]
    assert {c["experiment_id"] for c in cfgs} == {"E0_no_dpo", "E1_random", "E2_high_margin",
                                                  "E3_medium_margin", "E4_low_margin"}
    for c in cfgs:
        if c["train"]:
            assert c["data"]["condition"] == c["experiment_id"]

    def strip(c):
        c = deep_merge(c, {})
        c.pop("experiment_id"), c.pop("train")
        c["data"].pop("condition")
        return c

    reference = strip(cfgs[0])
    for c in cfgs[1:]:
        assert strip(c) == reference


def test_max_length_within_prep_overlong_filter():
    for p in EXPERIMENTS:
        cfg = load_config(p)
        prep = load_prep_config(ROOT / cfg["data"]["prep_config"])
        assert cfg["dpo"]["max_length"] <= prep["max_total_tokens"]


def test_initial_hyperparameters():
    cfg = load_config(EXPERIMENTS[1])
    assert cfg["seed"] == 42
    assert cfg["dpo"]["beta"] == 0.1 and cfg["dpo"]["max_length"] == 1024 and cfg["dpo"]["loss_type"] == ["sigmoid"]
    t = cfg["training"]
    assert (t["learning_rate"], t["num_train_epochs"], t["per_device_train_batch_size"],
            t["gradient_accumulation_steps"]) == (5e-6, 1, 1, 8)
    assert (cfg["lora"]["r"], cfg["lora"]["lora_alpha"]) == (16, 32)


def test_smoke_overlay_is_small_and_bounded():
    cfg = load_config(EXPERIMENTS[1], smoke_test=True)
    assert cfg["run_type"] == "smoke"
    steps, ga = cfg["training"]["max_steps"], cfg["training"]["gradient_accumulation_steps"]
    assert steps <= 20 and steps * ga * cfg["training"]["per_device_train_batch_size"] <= 100


def test_conditions_share_every_training_setting():
    cfgs = {c["experiment_id"]: c for c in map(load_config, EXPERIMENTS) if c["train"]}
    keys = ("model", "quantization", "lora", "dpo", "training", "seed", "checks")
    ref = cfgs["E1_random"]
    for c in cfgs.values():
        assert {k: c[k] for k in keys} == {k: ref[k] for k in keys}


def test_final_experiments_are_final_and_smoke_are_smoke():
    for p in EXPERIMENTS:
        assert load_config(p)["run_type"] == "final"
    for p in (ROOT / "configs" / "smoke").glob("*.yaml"):
        assert load_config(p)["run_type"] == "smoke"


def test_overrides_parse_yaml_scalars():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c=1e-5", "d=true"])
    assert cfg == {"a": {"b": 2, "c": 1e-5}, "d": True}


def test_invalid_condition_rejected():
    with pytest.raises(ValueError):
        load_config(EXPERIMENTS[1], ["data.condition=bogus"])


def test_selection_sizes_match_spec():
    import yaml

    scfg = yaml.safe_load((ROOT / "configs" / "data" / "selection.yaml").read_text())
    assert scfg["sizes"] == {"smoke": 100, "pilot": 500, "final": 4000}
    assert list(scfg["conditions"]) == ["E1_random", "E2_high_margin", "E3_medium_margin", "E4_low_margin"]
