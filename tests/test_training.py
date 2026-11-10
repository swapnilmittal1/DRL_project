"""Training utilities: masking check, fingerprints, length stats, and the train_dpo.py pipeline (tiny model)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from dpo_margin import training

ROOT = Path(__file__).resolve().parents[1]


def test_fingerprint_sensitive_to_values_and_names():
    a = [("w", torch.zeros(3)), ("v", torch.ones(2))]
    assert training.fingerprint(a) == training.fingerprint(list(reversed(a)))
    assert training.fingerprint(a) != training.fingerprint([("w", torch.zeros(3)), ("v", torch.ones(2) * 2)])
    assert training.fingerprint(a) != training.fingerprint([("x", torch.zeros(3)), ("v", torch.ones(2))])


def test_length_stats_counts_drops_and_truncation():
    tok = {"prompt_ids": [[1] * 5, [1] * 2], "chosen_ids": [[2] * 3, [2] * 2], "rejected_ids": [[3] * 7, [3]]}
    s = training.length_stats(tok, n_before=3, max_length=10)
    assert s["n_pairs_dropped_prompt_too_long"] == 1
    assert s["n_pairs_truncated"] == 1 and s["n_rejected_truncated"] == 1 and s["n_chosen_truncated"] == 0
    assert s["tokens_per_epoch"] == (8 + 10) + (4 + 3)


@pytest.mark.network
def test_masking_check_on_real_qwen_template(tmp_path):
    """TRL's collated batch: loss positions are exactly the assistant reply + <|im_end|>\\n."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    ctx = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
           {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "2+2?"}]
    ds = Dataset.from_list([{"prompt": ctx, "chosen": [{"role": "assistant", "content": "4"}],
                             "rejected": [{"role": "assistant", "content": "It is five, clearly."}]}] * 2)
    trainer = DPOTrainer(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        args=DPOConfig(output_dir=str(tmp_path), use_cpu=True, max_steps=1, report_to="none", save_strategy="no",
                       gradient_checkpointing=False, bf16=False,
                       model_init_kwargs={"revision": "edca82771e1fad8f0b7ce110ffe770c682043fec", "dtype": "float32"}),
        train_dataset=ds, processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj"]),
    )
    res = training.check_completion_masking(trainer, tok, n=2)
    assert res["completion_tokens_are_assistant_reply"] and res["prompt_tokens_excluded_from_loss"]
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["input_ids"].masked_fill(batch["completion_mask"] == 0, -100)
    assert tok.decode(labels[0][labels[0] != -100]) == "4<|im_end|>\n"
    assert "Hello!" not in tok.decode(labels[0][labels[0] != -100])  # earlier assistant turn is context, not target


@pytest.mark.network
def test_train_dpo_tiny_end_to_end(tmp_path):
    """Run the real CLI on the tiny model; check outputs, finite losses, fingerprints, and adapter reload."""
    if not (ROOT / "data/manifests/smoke/E2_high_margin.jsonl").exists():
        pytest.skip("run scripts/select_pairs.py --mode smoke")
    out = tmp_path / "run"
    cmd = [sys.executable, str(ROOT / "scripts/train_dpo.py"), "--config", str(ROOT / "configs/smoke/smoke_tiny.yaml"),
           "--output-dir", str(out), "--max-steps", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-3000:]
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["status"] == "completed" and meta["global_step"] == 2 and meta["losses_finite"]
    assert meta["initial_state"]["initial_policy_equals_reference"]
    assert meta["final_state"]["base_weights_unchanged"]
    assert meta["reload_check"]["passed"]
    assert meta["seeds"]["training_seed"] == 42 and len(meta["data"]["train_manifest_sha256"]) == 64
    hist = pd.read_csv(out / "history.csv")
    for col in ("loss", "learning_rate", "rewards/accuracies", "rewards/margins", "num_tokens", "wall_clock_s",
                "train_pairs_seen", "eval_loss"):
        assert col in hist.columns, col
    assert all(math.isfinite(x) for x in hist["loss"].dropna())
    for f in ("resolved_config.yaml", "provenance.json", "requirements_frozen.txt", "final_metrics.json",
              "trainer_state.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
        assert (out / f).exists(), f
