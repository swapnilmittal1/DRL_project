"""End-to-end check that our DPOConfig/DPOTrainer/LoRA wiring works with the pinned TRL version.

Uses a tiny random-weight model on CPU and synthetic pairs; says nothing about model quality.
"""

import math

import pytest
from datasets import Dataset

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_REV = "edca82771e1fad8f0b7ce110ffe770c682043fec"


@pytest.mark.network
def test_dpo_lora_trains_a_few_steps(tmp_path):
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    ds = Dataset.from_list([
        {"prompt": [{"role": "user", "content": f"Question {i}?"}],
         "chosen": [{"role": "assistant", "content": f"A helpful answer {i}."}],
         "rejected": [{"role": "assistant", "content": "No."}]}
        for i in range(8)
    ])
    args = DPOConfig(
        output_dir=str(tmp_path), use_cpu=True, seed=0, data_seed=0, max_steps=2,
        per_device_train_batch_size=2, learning_rate=1e-3, beta=0.1, loss_type=["sigmoid"],
        max_length=64, logging_steps=1, save_strategy="no", report_to="none",
        gradient_checkpointing=False, model_init_kwargs={"revision": TINY_REV, "dtype": "float32"},
    )
    trainer = DPOTrainer(
        model=TINY_MODEL, args=args, train_dataset=ds,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
    )
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    assert losses and all(math.isfinite(x) for x in losses)
    trainer.save_model(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
