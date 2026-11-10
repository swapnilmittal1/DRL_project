#!/usr/bin/env python
"""Train one LoRA (or CUDA-only QLoRA) DPO adapter on Qwen2.5-1.5B-Instruct for one condition and seed.

Every run starts from the same point: a fresh process loads the pinned base checkpoint, TRL calls
set_seed(seed) before LoRA initialization, and the frozen base weights (adapter disabled) are the DPO
reference policy. Fingerprints of the base weights and of the initial LoRA weights are recorded so
identical starting points and an unchanged reference can be verified across conditions.

Usage:
  python scripts/train_dpo.py --condition E1_random --smoke-test            # ~100 pairs, 12 steps
  python scripts/train_dpo.py --condition E1_random --run-type pilot --max-steps 62
  python scripts/train_dpo.py --condition E3_medium_margin --seed 1          # final (4,000 pairs)
  python scripts/train_dpo.py --config configs/smoke/smoke_tiny.yaml         # tiny random model, any backend

Outputs (default --output-dir results/<run_type>/<experiment_id>/seed<seed>/):
  adapter/                 LoRA adapter + tokenizer (reloadable with peft.PeftModel.from_pretrained)
  resolved_config.yaml     merged config after --smoke-test/--override/CLI flags, plus resolved device/dtype
  provenance.json          git state, package versions, device, command line
  requirements_frozen.txt  pip freeze of the environment
  history.csv              one row per log event: TRL metrics, LR, step, wall clock, pairs seen, memory
  log_history.json         raw Trainer log history
  trainer_state.json       Trainer state at the end of training
  final_metrics.json       train_output metrics + final evaluation on the validation subset
  run_metadata.json        status, seeds, manifest hashes, data/truncation stats, masking check,
                           weight fingerprints, adapter file hashes, reload check, timings, peak memory
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from dpo_margin import provenance, training  # noqa: E402
from dpo_margin.config import CONDITIONS, RUN_TYPES, load_config  # noqa: E402
from dpo_margin.device import check_quantization, detect_device, resolve_dtype  # noqa: E402
from dpo_margin.prepare import sha256_text  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=CONDITIONS, help="training condition (sets data.condition and experiment_id)")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    p.add_argument("--seed", type=int, help="training seed (LoRA init, data order)")
    p.add_argument("--max-steps", type=int, help="cap on optimizer steps (identical budgets across conditions)")
    p.add_argument("--smoke-test", action="store_true", help="apply the config's smoke_test overlay (100 pairs, <=20 steps)")
    p.add_argument("--run-type", choices=RUN_TYPES, help="which manifests to use: smoke (100), pilot (500), final (4000)")
    p.add_argument("--output-dir", type=Path, help="default: results/<run_type>/<experiment_id>/seed<seed>")
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. training.learning_rate=1e-5")
    p.add_argument("--allow-dirty", action="store_true", help="allow final runs from an uncommitted tree (recorded)")
    p.add_argument("--overwrite", action="store_true", help="replace a completed run in --output-dir")
    p.add_argument("--dry-run", action="store_true", help="resolve config and data, then stop")
    return p.parse_args()


def resolve_config(args) -> dict:
    overrides = list(args.override)
    if args.run_type:
        overrides.append(f"run_type={args.run_type}")
    if args.condition:
        overrides += [f"data.condition={args.condition}", f"experiment_id={args.condition}"]
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.max_steps is not None:
        overrides.append(f"training.max_steps={args.max_steps}")
    return load_config(args.config, overrides, smoke_test=args.smoke_test)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    if not cfg["train"]:
        print(f"{cfg['experiment_id']}: train=false (baseline). Nothing to train.")
        return 0

    git = provenance.git_state()
    if cfg["run_type"] == "final" and (git["commit"] is None or git["dirty"]) and not args.allow_dirty:
        raise SystemExit("Final runs require a committed, clean git tree (or --allow-dirty, which is recorded).")

    info = detect_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"), info)
    qlora = bool(cfg["quantization"]["load_in_4bit"])
    check_quantization(qlora, info)
    if qlora and dtype != "bfloat16":
        raise SystemExit("QLoRA here requires bf16 compute (TRL casts adapters to bf16); this GPU lacks bf16.")

    out = (args.output_dir or ROOT / "results" / cfg["run_type"] / cfg["experiment_id"] / f"seed{cfg['seed']}")
    out = out if out.is_absolute() else ROOT / out
    if (out / "run_metadata.json").exists() and not args.overwrite:
        prev = json.loads((out / "run_metadata.json").read_text()).get("status")
        if prev == "completed":
            raise SystemExit(f"{out} already has a completed run; pass --overwrite to replace it.")

    run_data = training.load_run_data(ROOT, cfg)
    print(f"[{cfg['experiment_id']}] backend={info.backend} dtype={dtype} qlora={qlora} "
          f"train={len(run_data['train'])} eval={len(run_data['eval'])} -> {out}")
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    meta = {"status": "running", "experiment_id": cfg["experiment_id"], "condition": cfg["data"]["condition"],
            "run_type": cfg["run_type"], "smoke_test": args.smoke_test, "started_unix": time.time(),
            "seeds": {"training_seed": cfg["seed"], "data_seed": cfg["seed"],
                      "selection_seed": run_data["info"]["selection_seed"],
                      "eval_subset_seed": run_data["info"]["eval_seed"]},
            "data": run_data["info"]}
    write_json(out / "run_metadata.json", meta)

    def on_sigterm(signum, frame):
        raise SystemExit(f"terminated by signal {signum}")

    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        run(cfg, args, info, dtype, qlora, run_data, out, meta)
    except BaseException as e:  # record real failures, then re-raise
        meta.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        write_json(out / "run_metadata.json", meta)
        raise
    return 0


def run(cfg, args, info, dtype, qlora, run_data, out, meta) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    t_start = time.time()
    resolved = {**cfg, "resolved": {"backend": info.backend, "dtype": dtype, "qlora": qlora,
                                    "config_file": str(args.config), "output_dir": str(out),
                                    **run_data["info"]}}
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    write_json(out / "provenance.json", {**provenance.collect(), "device": info.to_dict(),
                                         "allow_dirty": args.allow_dirty})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
                            capture_output=True, text=True)
    (out / "requirements_frozen.txt").write_text(freeze.stdout)

    mcfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name_or_path"], revision=mcfg["revision"])
    if not tokenizer.chat_template:
        raise RuntimeError("tokenizer has no chat template")
    quant = None
    if qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    dpo_args = DPOConfig(
        output_dir=str(out / "trainer"),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        bf16=dtype == "bfloat16",
        use_cpu=info.backend == "cpu",
        model_init_kwargs={"revision": mcfg["revision"], "dtype": dtype,
                           "attn_implementation": mcfg.get("attn_implementation", "sdpa")},
        **cfg["dpo"],
        **cfg["training"],
    )
    # Model passed by ID: TRL calls set_seed(seed), loads the pinned checkpoint, then initializes LoRA.
    # With PEFT and no ref_model, TRL computes reference log-probs with the adapter disabled.
    trainer = DPOTrainer(
        model=mcfg["name_or_path"],
        args=dpo_args,
        train_dataset=run_data["train"],
        eval_dataset=run_data["eval"],
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
        quantization_config=quant,
    )
    model = trainer.accelerator.unwrap_model(trainer.model)

    # ---- preflight checks
    lengths = {"train": training.length_stats(trainer.train_dataset, len(run_data["train"]), dpo_args.max_length),
               "eval": training.length_stats(trainer.eval_dataset, len(run_data["eval"]), dpo_args.max_length)}
    dropped = lengths["train"]["n_pairs_dropped_prompt_too_long"]
    if cfg["run_type"] == "final" and dropped > cfg["checks"]["max_dropped_pairs_final"]:
        raise SystemExit(f"TRL would drop {dropped} train pairs whose prompt is >= max_length={dpo_args.max_length}; "
                         "this breaks equal N across conditions. Raise dpo.max_length (2048 = prep limit, no drops).")
    if dropped or lengths["train"]["n_pairs_truncated"]:
        print(f"NOTE: max_length={dpo_args.max_length}: {dropped} train pairs dropped (prompt too long), "
              f"{lengths['train']['n_pairs_truncated']} truncated.")
    masking = training.check_completion_masking(trainer, tokenizer)
    fp_init = training.base_and_adapter_fingerprints(model)
    probe = training.probe_batch(trainer)
    ref_before = training.completion_logps(model, probe, disable_adapter=True)
    policy_before = training.completion_logps(model, probe)
    init_equals_ref = all(abs(a - b) <= 1e-4 * max(1.0, abs(a)) for a, b in zip(policy_before, ref_before))
    if not init_equals_ref:
        raise RuntimeError(f"initial policy != reference before training: {policy_before} vs {ref_before}")
    eff_batch = dpo_args.per_device_train_batch_size * dpo_args.gradient_accumulation_steps * dpo_args.world_size
    meta.update(
        model={"name_or_path": mcfg["name_or_path"], "revision": mcfg["revision"], "dtype": dtype, "qlora": qlora,
               "chat_template_sha256": sha256_text(tokenizer.chat_template)},
        reference_policy=("frozen base checkpoint (LoRA adapter disabled); same pinned revision and dtype in every run"
                          if not qlora else "frozen 4-bit NF4 quantization of the base checkpoint (adapter disabled); "
                          "QLoRA runs must not be compared with LoRA runs"),
        data_lengths=lengths, completion_masking=masking,
        initial_state={**fp_init, "probe_ref_logps": ref_before, "probe_policy_logps": policy_before,
                       "initial_policy_equals_reference": init_equals_ref},
        effective_batch_pairs=eff_batch, preflight_s=round(time.time() - t_start, 1),
    )
    write_json(out / "run_metadata.json", meta)
    print(f"preflight ok: masking={masking} base={fp_init['base_weights_sha256'][:12]} "
          f"lora_init={fp_init['lora_weights_sha256'][:12]} trainable={fp_init['n_trainable_params']:,}")

    # ---- train
    logger = training.RunLogger(out / "history.csv", info.backend, eff_batch)
    trainer.add_callback(logger)
    t_train = time.time()
    train_output = trainer.train()
    train_s = time.time() - t_train
    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    write_json(out / "log_history.json", trainer.state.log_history)

    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    final_eval = (evals[-1] if evals and evals[-1].get("step") == trainer.state.global_step
                  else trainer.evaluate())
    train_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    finite = bool(train_losses) and all(math.isfinite(x) for x in train_losses) and math.isfinite(final_eval["eval_loss"])
    write_json(out / "final_metrics.json", {"train": train_output.metrics, "eval": final_eval})

    fp_after = training.base_and_adapter_fingerprints(model)
    ref_after = training.completion_logps(model, probe, disable_adapter=True)
    policy_after = training.completion_logps(model, probe)
    meta.update(
        status="trained",
        global_step=trainer.state.global_step, epoch=trainer.state.epoch, train_runtime_s=round(train_s, 1),
        tokens_processed=trainer._total_train_tokens,
        pairs_processed=trainer.state.global_step * eff_batch,
        losses_finite=finite,
        final_state={**fp_after, "probe_ref_logps": ref_after, "probe_policy_logps": policy_after,
                     "base_weights_unchanged": fp_after["base_weights_sha256"] == fp_init["base_weights_sha256"],
                     "lora_weights_changed": fp_after["lora_weights_sha256"] != fp_init["lora_weights_sha256"]},
        peak_memory=logger.peak | training.memory_stats(info.backend),
        adapter_files_sha256=training.sha256_dir(adapter_dir),
    )
    write_json(out / "run_metadata.json", meta)
    if not finite:
        raise FloatingPointError(f"non-finite loss: train={train_losses[-5:]} eval={final_eval.get('eval_loss')}")
    if not meta["final_state"]["base_weights_unchanged"]:
        raise RuntimeError("base (reference) weights changed during training")

    # ---- reload the saved adapter on a freshly loaded base model and compare log-probs
    if cfg["checks"]["verify_reload"]:
        trained_lora = {n: p.detach().float().cpu() for n, p in model.named_parameters() if "lora_" in n}
        del trainer, model
        gc.collect()
        if info.backend == "cuda":
            torch.cuda.empty_cache()
        elif info.backend == "mps":
            torch.mps.empty_cache()
        meta["reload_check"] = reload_check(cfg, dtype, quant, info, adapter_dir, probe, policy_after, ref_after,
                                            trained_lora)
        if not meta["reload_check"]["passed"]:
            write_json(out / "run_metadata.json", meta)
            raise RuntimeError(f"adapter reload check failed: {meta['reload_check']}")

    meta.update(status="completed", total_runtime_s=round(time.time() - t_start, 1), finished_unix=time.time())
    write_json(out / "run_metadata.json", meta)
    ev = final_eval
    print(f"Done in {meta['total_runtime_s']}s: steps={meta['global_step']} pairs={meta['pairs_processed']} "
          f"tokens={meta['tokens_processed']:,} train_loss={train_output.metrics.get('train_loss'):.4f} "
          f"eval_loss={ev['eval_loss']:.4f} eval_reward_acc={ev.get('eval_rewards/accuracies')} "
          f"eval_margin={ev.get('eval_rewards/margins')} reload={meta.get('reload_check', {}).get('passed')} "
          f"-> {out}")


def reload_check(cfg, dtype, quant, info, adapter_dir, probe, policy_after, ref_after, trained_lora) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    mcfg = cfg["model"]
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["name_or_path"], revision=mcfg["revision"], dtype=dtype, quantization_config=quant,
        attn_implementation=mcfg.get("attn_implementation", "sdpa"))
    if quant is None:
        base = base.to({"cuda": "cuda", "mps": "mps"}.get(info.backend, "cpu"))
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    reloaded = {n: p.detach().float().cpu() for n, p in model.named_parameters() if "lora_" in n}
    same_names = set(reloaded) == set(trained_lora)
    max_w = max((float((reloaded[n] - trained_lora[n]).abs().max()) for n in trained_lora if n in reloaded),
                default=float("inf"))
    pol = training.completion_logps(model, probe)
    ref = training.completion_logps(model, probe, disable_adapter=True)
    tol = 1e-3 if dtype == "float32" and quant is None else 5e-2

    def close(a, b):
        return all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in zip(a, b))

    res = {"lora_param_names_match": same_names, "max_abs_lora_weight_diff": max_w,
           "reloaded_policy_logps": pol, "trained_policy_logps": policy_after,
           "reloaded_ref_logps": ref, "trained_ref_logps": ref_after, "rel_tolerance": tol,
           "policy_logps_match": close(pol, policy_after), "ref_logps_match": close(ref, ref_after)}
    res["passed"] = bool(same_names and max_w == 0.0 and res["policy_logps_match"] and res["ref_logps_match"])
    del model, base
    gc.collect()
    return res


if __name__ == "__main__":
    sys.exit(main())
