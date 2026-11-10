# Preference-Margin Data Selection for DPO

An extension of Ivison et al., *Unpacking DPO and PPO: Disentangling Best Practices for Learning
from Preference Feedback* (NeurIPS 2024). That paper found that preference **data quality** is one
of the most important factors for DPO and PPO. This project asks a narrower question about data
selection.

## Scientific question

> Given a fixed budget of preference pairs, does selecting DPO training pairs by the **margin
> between the chosen and rejected response ratings** change downstream performance, compared with
> random selection?

Margin is `chosen-rating - rejected-rating`, using the UltraFeedback ratings in
[`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned).
The ratings are averages on a 1 to 5 scale, in steps of 0.25.

## Experimental conditions

| ID | Config | Training data (4,000 train-split pairs for each of E1 to E4) |
|----|--------|-------------------------------------------|
| E0_no_dpo | `configs/experiments/e0_no_dpo.yaml` | none; evaluate `Qwen/Qwen2.5-1.5B-Instruct` as is |
| E1_random | `configs/experiments/e1_random.yaml` | random sample of all positive-margin pairs |
| E2_high_margin | `configs/experiments/e2_high_margin.yaml` | random sample of the top margin tercile (margin > t2 = 1.5) |
| E3_medium_margin | `configs/experiments/e3_medium_margin.yaml` | random sample of the middle tercile (0.75 < margin ≤ 1.5) |
| E4_low_margin | `configs/experiments/e4_low_margin.yaml` | random sample of the lowest positive tercile (0 < margin ≤ 0.75) |

E1 to E4 differ **only** in `data.condition`; `tests/test_config.py` enforces this. Training method
for every condition: DPO (sigmoid loss, β = 0.1) with a LoRA adapter on the base policy. The
intervention is the training data; the DPO loss is never changed.

### Data pipeline

The preparation step is `scripts/prepare_data.py`, configured by `configs/data/prep.yaml`, with the
logic in `src/dpo_margin/prepare.py`. It runs these steps in order:

1. **Validate.** Check the schema and the `[user, assistant]` turn structure of every row. Write 12
   seeded examples to `reports/example_inspection.md`.
2. **Clean.** Drop rows with malformed structure or ratings, empty prompts or responses, identical
   chosen and rejected responses, or margin ≤ 0. Also drop pairs longer than `max_total_tokens`
   (2048, measured with the Qwen chat template, so training never truncates a response), and
   duplicate prompts after normalization (the lowest index is kept). Each dropped row records its
   reason in `data/splits/dropped.csv`.
3. **Split by prompt.** Assign each pair to a split using a salted SHA-256 of its normalized prompt,
   giving about 90/5/5 train/validation/test. A prompt can never appear in two splits.
4. **Thresholds.** Compute t1 and t2 as the 1/3 and 2/3 quantiles of the **train** margins only.
   Buckets are low (margin ≤ t1), medium (t1 < margin ≤ t2), and high (margin > t2).
5. **Write outputs.** Split manifests go to `data/splits/{train,validation,test}.csv`, with
   `SHA256SUMS` and `prep_metadata.json` alongside. The per-row feature table goes to
   `data/processed/pairs_features.parquet`, which is git-ignored and can be regenerated.

### Pair selection

`scripts/select_pairs.py --mode {smoke,pilot,final}` (config `configs/data/selection.yaml`, logic in
`src/dpo_margin/selection.py`) draws 100, 500, or 4,000 pairs per condition:

- **Train split only.** It reads only `data/splits/train.csv`, after checking its SHA-256 and
  recomputing t1 and t2 from it. Tests show that selection still works when `validation.csv` and
  `test.csv` are unreadable.
- **Source matching.** Each source category (sharegpt, flan_v2_niv2, ...) gets a fixed quota from
  its train-pool share. The quotas are identical for all four conditions, so E1 is a
  source-stratified random sample and E2 to E4 have exactly the same source counts.
- **Same seed.** Every condition uses the same seed, with an independent random stream per
  (condition, source) cell.
- **No silent shrinking.** If any (condition, source) cell has too few eligible pairs, selection
  fails with a table of required and available counts.
- **No duplicate prompts.** One pair per prompt. E2, E3, and E4 are disjoint; E1 overlaps each by
  chance.
- **Reference sample.** An unmatched sample is also drawn, only to report how matching shifts the
  margin distribution.

Outputs:

- `data/manifests/<mode>/<condition>.jsonl`: row IDs, hashes, and metadata.
- `selection_metadata.json`: the SHA-256 of each manifest.
- A shared validation evaluation set, `data/manifests/eval/validation_n200_s0.jsonl`.
- `results/selection/<mode>/selected_pairs_summary.csv`.
- Figures in `figures/selection/<mode>/`.
- `reports/selection/<mode>/{selection_report,sample_examples}.md`.

The training code never reads the test split. At training time, every pair's SHA-256 is checked
against the raw dataset.

`data.to_dpo_example` converts each row to TRL's conversational preference format, with an
explicit prompt: `prompt` is the full shared context (system, user, and earlier assistant turns),
and `chosen`/`rejected` are the final assistant messages. It raises an error if chosen and rejected
do not share the same context. `scripts/verify_trl_format.py` checks this against TRL's own
`extract_prompt` and tokenization, and runs a loss on a real batch. Its output is
`reports/trl_format_check.md`.

### Exploratory data analysis

`scripts/eda.py` writes `results/eda/analysis_summary.json`, seven figures in `figures/eda/` (PNG
and PDF), and `reports/eda_report.md`. The report is rendered from the summary JSON, so every number
in it is computed rather than typed. Two findings matter for experiment design; details are in the
report:

- **Buckets are unequal in size.** Margins are discrete, and the train thresholds are t1 = 0.75 and
  t2 = 1.5. Each condition still trains on the same number of pairs.
- **Margin is confounded.** Margin is driven almost entirely by the rejected rating. It also
  co-varies with source mix, the identity of the rejected model, and response length. A difference
  between E2 and E4 therefore cannot be attributed to margin alone.

## Repository layout

```
configs/            YAML configs: base.yaml (template) + experiments/ + smoke/ + data/{prep,selection}.yaml
src/dpo_margin/     config, device detection, prepare.py, selection.py (+ diagnostics), data.py (TRL format), EDA, provenance
scripts/            check_environment, prepare_data, eda, select_pairs, verify_trl_format, train_dpo
tests/              pytest suite (prep logic on synthetic data, real-artifact checks, configs, TRL smoke)
data/splits/        train/validation/test manifests, dropped.csv, SHA256SUMS, prep_metadata.json (committed)
data/processed/     per-row feature table (git-ignored, regenerable)
data/manifests/     <mode>/<condition>.jsonl per mode (pilot/final committed, smoke git-ignored) + eval/
reports/            eda_report.md, example_inspection.md, trl_format_check.md, selection/<mode>/ (generated)
adapters/           LoRA adapters, adapters/<run_type>/<experiment>/seed<k>/ (git-ignored)
results/            per-run config, provenance, and training logs, results/<run_type>/... (smoke git-ignored)
figures/            eda/, selection/<mode>/, and plots generated from results/final
logs/               environment reports and scratch logs (git-ignored)
```

## Setup

Requires Python 3.11. The latest NumPy and SciPy releases need Python 3.12 or newer, so they are
pinned to the last versions that support 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py --json logs/environment.json
pytest -q
```

`pytest -q` downloads a tiny random-weight model (`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`)
for the trainer smoke test. Run `pytest -q -m "not network"` to skip it when offline.

### Pinned versions

| Package | Version | Package | Version |
|---|---|---|---|
| torch | 2.15.0 | numpy | 2.4.6 |
| transformers | 5.22.2 | pandas | 3.1.1 |
| trl | 1.17.1 | scipy | 1.17.1 |
| peft | 0.21.2 | matplotlib | 3.11.2 |
| accelerate | 1.15.0 | seaborn | 0.13.2 |
| datasets | 5.2.0 | pyyaml | 6.0.3 |
| pytest | 9.2.0 | bitsandbytes (Linux only) | 0.50.2 |

`requirements-lock-macos-arm64.txt` lists every installed package on the development Mac. On the
CUDA training machine, generate a matching lock file with `pip freeze --exclude-editable`.

### TRL 1.17 API notes (checked against the installed source)

- `DPOConfig` no longer accepts `max_prompt_length`, `max_completion_length`, `rpo_alpha`, or
  `reference_free`. Use `max_length` with `truncation_mode="keep_start"`.
- `loss_type` is a list, for example `["sigmoid"]`.
- `DPOTrainer` takes `processing_class`, `peft_config`, and `quantization_config`. When the model is
  passed as a Hub ID string, it is loaded with `DPOConfig.model_init_kwargs`.
- With PEFT, the reference model is the base model with the adapter disabled, so no second copy is
  loaded.
- In Transformers 5, `warmup_ratio` and `logging_dir` are removed. A float `warmup_steps` below 1
  is treated as a ratio.
- Data must be in the conversational format: `prompt` is a list of messages, and `chosen` and
  `rejected` are each a list holding the assistant message. The argilla rows repeat the user turn
  inside `chosen` and `rejected`, so `to_dpo_example` removes it.

`scripts/check_environment.py` checks that every key under `dpo:` and `training:` in every config
is a real `DPOConfig` field.

### Hardware support

| Backend | Use | Precision | 4-bit QLoRA |
|---|---|---|---|
| CUDA | final runs | bf16 if the GPU supports it (`dtype: auto`), otherwise fp32 | yes, with bitsandbytes on Linux |
| MPS (Apple Silicon) | tests and tiny smoke runs | fp32 by default | **no**, rejected with an error |
| CPU | tests and tiny smoke runs | fp32 | **no**, rejected with an error |

fp16 is not offered. Training fp16 with LoRA needs loss scaling and tends to be unstable without
any warning, while bf16 and fp32 avoid that.

## Reproducing

### 1. Prepare data, run the EDA, and select training pairs

This downloads about 143 MB. Rerunning these steps produces byte-identical outputs.

```bash
python scripts/prepare_data.py     # about 1 minute; tokenizes every pair with the Qwen tokenizer
python scripts/eda.py
python scripts/select_pairs.py --mode smoke    # 100 pairs per condition
python scripts/select_pairs.py --mode pilot    # 500
python scripts/select_pairs.py --mode final    # 4,000
python scripts/verify_trl_format.py            # TRL format check on a real batch
(cd data/splits && shasum -a 256 -c SHA256SUMS)
```

### 2. DPO training: `scripts/train_dpo.py`

```text
python scripts/train_dpo.py --condition {E1_random,E2_high_margin,E3_medium_margin,E4_low_margin}
       [--config configs/base.yaml] [--seed 42] [--max-steps N] [--smoke-test]
       [--run-type smoke|pilot|final] [--output-dir DIR] [--override key.sub=value ...]
       [--overwrite] [--allow-dirty] [--dry-run]
```

**Training setup.** The script runs TRL `DPOTrainer` with standard sigmoid DPO, using `DPOConfig`
from TRL 1.17.1. It trains a LoRA adapter (r 16, α 32, on every attention and MLP projection) on
`Qwen/Qwen2.5-1.5B-Instruct` at a pinned revision. The tokenizer comes from the same revision and
uses its official chat template.

**Optional 4-bit QLoRA.** Enable it with `--override quantization.load_in_4bit=true`. It is only
allowed on CUDA with bitsandbytes installed and a GPU that supports bf16.

**Initial hyperparameters.** These live in `configs/base.yaml`. They are exploratory, marked
`UNTUNED`, and are **not** taken from Ivison et al.:

| Setting | Value |
|---|---|
| learning rate | 5e-6 |
| β | 0.1 |
| epochs | 1 |
| `max_length` | 1024 |
| batch size | 1 × gradient accumulation 8 |
| seed | 42 |

They are judged only on the held-out validation subset (`eval_loss`, `eval_rewards/accuracies`,
`eval_rewards/margins`), never on test benchmarks.

**What every run checks and records** (in `run_metadata.json`):

- **Loss mask.** One real collated batch is checked: the tokens that enter the loss decode to
  exactly the final assistant reply plus `<|im_end|>\n`. Prompt tokens, earlier turns, the system
  prompt and padding are excluded. (TRL sets labels to -100 wherever `completion_mask` is 0.)
- **Reference policy.** With PEFT and no `ref_model`, TRL computes reference log-probs with the
  adapter disabled, so the reference is the frozen original checkpoint. The script records a
  SHA-256 of all base weights before and after training and fails if they changed.
- **Identical starting point.** Every run is a fresh process that loads the pinned checkpoint.
  TRL calls `set_seed(seed)` before LoRA initialisation, so for a given seed and dtype the initial
  LoRA fingerprint is identical across conditions. The script checks that the initial policy's
  log-probs equal the reference log-probs (LoRA B = 0).
- **Truncation and drops.** At `max_length` 1024, the script counts truncated pairs and pairs that
  TRL drops because the prompt alone is at least `max_length` tokens. Final runs **fail** if any
  pair would be dropped, because that would make N unequal across conditions.
- **Reload check.** The saved adapter is reloaded on a freshly loaded base model. LoRA weights must
  match exactly, and policy and reference log-probs on a probe batch must match.
- **Finite losses.** A non-finite loss stops the run. Exceptions and SIGTERM are recorded as
  `status: failed`.

**Outputs.** By default these go to `results/<run_type>/<condition>/seed<seed>/`:

- `adapter/`, which is git-ignored.
- `resolved_config.yaml`, `provenance.json` (git state and package versions) and
  `requirements_frozen.txt`.
- `history.csv`, with one row per logged step:
  - TRL metrics: training loss, `rewards/accuracies`, `rewards/chosen`/`rejected`/`margins`,
    log-probs, `grad_norm` and `learning_rate`;
  - the step number, wall-clock time, pairs seen and cumulative `num_tokens`;
  - memory: CUDA allocated and peak, or MPS allocated and driver memory, plus process peak RSS;
  - the evaluation versions of the same metrics on evaluation steps.
- `log_history.json`, `trainer_state.json`, `final_metrics.json` and `run_metadata.json`. The
  metadata holds the status, seeds, manifest SHA-256s, truncation statistics, the masking check,
  weight fingerprints, adapter file hashes, the reload check, timings and peak memory.

#### Order of runs

```bash
# 1. Smoke test: about 100 pairs, 12 optimizer steps of 8 pairs (cap 20), 16 validation pairs.
python scripts/select_pairs.py --mode smoke            # smoke manifests are git-ignored
python scripts/train_dpo.py --condition E1_random --smoke-test

# 2. Pilot: E1 and then E2 with an identical budget. 60 steps × 8 = 480 pairs, which stays under the
#    smallest per-condition count after TRL's max_length drops.
python scripts/train_dpo.py --condition E1_random      --run-type pilot --max-steps 60
python scripts/train_dpo.py --condition E2_high_margin --run-type pilot --max-steps 60

# 3. The same command runs the other conditions; no code changes are needed.
python scripts/train_dpo.py --condition E3_medium_margin --run-type pilot --max-steps 60
python scripts/train_dpo.py --condition E4_low_margin    --run-type pilot --max-steps 60

# 4. Final runs: 4,000 pairs, committed code, after tuning. With max_length 1024, final runs refuse
#    to start because TRL would drop 47–79 pairs per condition. Use dpo.max_length=2048 (the prep
#    limit) or decide explicitly.
for seed in 42 43 44; do for c in E1_random E2_high_margin E3_medium_margin E4_low_margin; do
  python scripts/train_dpo.py --condition $c --seed $seed --override dpo.max_length=2048
done; done
```

`configs/smoke/smoke_tiny.yaml` runs the same pipeline with a tiny random-weight model on any
backend; it is used by `tests/test_training.py`.

#### Remote NVIDIA GPU (Linux, CUDA 12)

```bash
git clone <your-remote> DRL_project && cd DRL_project      # or: rsync -a --exclude .venv ./ host:DRL_project/
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt && pip install -e .
pip freeze --exclude-editable > requirements-lock-linux-cuda.txt
python scripts/check_environment.py --json logs/environment.json    # must report backend=cuda, bf16 True
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python scripts/select_pairs.py --mode smoke
python scripts/train_dpo.py --condition E1_random --smoke-test      # check history.csv memory columns first
nohup python scripts/train_dpo.py --condition E1_random --run-type pilot --max-steps 60 > logs/pilot_E1.log 2>&1 &
```

On CUDA, `dtype: auto` selects bf16. A 1.5B bf16 LoRA run at `max_length` 1024 should fit on a
16–24 GB GPU (T4/L4/A10). Check the smoke run's `gpu_mem_peak_reserved_gb` before launching
longer runs. If memory is short, add `--override quantization.load_in_4bit=true`, but then use
QLoRA for **every** condition: the reference becomes the 4-bit base, so QLoRA and LoRA runs are not
comparable.

#### Google Colab (GPU runtime)

```python
!git clone <your-remote> DRL_project
%cd DRL_project
!pip install -q -r requirements.txt && pip install -q -e .
!python scripts/check_environment.py
!python scripts/select_pairs.py --mode smoke
!python scripts/train_dpo.py --condition E1_random --smoke-test
!python scripts/train_dpo.py --condition E1_random --run-type pilot --max-steps 60 \
    --output-dir /content/drive/MyDrive/dpo_runs/pilot/E1_random/seed42   # after drive.mount('/content/drive')
```

Colab's Python may be newer than 3.11; the pins install on 3.12. A T4 has no native bf16, so
`dtype: auto` falls back to fp32 there. Use an L4 or A100 runtime for bf16. Save `--output-dir` to
Drive so that runs survive disconnects.

`train_dpo.py` refuses final runs when the git tree has uncommitted changes, unless you pass
`--allow-dirty`, which is then recorded.

## Reproducibility rules

- All settings live in YAML. Changes made on the command line (`--seed`, `--override a.b=v`) are
  saved in `resolved_config.yaml`.
- There are two seeds. `seed` controls training (LoRA initialization and data order). `seed` in
  `configs/data/selection.yaml` fixes which pairs are selected, so every training seed sees the
  same data. `train_dpo.py` checks each manifest's SHA-256 and row count before training.
- The model is pinned to a Hub commit hash in `configs/base.yaml`. The dataset and tokenizer are
  pinned in `configs/data/prep.yaml`.
- Results are never written by hand. Figures and tables are generated only from `results/final/`.

## Status

Implemented and verified:

- The environment check passes on the development Mac (Apple Silicon, MPS backend).
- The unit tests pass.
- Data preparation and the EDA have been run on the pinned dataset revision. Rerunning them gives
  byte-identical outputs.
- Source-matched selection manifests for E1 to E4 are built in smoke (100), pilot (500), and final
  (4,000) modes, with no shortfall. The TRL format check passes on a real batch.
- `scripts/train_dpo.py` (LoRA DPO, optional CUDA QLoRA) runs end to end with the tiny test model,
  including the masking, reference, fingerprint, and adapter-reload checks.
- Real Qwen2.5-1.5B smoke test on a 16 GB Mac (bf16, MPS): the checks before training passed and 8
  of 12 optimizer steps completed with finite loss. The run was then stopped because the machine was
  out of memory, so it is **not** a completed smoke test.

Not yet done:

- Complete the real-model smoke test and the E1 and E2 pilots on a CUDA GPU (commands above).
- Explain why the real-model step-1 loss was 0.679 rather than ln 2 on bf16 MPS (the tiny model
  gives exactly ln 2).
- Hyperparameters marked `UNTUNED` in `configs/base.yaml` must be fixed before final runs.
- The downstream evaluation (benchmarks for E0 to E4) is not implemented yet.
- No final runs have been done, so this repository contains **no experimental results**.
