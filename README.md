# Does preference margin matter for DPO data selection?

This is a course research project building on Ivison et al., *Unpacking DPO and PPO: Disentangling
Best Practices for Learning from Preference Feedback* (NeurIPS 2024). One of their main takeaways
is that the quality of the preference data matters a lot. We want to look at one narrow piece of
that: if you can only afford a fixed number of preference pairs, does it help to pick them by how
far apart the two responses were rated?

We use the UltraFeedback ratings from
[`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
and define the margin of a pair as `chosen rating - rejected rating`. Ratings are averages on a
1–5 scale in steps of 0.25. The policy is `Qwen/Qwen2.5-1.5B-Instruct`, trained with DPO and a
LoRA adapter.

## Conditions

| ID | Config | Training data |
|----|--------|---------------|
| E0_no_dpo | `configs/experiments/e0_no_dpo.yaml` | none, the base Qwen model as released |
| E1_random | `configs/experiments/e1_random.yaml` | random positive-margin pairs |
| E2_high_margin | `configs/experiments/e2_high_margin.yaml` | top margin tercile (margin > 1.5) |
| E3_medium_margin | `configs/experiments/e3_medium_margin.yaml` | middle tercile (0.75 < margin ≤ 1.5) |
| E4_low_margin | `configs/experiments/e4_low_margin.yaml` | lowest positive tercile (0 < margin ≤ 0.75) |

E1–E4 each get 4,000 pairs from the training split. The only thing that changes between them is
`data.condition` (there's a test that checks this). The loss is plain sigmoid DPO with β = 0.1 in
every case, since what we're varying is the data, not the objective.

## How the data is built

### Cleaning and splitting

`scripts/prepare_data.py` (settings in `configs/data/prep.yaml`, code in
`src/dpo_margin/prepare.py`) does the following:

1. Checks the schema and the `[user, assistant]` structure of every row, and dumps 12 random
   examples to `reports/example_inspection.md` so we could eyeball them.
2. Drops broken rows: bad structure or ratings, empty prompts or responses, identical chosen and
   rejected responses, margin ≤ 0, pairs over 2,048 tokens under the Qwen chat template, and
   duplicate prompts (keeping the first). Every dropped row and its reason ends up in
   `data/splits/dropped.csv`.
3. Splits by prompt into roughly 90/5/5 train/validation/test, using a salted SHA-256 of the
   normalized prompt, so the same prompt can't land in two splits.
4. Computes the bucket thresholds (the 1/3 and 2/3 quantiles of the margin) from the training
   split only. They come out to t1 = 0.75 and t2 = 1.5.
5. Writes `data/splits/{train,validation,test}.csv` along with `SHA256SUMS` and
   `prep_metadata.json`. A per-row feature table goes to `data/processed/`. That folder isn't
   committed, but the script regenerates it.

### Picking the pairs

`scripts/select_pairs.py --mode {smoke,pilot,final}` draws 100, 500 or 4,000 pairs per condition.
The settings are in `configs/data/selection.yaml` and the code is in `src/dpo_margin/selection.py`.
A few choices worth knowing about:

- It only ever reads `train.csv`, and it checks the file's hash first. One of the tests replaces
  the validation and test files with garbage to make sure selection doesn't touch them.
- We match on source (sharegpt, flan_v2_niv2, evol_instruct, ...). Each source gets the same quota
  in every condition, based on its share of the training pool. Without this, the margin buckets
  would also differ a lot in source mix.
- If some bucket/source combination doesn't have enough pairs, the script stops and prints the
  shortfall instead of quietly handing back fewer pairs.
- Each condition has at most one pair per prompt. E2–E4 never overlap; E1 overlaps each of them a
  little by chance.
- It also draws an unmatched sample, but only so we can see how much source matching shifts the
  margin distribution. It turns out to be very little.

The pair lists end up in `data/manifests/<mode>/`. The summary table is in `results/selection/`,
the plots in `figures/selection/`, and a short report and a few sample pairs in
`reports/selection/`. A fixed set of 200 validation pairs
(`data/manifests/eval/validation_n200_s0.jsonl`) is used for evaluation in every run.

For training, each row is converted to TRL's conversational format. `prompt` holds the whole
conversation up to the last user turn, and `chosen`/`rejected` hold the two final assistant
replies. The conversion refuses rows where chosen and rejected don't share the same context.
`scripts/verify_trl_format.py` compares the result against TRL's own `extract_prompt` and
tokenization on a real batch and writes `reports/trl_format_check.md`.

### What the data looks like

`scripts/eda.py` produces seven figures in `figures/eda/`, `results/eda/analysis_summary.json`, and
`reports/eda_report.md`. Two things stood out:

- The buckets aren't the same size, because the ratings are discrete. That's fine for the
  experiment, since every condition still trains on 4,000 pairs.
- Margin is tangled up with other things. It's driven mostly by how bad the rejected response is,
  and it also moves with the source, the model that wrote the rejected response, and response
  length. So if E2 beats E4, we can't put that down to margin alone.

## Repository layout

```
configs/         base.yaml (shared settings), experiments/, smoke/, data/{prep,selection}.yaml
src/dpo_margin/  data prep, pair selection, TRL conversion, training helpers, EDA
scripts/         check_environment, prepare_data, eda, select_pairs, verify_trl_format, train_dpo
tests/           pytest suite
data/splits/     cleaned train/validation/test splits + checksums
data/manifests/  selected pairs per mode and condition, plus the eval set
reports/         EDA and selection reports
results/         summary tables and per-run training records
figures/         EDA and selection plots (PNG + PDF)
logs/            environment report and run logs
```

## Setup

You need Python 3.11. NumPy and SciPy are pinned to the last releases that still support 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py --json logs/environment.json
pytest -q
```

Some tests download a tiny random-weight model (`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`).
Use `pytest -q -m "not network"` if you're offline.

Main versions: torch 2.15.0, transformers 5.22.2, trl 1.17.1, peft 0.21.2, accelerate 1.15.0,
datasets 5.2.0, and bitsandbytes 0.50.2 (Linux only). `requirements-lock-macos-arm64.txt` has the
full environment from the Mac we developed on.

### Notes on TRL 1.17

A few things changed in recent TRL and Transformers releases, in case you're used to older
examples:

- `DPOConfig` no longer has `max_prompt_length`, `max_completion_length`, `rpo_alpha`, or
  `reference_free`. There's just `max_length` with `truncation_mode="keep_start"`.
- `loss_type` is a list now, e.g. `["sigmoid"]`.
- `warmup_ratio` and `logging_dir` are gone in Transformers 5. A float `warmup_steps` below 1 is
  treated as a ratio.
- With PEFT and no `ref_model`, the reference is just the base model with the adapter switched off,
  so there's no second copy in memory.

`check_environment.py` checks every `dpo:` and `training:` key in our configs against the installed
`DPOConfig`.

### Hardware

Real training is meant for a CUDA GPU, where `dtype: auto` picks bf16 if the card supports it
natively and fp32 otherwise. 4-bit QLoRA only works there, with bitsandbytes. A Mac (MPS) or CPU
is fine for the tests and for the tiny-model pipeline check. We don't use fp16, since fp16 with LoRA
needs loss scaling and is easy to get subtly wrong.

## Running things

### Data

Downloads about 143 MB. Running it twice gives byte-identical files.

```bash
python scripts/prepare_data.py      # ~1 minute, tokenizes everything with the Qwen tokenizer
python scripts/eda.py
python scripts/select_pairs.py --mode smoke     # 100 pairs per condition
python scripts/select_pairs.py --mode pilot     # 500
python scripts/select_pairs.py --mode final     # 4,000
python scripts/verify_trl_format.py
(cd data/splits && shasum -a 256 -c SHA256SUMS)
```

### Training

```text
python scripts/train_dpo.py --condition {E1_random,E2_high_margin,E3_medium_margin,E4_low_margin}
       [--config configs/base.yaml] [--seed 42] [--max-steps N] [--smoke-test]
       [--run-type smoke|pilot|final] [--output-dir DIR] [--override key.sub=value ...]
       [--overwrite] [--allow-dirty] [--dry-run]
```

This trains a LoRA adapter (r 16, α 32, on all attention and MLP projections) on the pinned Qwen
checkpoint with TRL's `DPOTrainer`, using the model's own chat template. To use QLoRA, add
`--override quantization.load_in_4bit=true`. That only works on CUDA with bitsandbytes and a bf16
GPU.

The starting hyperparameters are our own guesses, not values from the paper. They're marked
`UNTUNED` in `configs/base.yaml`:

| learning rate | β | epochs | max_length | batch | seed |
|---|---|---|---|---|---|
| 5e-6 | 0.1 | 1 | 1024 | 1 × 8 grad. accumulation | 42 |

We tune them only on the validation pairs (eval loss, reward accuracy and reward margin), never on
the test split or downstream benchmarks.

Each run also checks a few things that would otherwise be easy to get silently wrong, and writes
the results to `run_metadata.json`:

- **Loss mask.** It checks one real batch to confirm the loss only sees the final assistant reply
  (plus `<|im_end|>`), not the prompt, the system message, earlier turns, or padding.
- **Reference model.** The reference is the original checkpoint with the adapter disabled. The
  script hashes the base weights before and after training and fails if they changed.
- **Starting point.** Every run loads the pinned checkpoint fresh and seeds LoRA the same way, so
  for a given seed all four conditions start from identical weights. Before training, the policy's
  log-probs are compared against the reference's, and they should match.
- **Truncation.** At `max_length` 1024 some pairs get cut off, and TRL silently drops pairs whose
  prompt alone is too long. Both are counted. Final runs refuse to start if any pair would be
  dropped, because that would make the conditions unequal.
- **Reloading.** At the end, the saved adapter is loaded back onto a fresh copy of the base model
  and has to give the same log-probs.

A NaN loss stops the run, and crashes or kills are recorded as `status: failed`.

By default everything goes to `results/<run_type>/<condition>/seed<seed>/`:

- the adapter;
- the resolved config, package versions and git state;
- `history.csv`, with one row per step: loss, reward accuracy and margins, learning rate, tokens
  seen, time and memory;
- the final metrics and the run metadata.

The adapter weights aren't committed.

The plan, in order:

```bash
# 1. smoke test: ~100 pairs, 12 optimizer steps
python scripts/train_dpo.py --condition E1_random --smoke-test

# 2. pilot: E1, then E2 on the same budget (60 steps x 8 = 480 pairs)
python scripts/train_dpo.py --condition E1_random      --run-type pilot --max-steps 60
python scripts/train_dpo.py --condition E2_high_margin --run-type pilot --max-steps 60

# 3. E3 and E4 are the same command with a different --condition
python scripts/train_dpo.py --condition E3_medium_margin --run-type pilot --max-steps 60
python scripts/train_dpo.py --condition E4_low_margin    --run-type pilot --max-steps 60

# 4. final runs (4,000 pairs, after tuning, from a committed tree)
for seed in 42 43 44; do for c in E1_random E2_high_margin E3_medium_margin E4_low_margin; do
  python scripts/train_dpo.py --condition $c --seed $seed --override dpo.max_length=2048
done; done
```

The final runs use `max_length` 2048 because at 1024, TRL would drop 47–79 of the 4,000 pairs
depending on the condition, and the script won't allow that. `configs/smoke/smoke_tiny.yaml` runs
the whole pipeline on the tiny test model, which is handy for checking changes quickly on a laptop.

### On a remote GPU

```bash
git clone https://github.com/swapnilmittal1/DRL_project.git && cd DRL_project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt && pip install -e .
python scripts/check_environment.py --json logs/environment.json   # should say backend=cuda
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python scripts/select_pairs.py --mode smoke
python scripts/train_dpo.py --condition E1_random --smoke-test     # look at the memory columns first
nohup python scripts/train_dpo.py --condition E1_random --run-type pilot --max-steps 60 > logs/pilot_E1.log 2>&1 &
```

A 1.5B model in bf16 with LoRA at length 1024 should fit on a 16–24 GB card (L4, A10). Check
`gpu_mem_peak_reserved_gb` in the smoke run's history before starting anything long. If memory is
tight, QLoRA is an option, but then use it for every condition: the reference model becomes the
4-bit base, so QLoRA and LoRA runs can't be compared.

### On Colab

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/swapnilmittal1/DRL_project.git
%cd DRL_project
!pip install -q -r requirements.txt && pip install -q -e .
!python scripts/check_environment.py
!python scripts/select_pairs.py --mode smoke
!python scripts/train_dpo.py --condition E1_random --smoke-test
!python scripts/train_dpo.py --condition E1_random --run-type pilot --max-steps 60 \
    --output-dir /content/drive/MyDrive/dpo_runs/pilot/E1_random/seed42
```

A T4 doesn't do bf16 natively, so it falls back to fp32 there. An L4 or A100 runtime is a better
fit. Writing `--output-dir` to Drive means a disconnect doesn't cost you the run.

## Reproducibility

Everything is configured in YAML. Anything changed from the command line is saved in
`resolved_config.yaml` for that run. There are two seeds:

- the training seed, which sets the LoRA init and data order;
- the selection seed in `configs/data/selection.yaml`, which fixes which pairs each condition gets.

The model, dataset and tokenizer are all pinned to specific Hub revisions. The training script
checks the hash and size of each pair list before it starts. Final runs also need a clean git tree,
unless you pass `--allow-dirty`, which gets recorded.

## Where things stand

Done so far:

- the environment check and the test suite pass on the development Mac;
- data prep, EDA and pair selection for all three sizes, with no shortfalls;
- the TRL format check on a real batch;
- the training script runs end to end on the tiny test model, including all the checks above.

We also tried the real 1.5B model on the Mac (16 GB, bf16 on MPS). The pre-training checks passed
and 8 of 12 smoke steps ran with finite loss, but the machine ran out of memory and we stopped it.
So that doesn't count as a finished smoke test.

Still to do:

- finish the smoke test and the E1/E2 pilots on a GPU;
- figure out why the first-step loss on the real model was 0.679 instead of ln 2 ≈ 0.693 (the tiny
  model gives exactly ln 2, so it may just be bf16 noise on MPS);
- tune the `UNTUNED` hyperparameters;
- write the downstream evaluation for E0–E4.

There are no experimental results in this repo yet.
