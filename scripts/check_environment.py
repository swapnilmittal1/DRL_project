#!/usr/bin/env python
"""Print the software/hardware environment and check TRL API compatibility with our configs.

Usage: python scripts/check_environment.py [--json logs/environment.json]
Exit code is non-zero if a hard requirement fails (wrong Python, missing package, config/API mismatch).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpo_margin import provenance  # noqa: E402

REQUIRED = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy", "pandas",
            "scipy", "matplotlib", "seaborn", "pyyaml", "pytest")


def check_trl_api(config_paths: list[Path]) -> list[str]:
    """Every key under `dpo:` and `training:` in our configs must be a DPOConfig field."""
    from trl import DPOConfig

    from dpo_margin.config import load_config

    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    problems = []
    for path in config_paths:
        cfg = load_config(path)
        for section in ("dpo", "training"):
            for key in cfg.get(section, {}):
                if key not in fields:
                    problems.append(f"{path.relative_to(ROOT)}: {section}.{key} is not a DPOConfig field")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="also write the report as JSON to this path")
    args = parser.parse_args()

    failures: list[str] = []
    report: dict = {}

    py = sys.version_info
    report["python"] = sys.version.split()[0]
    report["python_executable"] = sys.executable
    print(f"Python            : {report['python']} ({sys.executable})")
    if (py.major, py.minor) != (3, 11):
        failures.append(f"Python 3.11 required, found {py.major}.{py.minor}")

    versions = provenance.package_versions()
    report["packages"] = versions
    print("\nPackages")
    for name, ver in versions.items():
        print(f"  {name:<13}: {ver or 'not installed'}")
        if name in REQUIRED and ver is None:
            failures.append(f"missing package: {name}")

    import torch

    from dpo_margin.device import bitsandbytes_available, detect_device

    info = detect_device()
    report["device"] = info.to_dict()
    report["torch_cuda_build"] = torch.version.cuda
    report["mps_built"] = torch.backends.mps.is_built()
    print("\nHardware")
    print(f"  backend        : {info.backend}")
    print(f"  device name    : {info.name}")
    print(f"  device count   : {info.device_count}")
    vram = f"{info.total_memory_gb} GB" if info.total_memory_gb is not None else "n/a (no dedicated VRAM)"
    print(f"  VRAM           : {vram}")
    import psutil

    report["system_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    print(f"  system RAM     : {report['system_ram_gb']} GB (shared with the GPU on MPS)")
    print(f"  torch CUDA build: {torch.version.cuda or 'none (CPU/MPS build)'}")
    print(f"  MPS built      : {torch.backends.mps.is_built()}")

    print("\nNumerical precision (probed with a matmul + backward on the device)")
    print(f"  float32        : True")
    print(f"  bfloat16       : {info.bf16}")
    print(f"  float16        : {info.fp16}")
    print(f"  TF32 (Ampere+) : {info.tf32}")

    bnb = bitsandbytes_available()
    report["bitsandbytes_installed"] = bnb
    qlora_ok = info.backend == "cuda" and bnb
    print(f"\n4-bit QLoRA (bitsandbytes): {'available' if qlora_ok else 'NOT available'}"
          f" (backend={info.backend}, bitsandbytes installed={bnb})")

    # base.yaml is a template (no condition); every run config inherits it.
    configs = sorted(p for p in (ROOT / "configs").rglob("*.yaml")
                     if "data" not in p.relative_to(ROOT).parts[1:-1] and p.name != "base.yaml")
    problems = check_trl_api(configs)
    report["trl_api_problems"] = problems
    print(f"\nTRL API check     : {len(configs)} config files, {len(problems)} problem(s)")
    for p in problems:
        print(f"  - {p}")
    failures.extend(problems)

    print("\nTraining readiness")
    if info.backend == "cuda":
        print("  CUDA available: final runs can use this machine.")
    elif info.backend == "mps":
        print("  MPS only: suitable for unit tests and tiny smoke tests; final runs need a CUDA GPU.")
    else:
        print("  CPU only: suitable for unit tests and tiny smoke tests; final runs need a CUDA GPU.")

    report["failures"] = failures
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")

    print("\nRESULT:", "OK" if not failures else "FAILED")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
