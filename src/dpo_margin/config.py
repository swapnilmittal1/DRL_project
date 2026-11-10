"""YAML config loading with single-parent inheritance and dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONDITIONS = ("E1_random", "E2_high_margin", "E3_medium_margin", "E4_low_margin")
RUN_TYPES = ("smoke", "pilot", "final")
DTYPES = ("auto", "bfloat16", "float32")

REQUIRED_KEYS = (
    "experiment_id",
    "run_type",
    "seed",
    "train",
    "model.name_or_path",
    "model.revision",
    "data.prep_config",
    "data.selection_config",
    "data.condition",
)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_with_inheritance(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config inheritance cycle: {' -> '.join(map(str, seen + (path,)))}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is None:
        return cfg
    base = _load_with_inheritance((path.parent / parent), seen + (path,))
    return deep_merge(base, cfg)


def get(cfg: dict, dotted: str) -> Any:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `a.b.c=value` overrides; values are parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    return cfg


def _parse_scalar(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML 1.1 reads "1e-5" (no dot) as a string.
        try:
            return float(value)
        except ValueError:
            pass
    return value


def validate(cfg: dict) -> None:
    for key in REQUIRED_KEYS:
        get(cfg, key)
    if cfg["run_type"] not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}")
    if cfg["train"] and cfg["data"]["condition"] not in CONDITIONS:
        raise ValueError(f"data.condition must be one of {CONDITIONS} when train is true")
    if cfg.get("model", {}).get("dtype", "auto") not in DTYPES:
        raise ValueError(f"model.dtype must be one of {DTYPES}")
    if not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an integer")


def load_config(path: str | Path, overrides: list[str] | None = None, smoke_test: bool = False) -> dict:
    """Load YAML (with inheritance), then the `smoke_test:` overlay if requested, then dotted overrides."""
    cfg = _load_with_inheritance(Path(path))
    if smoke_test:
        cfg = deep_merge(cfg, cfg.get("smoke_test") or {})
    cfg = apply_overrides(cfg, overrides or [])
    validate(cfg)
    return cfg
