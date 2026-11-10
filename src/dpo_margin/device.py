"""Backend detection (CUDA / MPS / CPU) and precision resolution."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

import torch


@dataclass
class DeviceInfo:
    backend: str  # "cuda" | "mps" | "cpu"
    name: str
    total_memory_gb: float | None
    device_count: int
    bf16: bool
    fp16: bool
    tf32: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _probe(device: str, dtype: torch.dtype) -> bool:
    """Run a tiny matmul + backward to check the dtype actually works on this device."""
    try:
        a = torch.randn(8, 8, device=device, dtype=dtype, requires_grad=True)
        (a @ a).sum().backward()
        return bool(torch.isfinite(a.grad).all().item())
    except Exception:
        return False


def detect_device() -> DeviceInfo:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            backend="cuda",
            name=props.name,
            total_memory_gb=round(props.total_memory / 1024**3, 2),
            device_count=torch.cuda.device_count(),
            bf16=torch.cuda.is_bf16_supported(including_emulation=False),  # T4/V100: emulated only -> fp32
            fp16=_probe("cuda", torch.float16),
            tf32=props.major >= 8,
        )
    if torch.backends.mps.is_available():
        # MPS shares unified system memory; there is no dedicated VRAM figure.
        return DeviceInfo(
            backend="mps",
            name="Apple Silicon GPU (MPS)",
            total_memory_gb=None,
            device_count=1,
            bf16=_probe("mps", torch.bfloat16),
            fp16=_probe("mps", torch.float16),
            tf32=False,
        )
    return DeviceInfo(
        backend="cpu",
        name="CPU",
        total_memory_gb=None,
        device_count=0,
        bf16=_probe("cpu", torch.bfloat16),
        fp16=_probe("cpu", torch.float16),
        tf32=False,
    )


def resolve_dtype(requested: str, info: DeviceInfo) -> str:
    """Pick the model/training dtype. `auto` uses bf16 only on CUDA GPUs that support it.

    fp16 is intentionally not offered: fp16 + LoRA needs loss scaling and is a common
    source of silent instability; bf16 or fp32 avoids that.
    """
    if requested == "auto":
        return "bfloat16" if info.backend == "cuda" and info.bf16 else "float32"
    if requested == "bfloat16" and not info.bf16:
        raise RuntimeError(f"bfloat16 requested but not supported on backend {info.backend!r}")
    return requested


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def check_quantization(load_in_4bit: bool, info: DeviceInfo) -> None:
    if not load_in_4bit:
        return
    if info.backend != "cuda":
        raise RuntimeError(
            f"4-bit quantization (bitsandbytes QLoRA) is only supported on CUDA in this project; "
            f"detected backend {info.backend!r}. Set quantization.load_in_4bit: false."
        )
    if not bitsandbytes_available():
        raise RuntimeError("quantization.load_in_4bit is true but bitsandbytes is not installed.")
