import pytest

from dpo_margin.device import DeviceInfo, check_quantization, detect_device, resolve_dtype


def _info(backend, bf16=True):
    return DeviceInfo(backend=backend, name="x", total_memory_gb=None, device_count=1, bf16=bf16, fp16=True, tf32=False)


def test_detect_device_runs():
    assert detect_device().backend in {"cuda", "mps", "cpu"}


def test_auto_dtype_is_bf16_only_on_cuda():
    assert resolve_dtype("auto", _info("cuda")) == "bfloat16"
    assert resolve_dtype("auto", _info("cuda", bf16=False)) == "float32"
    assert resolve_dtype("auto", _info("mps")) == "float32"
    assert resolve_dtype("auto", _info("cpu")) == "float32"


def test_bf16_request_fails_when_unsupported():
    with pytest.raises(RuntimeError):
        resolve_dtype("bfloat16", _info("cpu", bf16=False))


@pytest.mark.parametrize("backend", ["mps", "cpu"])
def test_4bit_rejected_off_cuda(backend):
    with pytest.raises(RuntimeError, match="only supported on CUDA"):
        check_quantization(True, _info(backend))


def test_no_quantization_always_ok():
    check_quantization(False, _info("cpu"))
