# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers import litedsa


@pytest.fixture(autouse=True)
def clear_litedsa_caches():
    litedsa.litedsa_available.cache_clear()
    with litedsa._OUTPUT_BUFS_LOCK:
        litedsa._OUTPUT_BUFS.clear()
    yield
    litedsa.litedsa_available.cache_clear()
    with litedsa._OUTPUT_BUFS_LOCK:
        litedsa._OUTPUT_BUFS.clear()


def test_union_seq_space_rounding_and_fixed_ab_mode(monkeypatch):
    monkeypatch.setattr(litedsa, "_DYNAMIC_SEQ_SPACE", True)
    assert litedsa._union_seq_space(1) == 1024
    assert litedsa._union_seq_space(196609) == 197632
    assert litedsa._union_seq_space(1 << 20) == 1 << 20
    with pytest.raises(ValueError):
        litedsa._union_seq_space(0)
    with pytest.raises(ValueError):
        litedsa._union_seq_space((1 << 20) + 1)

    monkeypatch.setattr(litedsa, "_DYNAMIC_SEQ_SPACE", False)
    assert litedsa._union_seq_space(196609) == 1 << 20


def test_output_buffers_reuse_and_grow_on_cpu():
    device = torch.device("cpu")
    out1, max1, lse1 = litedsa._get_or_create_output_buffers(
        device, stream_handle=11, owner_thread=22, ng=3
    )
    assert out1.shape == (3, 128, 512)
    assert max1.shape == lse1.shape == (3, 128)

    out2, max2, lse2 = litedsa._get_or_create_output_buffers(
        device, stream_handle=11, owner_thread=22, ng=2
    )
    assert out2.data_ptr() == out1.data_ptr()
    assert max2.data_ptr() == max1.data_ptr()
    assert lse2.data_ptr() == lse1.data_ptr()

    out3, max3, lse3 = litedsa._get_or_create_output_buffers(
        device, stream_handle=11, owner_thread=22, ng=5
    )
    assert out3.shape == (5, 128, 512)
    assert max3.shape == lse3.shape == (5, 128)
    assert out3.data_ptr() != out1.data_ptr()
    key = litedsa._output_buffer_key(device, 11, 22)
    assert litedsa._OUTPUT_BUFS[key][0] == 8


def test_output_buffer_pool_is_lru_bounded(monkeypatch):
    monkeypatch.setattr(litedsa, "_OUTPUT_BUF_CACHE_MAX_POOLS", 2)
    device = torch.device("cpu")
    litedsa._get_or_create_output_buffers(device, 1, 7, 1)
    litedsa._get_or_create_output_buffers(device, 2, 7, 1)
    litedsa._get_or_create_output_buffers(device, 1, 7, 1)
    litedsa._get_or_create_output_buffers(device, 3, 7, 1)

    assert len(litedsa._OUTPUT_BUFS) == 2
    assert litedsa._output_buffer_key(device, 1, 7) in litedsa._OUTPUT_BUFS
    assert litedsa._output_buffer_key(device, 2, 7) not in litedsa._OUTPUT_BUFS
    assert litedsa._output_buffer_key(device, 3, 7) in litedsa._OUTPUT_BUFS


def test_reuse_is_disabled_for_autograd_and_capture(monkeypatch):
    monkeypatch.setattr(litedsa, "_REUSE_OUTPUT_BUFS", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.enable_grad():
        assert not litedsa._reuse_output_buffers_allowed()
    with torch.no_grad():
        assert litedsa._reuse_output_buffers_allowed()

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.no_grad(), pytest.raises(RuntimeError, match="capture-safe"):
        litedsa._reuse_output_buffers_allowed()


def test_tvm_ffi_override_counts_as_available(monkeypatch):
    module = SimpleNamespace(union_qm=object(), masked_mla_fp8=object())
    monkeypatch.setattr(litedsa, "_LITEDSA_SO", "/tmp/litedsa.so")
    monkeypatch.setattr(litedsa, "_litedsa_mod", lambda: module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))

    assert litedsa.litedsa_available()


def test_dsa_mode_is_explicit_and_validated(monkeypatch):
    get_mode = envs.environment_variables["VLLM_DSA_MODE"]
    monkeypatch.delenv("VLLM_DSA_MODE", raising=False)
    assert get_mode() == "raw"

    monkeypatch.setenv("VLLM_DSA_MODE", "LiTeDsA")
    assert get_mode() == "litedsa"

    monkeypatch.setenv("VLLM_DSA_MODE", "litedas")
    with pytest.raises(ValueError, match="Valid options"):
        get_mode()


def test_litedsa_override_envs_are_registered():
    expected = {
        "VLLM_LITEDSA_SO",
        "VLLM_LITEDSA_UNION_SO",
        "VLLM_LITEDSA_DYNAMIC_SPAN",
        "VLLM_LITEDSA_UNION_STATS",
        "VLLM_LITEDSA_REUSE_OUTPUT_BUFS",
        "VLLM_LITETOPK_FP4_PRODUCTION_MIN_S",
    }
    assert expected <= envs.environment_variables.keys()
