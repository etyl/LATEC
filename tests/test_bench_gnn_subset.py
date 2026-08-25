import argparse

import pytest

from scripts.bench_gnn_subset import GpuSampler, HostMemorySampler, parse_chunk_size


def test_parse_chunk_size_accepts_scalar_and_per_axis_edges():
    assert parse_chunk_size("auto") is None
    assert parse_chunk_size("32") == 32
    assert parse_chunk_size("64,32,32,32") == (64, 32, 32, 32)


@pytest.mark.parametrize("value", ["-1", "32,bad"])
def test_parse_chunk_size_rejects_invalid_edges(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_chunk_size(value)


def test_host_memory_sampler_reports_scoped_peak_and_increase(monkeypatch):
    samples = iter((100.0, 125.0))
    sampler = HostMemorySampler(interval=3600)
    monkeypatch.setattr(sampler, "_rss_mib", lambda: next(samples))

    with sampler:
        pass

    assert sampler.baseline_mib == 100.0
    assert sampler.peak_mib == 125.0
    assert sampler.increase_mib == 25.0


def test_gpu_sampler_reports_scoped_device_memory_increase():
    sampler = GpuSampler.__new__(GpuSampler)
    sampler.util = [10.0, 20.0]
    sampler.mem = [500.0, 650.0]
    sampler.baseline_mem_mib = 500.0
    stats = sampler.summary()

    assert stats["gpu_mem_peak_mib"] == 650.0
    assert stats["gpu_mem_increase_mib"] == 150.0
