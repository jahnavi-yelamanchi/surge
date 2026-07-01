"""Unit tests for the metrics math (no GPU/server needed)."""

from bench.metrics import RequestResult, _percentile, summarize


def test_percentile_basic():
    values = [float(i) for i in range(1, 101)]  # 1..100
    assert _percentile(values, 50) == 50.5
    assert _percentile(values, 99) == round(_percentile(values, 99), 6)
    # p99 of 1..100 (linear interp over ranks 0..99) = 99.01
    assert abs(_percentile(values, 99) - 99.01) < 1e-9
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 100) == 100.0


def test_percentile_edge_cases():
    assert _percentile([], 50) != _percentile([], 50)  # NaN != NaN
    assert _percentile([42.0], 99) == 42.0


def test_summarize_throughput_and_latency():
    results = [
        RequestResult(
            success=True,
            prompt_tokens=10,
            output_tokens=4,
            ttft=0.1,
            e2e=0.4,
            itls=[0.1, 0.1, 0.1],
        ),
        RequestResult(
            success=True,
            prompt_tokens=10,
            output_tokens=4,
            ttft=0.2,
            e2e=0.5,
            itls=[0.1, 0.1, 0.1],
        ),
        RequestResult(success=False, error="boom"),
    ]
    s = summarize(results, duration=2.0)

    assert s["num_requests"] == 3
    assert s["num_success"] == 2
    assert s["num_errors"] == 1
    assert s["total_output_tokens"] == 8
    # 2 successful requests over 2s
    assert s["request_throughput"] == 1.0
    # 8 output tokens over 2s
    assert s["output_token_throughput"] == 4.0
    # TTFT mean is (100ms + 200ms) / 2 = 150ms
    assert abs(s["ttft_ms"]["mean"] - 150.0) < 1e-6
    # every inter-token gap is 100ms
    assert abs(s["itl_ms"]["p99"] - 100.0) < 1e-6


def test_summarize_all_failed():
    s = summarize([RequestResult(success=False)], duration=1.0)
    assert s["num_success"] == 0
    assert s["request_throughput"] == 0.0
    assert s["itl_ms"]["p50"] != s["itl_ms"]["p50"]  # NaN, no data
