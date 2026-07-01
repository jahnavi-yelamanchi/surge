"""Latency/throughput metrics for the Surge benchmark.

A `RequestResult` captures the timing of one streamed completion. `summarize()`
turns a batch of them (plus the wall-clock duration of the load phase) into the
summary we report: request/token throughput and the p50/p95/p99 of three
latencies that matter for LLM serving:

- TTFT  (time to first token)      — how long until the user sees anything
- ITL   (inter-token latency)      — the gap between successive output tokens;
                                      its p99 is the "100ms p99" SLO we hold fixed
- E2E   (end-to-end latency)       — full request wall time
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    """Timing for a single streamed request. Times are in seconds."""

    success: bool = False
    error: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft: float = 0.0  # arrival time of the first output token, relative to send
    e2e: float = 0.0  # total wall time of the request
    itls: list[float] = field(default_factory=list)  # gaps between output tokens


def _percentile(values: list[float], p: float) -> float:
    """p-th percentile (p in [0, 100]) via linear interpolation; NaN if empty."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def _pct_block(values: list[float], scale: float = 1.0) -> dict[str, float]:
    """{mean,p50,p95,p99} for a series, multiplied by `scale` (e.g. 1000 for ms)."""
    return {
        "mean": (statistics.fmean(values) * scale) if values else float("nan"),
        "p50": _percentile(values, 50) * scale,
        "p95": _percentile(values, 95) * scale,
        "p99": _percentile(values, 99) * scale,
    }


def summarize(results: list[RequestResult], duration: float) -> dict:
    """Aggregate per-request results collected over `duration` seconds.

    Latency blocks are reported in milliseconds. Throughput uses the wall-clock
    duration of the load phase so it reflects sustained serving rate.
    """
    ok = [r for r in results if r.success]
    n_ok = len(ok)
    n_err = len(results) - n_ok

    total_output_tokens = sum(r.output_tokens for r in ok)
    total_prompt_tokens = sum(r.prompt_tokens for r in ok)
    all_itls = [gap for r in ok for gap in r.itls]

    return {
        "num_requests": len(results),
        "num_success": n_ok,
        "num_errors": n_err,
        "duration_s": duration,
        "request_throughput": (n_ok / duration) if duration > 0 else float("nan"),
        "output_token_throughput": (
            (total_output_tokens / duration) if duration > 0 else float("nan")
        ),
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "ttft_ms": _pct_block([r.ttft for r in ok], scale=1000.0),
        "itl_ms": _pct_block(all_itls, scale=1000.0),
        "e2e_ms": _pct_block([r.e2e for r in ok], scale=1000.0),
    }
