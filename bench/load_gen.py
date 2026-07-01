"""Async load generator for the Surge benchmark.

Sends streaming completion requests to an OpenAI-compatible vLLM server and
records the timing needed by `metrics.RequestResult` (TTFT, inter-token gaps,
end-to-end latency). Requests arrive on a Poisson process at a target QPS, which
mimics real, jittery traffic far better than a fixed cadence.

Run standalone against a live server:

    python -m bench.load_gen --base-url http://127.0.0.1:8000 \
        --model NousResearch/Meta-Llama-3-8B-Instruct \
        --qps 4 --duration 30 --input-tokens 512 --output-tokens 128

The M3 runner imports `generate_load` directly rather than shelling out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx

from bench.metrics import RequestResult, summarize

# A word ~ 1 token for Llama-3; good enough to hit an approximate prompt length.
_FILLER_WORD = "benchmark "


def make_prompt(approx_tokens: int) -> str:
    """A deterministic prompt of roughly `approx_tokens` tokens."""
    return ("Repeat this text. " + _FILLER_WORD * max(1, approx_tokens)).strip()


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    output_tokens: int,
) -> RequestResult:
    """Stream a single completion and time first/inter/last token arrivals."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,  # force exactly output_tokens so runs are comparable
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = RequestResult()
    start = time.perf_counter()
    last_token_ts: float | None = None
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/completions", json=payload, timeout=None
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                now = time.perf_counter()
                # Usage-only final chunk carries token counts, no text.
                if chunk.get("usage"):
                    result.prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    result.output_tokens = chunk["usage"].get("completion_tokens", 0)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if result.ttft == 0.0:
                        result.ttft = now - start
                    else:
                        result.itls.append(now - last_token_ts)
                    last_token_ts = now
        result.e2e = time.perf_counter() - start
        # Fall back to counting streamed tokens if the server omitted usage.
        if result.output_tokens == 0:
            result.output_tokens = len(result.itls) + (1 if result.ttft else 0)
        result.success = result.output_tokens > 0
    except Exception as exc:  # noqa: BLE001 - record and keep the run going
        result.error = f"{type(exc).__name__}: {exc}"
        result.success = False
    return result


async def generate_load(
    base_url: str,
    model: str,
    qps: float,
    duration: float,
    input_tokens: int,
    output_tokens: int,
    seed: int = 0,
) -> tuple[list[RequestResult], float]:
    """Fire Poisson-arriving requests for `duration` seconds; return results + wall time."""
    rng = random.Random(seed)
    prompt = make_prompt(input_tokens)
    tasks: list[asyncio.Task] = []

    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(limits=limits) as client:
        wall_start = time.perf_counter()
        elapsed = 0.0
        while elapsed < duration:
            await asyncio.sleep(rng.expovariate(qps))  # exponential inter-arrival
            elapsed = time.perf_counter() - wall_start
            if elapsed >= duration:
                break
            tasks.append(
                asyncio.create_task(
                    one_request(client, base_url, model, prompt, output_tokens)
                )
            )
        results = await asyncio.gather(*tasks) if tasks else []
        wall_duration = time.perf_counter() - wall_start
    return list(results), wall_duration


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surge async load generator")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--input-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="write summary JSON to this path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results, wall = asyncio.run(
        generate_load(
            args.base_url,
            args.model,
            args.qps,
            args.duration,
            args.input_tokens,
            args.output_tokens,
            args.seed,
        )
    )
    summary = summarize(results, wall)
    summary["config"] = vars(args)
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
