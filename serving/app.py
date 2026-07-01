"""Modal app for Surge — serves Llama-3-8B with vLLM and runs benchmarks on a GPU.

Entry points (see Makefile):
    modal run serving/app.py::smoke    # M1: boot the server, send one request
    modal run serving/app.py::sweep    # M3+: baseline-vs-tuned benchmark sweep

The smoke test boots the *real* vLLM OpenAI-compatible server as a subprocess and
hits it over localhost — the same serving path the benchmark later measures. The
helpers here (`start_server`, `wait_for_health`) are reused by the M3 runner.
"""

import subprocess
import time

import modal

# Ungated mirror of Llama-3-8B-Instruct — identical weights, no HF signup/token.
MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
VLLM_PORT = 8000
BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"

# Cheap GPU for dev/smoke; the final sweep (M6) overrides this to an A100.
GPU_SMOKE = "a10g"

app = modal.App("surge")

# Persist the ~16GB of HF weights across runs so we download them only once.
hf_cache = modal.Volume.from_name("surge-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3.post1",  # pinned for reproducibility; some knobs vary by version
        # vLLM 0.6.3 leaves transformers unpinned; the package mirror otherwise
        # resolves it to transformers 5.x, whose tokenizer API breaks vLLM's
        # get_cached_tokenizer (all_special_tokens_extended). Pin known-good versions.
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "httpx==0.27.2",
    )
    .run_commands(
        # We run vLLM with the `lm-format-enforcer` guided-decoding backend instead
        # of the default `outlines` (whose pyairports dep is broken on Modal's
        # package mirror). We never use guided decoding for benchmarking, so this
        # just avoids a dead import. Verify the backend lib imports at build time.
        "python -c 'import lmformatenforcer'"
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "VLLM_DO_NOT_TRACK": "1",  # no usage telemetry
        }
    )
    # Ship our local load generator into the container so the benchmark runs
    # server-side (client and server share localhost -> no internet jitter).
    .add_local_python_source("bench")
)


def start_server(extra_args: list[str] | None = None) -> subprocess.Popen:
    """Launch `vllm serve` as a subprocess. extra_args sets the config knobs."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--port",
        str(VLLM_PORT),
        "--disable-log-requests",
        # Use lm-format-enforcer instead of the default `outlines` backend, which
        # has a broken pyairports dep on Modal's mirror. We never request guided
        # decoding, so this only changes which (unused) module gets imported.
        "--guided-decoding-backend",
        "lm-format-enforcer",
    ] + (extra_args or [])
    print(f"[surge] starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_health(timeout: float = 900.0) -> None:
    """Block until the server answers /health (covers the first-run model download)."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                print(f"[surge] server healthy after {time.time() - start:.0f}s")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM server not healthy within {timeout:.0f}s")


@app.function(
    image=image,
    gpu=GPU_SMOKE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=60 * 60,
)
def smoke() -> None:
    """M1 smoke test: boot vLLM, send one chat completion, print the reply."""
    import httpx

    proc = start_server(["--max-model-len", "4096", "--gpu-memory-utilization", "0.90"])
    try:
        wait_for_health()
        hf_cache.commit()  # persist freshly downloaded weights for next time

        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": "In one sentence, what is paged attention in LLM serving?",
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        print("\n=== vLLM smoke test OK ===")
        print(reply.strip())
        print("==========================\n")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


# Default load shape for a benchmark point.
DEFAULT_INPUT_TOKENS = 512
DEFAULT_OUTPUT_TOKENS = 128


@app.function(
    image=image,
    gpu=GPU_SMOKE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=60 * 60,
)
def run_benchmark(
    config_name: str,
    config_args: list[str],
    qps: float,
    duration: float,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """Boot vLLM with `config_args`, drive Poisson load at `qps`, return a summary."""
    import asyncio

    from bench.load_gen import generate_load
    from bench.metrics import summarize

    proc = start_server(config_args)
    try:
        wait_for_health()
        hf_cache.commit()
        results, wall = asyncio.run(
            generate_load(
                BASE_URL, MODEL_NAME, qps, duration, input_tokens, output_tokens
            )
        )
        summary = summarize(results, wall)
        summary["config_name"] = config_name
        summary["config_args"] = config_args
        summary["qps"] = qps
        summary["gpu"] = GPU_SMOKE
        summary["input_tokens"] = input_tokens
        summary["output_tokens"] = output_tokens
        return summary
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


@app.local_entrypoint()
def sweep(
    qps: float = 3.0,
    duration: float = 30.0,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> None:
    """Run every preset at a single QPS and write one JSON per config locally.

    M3: a single-QPS baseline-vs-tuned comparison to validate the pipeline.
    M4 upgrades this to a QPS sweep that finds max throughput at a latency SLO.
    """
    import json
    import os

    from serving.configs import CONFIGS

    os.makedirs("results/raw", exist_ok=True)
    summaries = {}
    for name, cfg in CONFIGS.items():
        print(f"[surge] running '{name}' @ {qps} qps for {duration:g}s ...")
        summary = run_benchmark.remote(
            name, cfg["args"], qps, duration, input_tokens, output_tokens
        )
        summaries[name] = summary
        path = f"results/raw/{name}_qps{qps:g}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[surge] wrote {path}")

    print("\n=== summary ===")
    for name, s in summaries.items():
        print(
            f"{name:9s} tok/s={s['output_token_throughput']:7.1f}  "
            f"p99 ITL={s['itl_ms']['p99']:6.1f}ms  "
            f"p99 e2e={s['e2e_ms']['p99']:7.1f}ms  "
            f"errors={s['num_errors']}"
        )
