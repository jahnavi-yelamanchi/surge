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
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "VLLM_DO_NOT_TRACK": "1",  # no usage telemetry
        }
    )
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


@app.function(image=image, gpu=GPU_SMOKE, timeout=60 * 60)
def sweep() -> None:
    """M3+: baseline-vs-tuned benchmark sweep. Implemented in a later milestone."""
    raise NotImplementedError("sweep is implemented in milestone M3/M4")
