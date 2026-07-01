"""Named vLLM engine config presets for the benchmark.

Each preset is a list of extra `vllm serve` flags layered on top of the base
command in app.py. `baseline` is deliberately naive so the tuning gains are
honest and attributable; `tuned` turns the knobs documented in
docs/tuning-guide.md. The knobs map directly onto the resume bullets:

  paged-attention / KV cache  -> --gpu-memory-utilization
  in-flight (continuous) batch -> --max-num-seqs, --max-num-batched-tokens,
                                  --enable-chunked-prefill
"""

MAX_MODEL_LEN = 4096

CONFIGS: dict[str, dict] = {
    "baseline": {
        "description": (
            "Naive defaults: half the KV-cache budget, a small in-flight batch, "
            "and no chunked prefill."
        ),
        "args": [
            "--gpu-memory-utilization", "0.50",
            "--max-num-seqs", "32",
            "--max-model-len", str(MAX_MODEL_LEN),
        ],
    },
    "tuned": {
        "description": (
            "Larger KV-cache budget, a wider in-flight batch, and chunked prefill "
            "enabled for better prefill/decode overlap."
        ),
        "args": [
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "256",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "8192",
            "--max-model-len", str(MAX_MODEL_LEN),
        ],
    },
}


def config_args(name: str) -> list[str]:
    return CONFIGS[name]["args"]
