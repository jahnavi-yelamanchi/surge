# Surge — vLLM Serving Optimization for Llama-3

Surge is a small, reproducible benchmarking harness that tunes [vLLM](https://github.com/vllm-project/vllm)
serving for **Llama-3-8B** on a single GPU and measures the resulting
throughput/latency trade-off. It answers one question precisely:

> **How much more throughput can we get at a fixed tail-latency SLO by tuning vLLM's
> PagedAttention, continuous-batching, and KV-cache-eviction knobs — versus a naive default config?**

Everything runs on [Modal](https://modal.com) (serverless GPUs), so there is no
standing infrastructure cost and the whole experiment is one command to reproduce.

> **Status:** 🚧 work in progress — built milestone-by-milestone (see commit history).

---

## What this is (and isn't)

- ✅ A **serving** benchmark: we run a pre-trained model and measure how it behaves under load.
- ✅ Honest config tuning + a custom async load generator with Poisson/bursty traffic.
- ❌ **Not** model training or fine-tuning — there are no gradients here.
- ❌ **Not** a CUDA-kernel rewrite — we tune vLLM's existing knobs and measure rigorously.

## The knobs we tune

| Lever | vLLM parameters | What it controls |
|---|---|---|
| PagedAttention / KV cache | `gpu_memory_utilization`, `block_size`, `kv_cache_dtype` | How much KV cache fits and how it's paged |
| In-flight (continuous) batching | `max_num_seqs`, `max_num_batched_tokens`, `enable_chunked_prefill` | How many requests share each forward pass |
| KV-cache eviction | `preemption_mode` (recompute vs. swap) | What happens when KV cache is exhausted under load |

## Headline result

> _Populated after the final A100 sweep (milestone M6). See [`docs/results.md`](docs/results.md)._

<!-- ![Throughput at fixed p99 SLO](results/plots/throughput_at_slo.png) -->

---

## Repository layout

```
serving/    Modal app + vLLM config presets (the GPU side)
bench/      Async load generator, latency metrics, benchmark runner
analyze/    Turn raw result JSON into plots
results/    Committed raw measurements (raw/) and charts (plots/)
docs/       Methodology, results write-up, and a tuning guide
```

## Quick start

```bash
pip install -r requirements.txt
modal token new          # one-time Modal auth

make smoke               # boot vLLM on a cheap GPU, send one request
make sweep               # run the baseline-vs-tuned benchmark sweep
make plots               # regenerate charts from results/raw/
```

See [`docs/methodology.md`](docs/methodology.md) for exactly how latency and the
SLO are defined, and [`docs/tuning-guide.md`](docs/tuning-guide.md) for what each
knob does.

## License

MIT
