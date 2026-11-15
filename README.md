# LLM Roofline DB

Open-source architecture specifications, hardware memory-bandwidth/compute profiles, live API pricing ingestion, and a deterministic **Roofline inference & VRAM sizing engine** for open-weight LLMs across Apple Silicon, AMD Strix Halo Mini-PCs, NVIDIA/AMD desktop GPUs, and Cloud GPUs.

**Interactive Web Calculator & Hardware Matrix:** [https://llmroofline.dev](https://llmroofline.dev)

---

## Why Another LLM Calculator?

Most VRAM and token-speed calculators divide file size by advertised memory bandwidth. That breaks down on modern 2025–2026 architectures:

1. **Mixture-of-Experts (MoE) & Batch Union**: `Qwen3-30B-A3B` stores `30.5B` weights in memory (`17.6 GiB` @ `Q4_K_M`) but only streams `~3.3B` parameters per token at `batch=1`, then approaches the union of active experts as concurrency grows.
2. **Multi-Head Latent Attention (MLA) & Hybrid Sliding-Window KV Cache**: `DeepSeek-R1/V3` stores compressed `(kv_lora_rank + qk_rope_head_dim)` vectors instead of full GQA K/V heads; `Gemma 3` and `Command A` mix short sliding-window local layers with sparse global attention layers.
3. **Long-Context KV Bandwidth Drag & Prefill TTFT**: At `32k–128k` context, reading the KV cache every token step eats memory bandwidth alongside weights, while **Prefill (TTFT)** hits the hardware's **FP16 TFLOPS compute ceiling** (exposing why low-TFLOPS UMA systems lag CUDA GPUs on prompt ingestion).
4. **Unified Memory OS Wired Ceilings**: macOS (`iogpu.wired_limit_mb`) and AMD Strix Halo Linux TTM (`amdgpu.gttsize`) have distinct default vs. tuned memory limits before swap/OOM occurs.

---

## Quick Start (Zero-Dependency CLI)

```bash
# Calculate exact VRAM, KV cache, decode tok/s, prefill tok/s, and TTFT
python3 cli/roofline.py calc \
  --model qwen3-30b-a3b \
  --hw macbook-pro-m3-max-64gb \
  --quant Q4_K_M \
  --ctx 32768

# Rank all hardware profiles capable of running a model
python3 cli/roofline.py rank-hw --model qwen3-32b --quant Q4_K_M --ctx 32768
```

Machine-readable JSON (`--json`) is supported on both subcommands.

---

## Querying the Parquet Matrix with DuckDB

Every release builds `dist/roofline_matrix.parquet` and `dist/llm_roofline.duckdb`:

```sql
SELECT model_slug, hardware_slug, total_32k_gib, decode_32k_tok_s, ttft_32k_s
FROM read_parquet('dist/roofline_matrix.parquet')
WHERE fit_32k IN ('comfortable', 'tight')
  AND model_slug = 'qwen3-30b-a3b'
ORDER BY decode_32k_tok_s DESC;
```

---

## Monthly Dataset Refresh Pipeline

```bash
python3 -m venv .venv
.venv/bin/pip install duckdb pytest
make all
```

`pipeline/refresh_live_sources.py` pulls live download counts and architectural `config.json` verification from Hugging Face plus live serverless token pricing from OpenRouter, then `pipeline/build_duckdb.py` regenerates the DuckDB database, Parquet/CSV exports, and curated JSON manifests.

---

## License

- **Code (`cli/`, `pipeline/`, `tests/`)**: MIT License
- **Structured Datasets (`data/`, `dist/*.json`, `dist/*.csv`, `dist/*.parquet`)**: CC-BY-4.0 (please attribute with a link to this repository or [llmroofline.dev](https://llmroofline.dev)).
