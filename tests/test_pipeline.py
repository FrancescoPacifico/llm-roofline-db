"""Automated verification suite for LLM Roofline DB equations, exports, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import duckdb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.roofline import evaluate_pair, main as cli_main  # noqa: E402


def _load_dist() -> tuple[dict[str, dict], dict[str, dict]]:
    models = {m["slug"]: m for m in json.loads((ROOT / "dist" / "models.json").read_text())}
    hw = {h["slug"]: h for h in json.loads((ROOT / "dist" / "hardware.json").read_text())}
    return models, hw


def test_catalog_counts_and_files() -> None:
    models, hw = _load_dist()
    assert len(models) >= 32
    assert len(hw) >= 28
    assert (ROOT / "dist" / "roofline_matrix.parquet").exists()
    assert (ROOT / "dist" / "roofline_matrix.csv").exists()
    spokes = json.loads((ROOT / "dist" / "spokes_wave1.json").read_text())
    comparisons = json.loads((ROOT / "dist" / "comparisons_wave2.json").read_text())
    assert 200 <= len(spokes) <= 320
    assert 80 <= len(comparisons) <= 130


def test_moe_active_vs_total_math() -> None:
    models, hw = _load_dist()
    m_moe = models["qwen3-30b-a3b"]
    m_dense = models["qwen3-32b"]
    m3 = hw["macbook-pro-m3-max-64gb"]

    r_moe = evaluate_pair(m_moe, m3, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16", batch_size=1)
    r_dense = evaluate_pair(m_dense, m3, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16", batch_size=1)

    assert r_moe["fit_status"] == "comfortable"
    # MoE with 3.3B active must decode multiple times faster than 32.8B dense on same bandwidth
    assert r_moe["decode_tok_s"] > r_dense["decode_tok_s"] * 3.5

    # Batching increases union of routed active experts and reduces per-user decode speed slightly while raising total throughput
    r_moe_b8 = evaluate_pair(m_moe, m3, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16", batch_size=8)
    assert r_moe_b8["eff_active_params_b"] > r_moe["eff_active_params_b"]
    assert r_moe_b8["decode_tok_s"] > r_moe["decode_tok_s"]
    assert r_moe_b8["per_user_decode_tok_s"] < r_moe["per_user_decode_tok_s"]


def test_mla_kv_cache_efficiency() -> None:
    models, hw = _load_dist()
    r1 = models["deepseek-r1-671b"]
    llama_405 = models["llama-3-1-405b"]
    m3_512 = hw["mac-studio-m3-ultra-512gb"]

    res_r1 = evaluate_pair(r1, m3_512, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")
    res_405 = evaluate_pair(llama_405, m3_512, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")

    # DeepSeek-R1 671B MLA (kv_lora_rank=512 + rope=64) uses drastically smaller KV cache than 126-layer GQA 405B
    assert res_r1["kv_cache_gib"] < res_405["kv_cache_gib"] * 0.5
    assert res_r1["fit_status"] in ("comfortable", "tight")


def test_sliding_window_hybrid_kv_cache() -> None:
    models, hw = _load_dist()
    gemma27 = models["gemma-3-27b"]
    rtx5090 = hw["rtx-5090-32gb"]

    res_32k = evaluate_pair(gemma27, rtx5090, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")
    # 52 local layers capped at sliding_window=1024 + 10 global layers @ 32768 keeps KV compact
    assert res_32k["kv_cache_gib"] < 3.0


def test_uma_tunable_ceiling_status() -> None:
    models, hw = _load_dist()
    q235 = models["qwen3-235b-a22b"]
    # 128 GB Strix Halo tuned limit is 110 GiB vs 96 GiB default; check 70B on 64 GB Mac Mini M4 Pro (48.0 vs 51.92 tuned)
    llama70 = models["llama-3-3-70b"]
    m4pro64 = hw["mac-mini-m4-pro-64gb"]
    res = evaluate_pair(llama70, m4pro64, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")
    assert res["fit_status"] == "tunable_uma"
    assert res["total_mem_gib"] <= res["tuned_usable_mem_gib"]
    assert q235["is_moe"] is True


def test_duckdb_parquet_queryable() -> None:
    parquet = ROOT / "dist" / "roofline_matrix.parquet"
    rows = duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{parquet}') WHERE fit_32k != 'oom'"
    ).fetchall()
    assert rows and rows[0][0] > 200


def test_cli_json_invocation() -> None:
    assert cli_main(
        ["calc", "--model", "qwen3-30b-a3b", "--hw", "macbook-pro-m3-max-64gb", "--quant", "Q4_K_M", "--ctx", "32768", "--json"]
    ) == 0
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli" / "roofline.py"),
            "calc",
            "--model",
            "qwen3-30b-a3b",
            "--hw",
            "macbook-pro-m3-max-64gb",
            "--quant",
            "Q4_K_M",
            "--ctx",
            "32768",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["fit_status"] == "comfortable"
    assert payload["decode_tok_s"] > 30
