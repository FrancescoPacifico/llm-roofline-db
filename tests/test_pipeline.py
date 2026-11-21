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

from cli.roofline import compute_kv_bytes, compute_memory_limits, evaluate_pair, main as cli_main  # noqa: E402
from pipeline.build_duckdb import preserve_published_comparisons, preserve_published_spokes  # noqa: E402


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


def test_backfill_catalog_and_hybrid_kv_math() -> None:
    models, hw = _load_dist()
    assert len(models) >= 58
    assert len(hw) >= 36

    qwen35_35b = models["qwen3-5-35b-a3b"]
    qwen3_32b = models["qwen3-32b"]
    m5_max = hw["mac-studio-m5-max-128gb"]

    r_hybrid_32k = evaluate_pair(qwen35_35b, m5_max, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")
    r_hybrid_128k = evaluate_pair(qwen35_35b, m5_max, quant="Q4_K_M", ctx_tokens=131072, kv_quant="FP16")
    r_gqa_32k = evaluate_pair(qwen3_32b, m5_max, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")

    assert r_hybrid_32k["fit_status"] == "comfortable"
    assert r_hybrid_32k["kv_cache_gib"] < 1.0
    assert r_hybrid_32k["kv_cache_gib"] < r_gqa_32k["kv_cache_gib"] * 0.15
    assert r_hybrid_128k["kv_cache_gib"] < 3.0


def test_gemma4_heterogeneous_sliding_window_kv() -> None:
    models, _ = _load_dist()
    g4_31b = models["gemma-4-31b"]
    g4_26b = models["gemma-4-26b-a4b"]

    kv_31b_bytes = compute_kv_bytes(g4_31b, ctx_tokens=32768, kv_quant="FP16", batch_size=1)
    kv_26b_bytes = compute_kv_bytes(g4_26b, ctx_tokens=32768, kv_quant="FP16", batch_size=1)

    # 10 global layers (4 kv_heads * 512 dim) + 50 sliding layers capped at 1024 tokens (16 kv_heads * 256 dim)
    expected_31b = (2 * 10 * 32768 * 4 * 512 * 2) + (2 * 50 * 1024 * 16 * 256 * 2)
    assert kv_31b_bytes == float(expected_31b)
    # 5 global layers (2 kv_heads * 512 dim) + 25 sliding layers capped at 1024 tokens (8 kv_heads * 256 dim)
    expected_26b = (2 * 5 * 32768 * 2 * 512 * 2) + (2 * 25 * 1024 * 8 * 256 * 2)
    assert kv_26b_bytes == float(expected_26b)


def test_dgx_spark_uma_limits_and_gpt_oss() -> None:
    models, hw = _load_dist()
    spark = hw["nvidia-dgx-spark-gb10-128gb"]
    usable, tuned = compute_memory_limits(spark)
    assert usable == 112.0
    assert tuned == 119.0

    gpt120 = models["gpt-oss-120b"]
    res = evaluate_pair(gpt120, spark, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")
    assert res["fit_status"] == "comfortable"
    assert res["decode_tok_s"] > 20


def test_anti_404_manifest_preservation(tmp_path: Path) -> None:
    spokes_path = tmp_path / "spokes_wave1.json"
    spokes_path.write_text(json.dumps([{"slug": "legacy-model-on-legacy-hw", "worthiness_score": 60, "decode_32k_tok_s": 5.0}]))
    all_rows = {
        "legacy-model-on-legacy-hw": {"slug": "legacy-model-on-legacy-hw", "worthiness_score": 71, "decode_32k_tok_s": 12.0},
        "new-a-on-hw": {"slug": "new-a-on-hw", "worthiness_score": 95, "decode_32k_tok_s": 50.0},
        "new-b-on-hw": {"slug": "new-b-on-hw", "worthiness_score": 90, "decode_32k_tok_s": 40.0},
    }
    candidates = [all_rows["new-a-on-hw"], all_rows["new-b-on-hw"]]

    wave0_spokes = preserve_published_spokes(candidates, all_rows, spokes_path, active_wave=0, cap=2)
    assert [s["slug"] for s in wave0_spokes] == ["new-a-on-hw", "new-b-on-hw"]

    wave1_spokes = preserve_published_spokes(candidates, all_rows, spokes_path, active_wave=1, cap=2)
    wave1_slugs = {s["slug"] for s in wave1_spokes}
    assert "legacy-model-on-legacy-hw" in wave1_slugs
    assert next(s for s in wave1_spokes if s["slug"] == "legacy-model-on-legacy-hw")["decode_32k_tok_s"] == 12.0

    comp_path = tmp_path / "comparisons_wave2.json"
    comp_path.write_text(json.dumps([{"slug": "hw-a-vs-hw-b", "hw_a": "hw-a", "hw_b": "hw-b", "title": "Old", "priority": 2}]))
    hw_map = {"hw-a": {}, "hw-b": {}, "hw-c": {}, "hw-d": {}}
    fresh_comps = [{"slug": "hw-c-vs-hw-d", "hw_a": "hw-c", "hw_b": "hw-d", "title": "New", "priority": 1}]

    wave0_comps = preserve_published_comparisons(fresh_comps, hw_map, comp_path, active_wave=0, cap=1)
    assert [c["slug"] for c in wave0_comps] == ["hw-c-vs-hw-d"]

    wave2_comps = preserve_published_comparisons(fresh_comps, hw_map, comp_path, active_wave=2, cap=2)
    assert [c["slug"] for c in wave2_comps] == ["hw-a-vs-hw-b", "hw-c-vs-hw-d"]
