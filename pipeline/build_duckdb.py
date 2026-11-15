#!/usr/bin/env python3
"""Build DuckDB warehouse, Parquet/CSV matrix, and wave-gated pSEO manifests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.roofline import compute_memory_limits, evaluate_pair, max_context_for_limit  # noqa: E402


def compute_worthiness_score(
    model: dict[str, Any],
    hw: dict[str, Any],
    res_8k: dict[str, Any],
    res_32k: dict[str, Any],
    max_ctx_tuned: int,
) -> tuple[int, str]:
    """Score a (model, hw) pair from 0..100 for Wave 1 pSEO inclusion."""
    score = 0
    params = float(model["params_total_b"])
    usable, tuned = compute_memory_limits(hw)
    w_q4 = res_8k["weights_gib"]
    w_iq4 = w_q4 * (4.25 / 4.85)
    w_q3 = w_q4 * (3.91 / 4.85)

    # Tier priority
    score += 18 if model.get("search_tier", 2) == 1 else 8

    # Hardware popularity priority
    if hw["slug"] in {
        "mac-mini-m4-pro-48gb",
        "mac-mini-m4-pro-64gb",
        "macbook-pro-m3-max-64gb",
        "mac-studio-m4-max-128gb",
        "mac-studio-m3-ultra-256gb",
        "mac-studio-m3-ultra-512gb",
        "ryzen-ai-max-395-128gb",
        "ryzen-ai-max-395-64gb",
        "rtx-5090-32gb",
        "rtx-5080-16gb",
        "rtx-5070-ti-16gb",
        "rtx-4090-24gb",
        "rtx-3090-24gb",
        "dual-rtx-3090-48gb",
        "rtx-3060-12gb",
        "cloud-h100-sxm-80gb",
        "cloud-h200-sxm-141gb",
        "cloud-l40s-48gb",
    }:
        score += 14
    else:
        score += 6

    # Decision-boundary realism: avoid trivial mismatches
    utilization = res_8k["total_mem_gib"] / max(1.0, tuned)
    if res_8k["fit_status"] in ("comfortable", "tight", "tunable_uma"):
        if 0.30 <= utilization <= 1.0:
            score += 48
            reason = "sweet_spot_fit"
        elif 0.16 <= utilization < 0.30:
            score += 24
            reason = "high_headroom_fit"
        else:
            score += 5
            reason = "overprovisioned"
    elif w_iq4 + 1.5 <= tuned or w_q3 + 1.5 <= tuned:
        # Needs aggressive quant (IQ4_XS / Q3_K_M) or KV compression
        score += 44
        reason = "quant_boundary_fit"
    elif utilization <= 1.45 and params >= 24.0:
        # High-intent OOM cliff ("can I run X on Y?" -> cloud GPU / upgrade)
        score += 40
        reason = "oom_upgrade_cliff"
    else:
        score += 0
        reason = "unviable_mismatch"

    # Long context drag bonus (32k fit transitions or tight fit)
    if res_8k["fit_status"] != "oom" and res_32k["fit_status"] in ("tight", "tunable_uma", "oom"):
        score += 12
    elif max_ctx_tuned >= 32768 and res_8k["fit_status"] != "oom":
        score += 6

    # MoE / flagship architecture interest bonus
    if model.get("is_moe") and res_8k["fit_status"] != "oom":
        score += 6

    return min(100, score), reason


def build_wave2_comparisons(hardware: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate curated high-intent head-to-head hardware comparisons."""
    by_slug = {h["slug"]: h for h in hardware}
    curated_pairs = [
        ("mac-studio-m4-max-128gb", "ryzen-ai-max-395-128gb", "Apple M4 Max 128GB vs AMD Strix Halo 128GB for Local LLMs"),
        ("mac-mini-m4-pro-64gb", "ryzen-ai-max-395-64gb", "Mac Mini M4 Pro 64GB vs Ryzen AI Max+ 395 64GB"),
        ("macbook-pro-m3-max-64gb", "mac-studio-m4-max-64gb", "MacBook Pro M3 Max 64GB vs Mac Studio M4 Max 64GB"),
        ("mac-studio-m4-max-128gb", "dual-rtx-3090-48gb", "Mac Studio M4 Max 128GB vs Dual RTX 3090 48GB Rig"),
        ("mac-studio-m3-ultra-256gb", "mac-studio-m2-ultra-192gb", "Mac Studio M3 Ultra 256GB vs M2 Ultra 192GB"),
        ("mac-studio-m3-ultra-512gb", "cloud-h200-sxm-141gb", "Mac Studio M3 Ultra 512GB vs Cloud H200 141GB"),
        ("rtx-5090-32gb", "rtx-4090-24gb", "NVIDIA RTX 5090 32GB vs RTX 4090 24GB for LLM Inference"),
        ("rtx-5090-32gb", "dual-rtx-3090-48gb", "RTX 5090 32GB vs Dual RTX 3090 48GB for Local AI"),
        ("rtx-5090-32gb", "mac-studio-m4-max-64gb", "RTX 5090 32GB vs Mac Studio M4 Max 64GB"),
        ("rtx-4090-24gb", "rtx-3090-24gb", "RTX 4090 24GB vs Used RTX 3090 24GB Price/Speed"),
        ("rtx-5080-16gb", "rtx-5070-ti-16gb", "RTX 5080 16GB vs RTX 5070 Ti 16GB for Local LLMs"),
        ("rtx-5070-ti-16gb", "rtx-5060-ti-16gb", "RTX 5070 Ti 16GB vs RTX 5060 Ti 16GB"),
        ("rtx-3090-24gb", "rx-7900-xtx-24gb", "Used RTX 3090 24GB vs Radeon RX 7900 XTX 24GB"),
        ("rtx-3090-24gb", "rtx-3060-12gb", "RTX 3090 24GB vs RTX 3060 12GB Budget LLM Upgrade"),
        ("rtx-5060-ti-16gb", "mac-mini-m4-24gb", "RTX 5060 Ti 16GB vs Mac Mini M4 24GB"),
        ("ryzen-ai-max-395-128gb", "dual-rtx-3090-48gb", "Strix Halo 128GB Mini-PC vs Dual RTX 3090 48GB"),
        ("ryzen-ai-max-395-128gb", "rtx-5090-32gb", "Strix Halo 128GB Mini-PC vs RTX 5090 32GB"),
        ("ryzen-ai-hx-370-64gb", "mac-mini-m4-24gb", "Ryzen AI 9 HX 370 64GB vs Mac Mini M4 24GB"),
        ("rtx-pro-6000-96gb", "mac-studio-m4-max-128gb", "RTX PRO 6000 96GB vs Mac Studio M4 Max 128GB"),
        ("rtx-pro-6000-96gb", "cloud-h100-sxm-80gb", "RTX PRO 6000 96GB Workstation vs Cloud H100 80GB"),
        ("cloud-h100-sxm-80gb", "cloud-a100-sxm-80gb", "Cloud H100 SXM 80GB vs A100 SXM 80GB Cost/Speed"),
        ("cloud-h200-sxm-141gb", "cloud-h100-sxm-80gb", "Cloud H200 141GB vs H100 80GB for Reasoning LLMs"),
        ("cloud-l40s-48gb", "rtx-4090-24gb", "Cloud L40S 48GB vs Cloud/Local RTX 4090 24GB"),
        ("mac-mini-m4-pro-48gb", "rtx-4090-24gb", "Mac Mini M4 Pro 48GB vs RTX 4090 24GB"),
        ("mac-mini-m4-pro-48gb", "ryzen-ai-max-395-64gb", "Mac Mini M4 Pro 48GB vs Strix Halo 64GB"),
        ("macbook-pro-m3-max-128gb", "mac-studio-m4-max-128gb", "MacBook Pro M3 Max 128GB vs Mac Studio M4 Max 128GB"),
        ("mac-mini-m4-16gb", "rtx-3060-12gb", "Mac Mini M4 16GB vs RTX 3060 12GB Entry Local AI"),
    ]

    comparisons: list[dict[str, Any]] = []
    # Also expand strong category comparisons programmatically up to ~115 curated pairs
    seen_slugs: set[str] = set()
    for a, b, title in curated_pairs:
        if a in by_slug and b in by_slug:
            slug = f"{a}-vs-{b}"
            seen_slugs.add(slug)
            comparisons.append({"slug": slug, "hw_a": a, "hw_b": b, "title": title, "priority": 1})

    key_local = [
        "mac-mini-m4-24gb",
        "mac-mini-m4-pro-48gb",
        "mac-mini-m4-pro-64gb",
        "macbook-pro-m3-max-64gb",
        "mac-studio-m4-max-64gb",
        "mac-studio-m4-max-128gb",
        "mac-studio-m3-ultra-256gb",
        "ryzen-ai-max-395-128gb",
        "ryzen-ai-max-395-64gb",
        "ryzen-ai-hx-370-64gb",
        "rtx-5090-32gb",
        "rtx-5080-16gb",
        "rtx-5070-ti-16gb",
        "rtx-5060-ti-16gb",
        "rtx-4090-24gb",
        "rtx-3090-24gb",
        "dual-rtx-3090-48gb",
        "rx-7900-xtx-24gb",
        "cloud-l40s-48gb",
        "cloud-a100-sxm-80gb",
        "cloud-h100-sxm-80gb",
    ]
    for i, a in enumerate(key_local):
        for b in key_local[i + 1 :]:
            slug = f"{a}-vs-{b}"
            rev_slug = f"{b}-vs-{a}"
            if slug in seen_slugs or rev_slug in seen_slugs:
                continue
            ha, hb = by_slug[a], by_slug[b]
            ua, _ = compute_memory_limits(ha)
            ub, _ = compute_memory_limits(hb)
            ratio = max(ua, ub) / max(1.0, min(ua, ub))
            # Only compare reasonably proximate hardware tiers or direct local-vs-cloud choices
            if ratio <= 2.6 or ha["category"] != hb["category"] and ratio <= 3.5:
                seen_slugs.add(slug)
                comparisons.append(
                    {
                        "slug": slug,
                        "hw_a": a,
                        "hw_b": b,
                        "title": f"{ha['name']} vs {hb['name']} for Local LLM Inference",
                        "priority": 2,
                    }
                )
            if len(comparisons) >= 120:
                break
        if len(comparisons) >= 120:
            break

    return comparisons


def main() -> int:
    seed_dir = ROOT / "data" / "seed"
    live_models = ROOT / "data" / "live" / "models_live.json"
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    models_path = live_models if live_models.exists() else seed_dir / "models_seed.json"
    models: list[dict[str, Any]] = json.loads(models_path.read_text())
    hardware: list[dict[str, Any]] = json.loads((seed_dir / "hardware_seed.json").read_text())
    quantizations: list[dict[str, Any]] = json.loads((seed_dir / "quantizations_seed.json").read_text())

    # Enrich hardware with computed usable limits so frontend reads canonical ceilings directly
    for hw in hardware:
        usable, tuned = compute_memory_limits(hw)
        hw["usable_mem_gib"] = usable
        hw["tuned_usable_mem_gib"] = tuned

    matrix_rows: list[dict[str, Any]] = []
    spokes_candidates: list[dict[str, Any]] = []

    for model in models:
        for hw in hardware:
            _, tuned_limit = compute_memory_limits(hw)
            usable_limit, _ = compute_memory_limits(hw)
            res_8k = evaluate_pair(model, hw, quant="Q4_K_M", ctx_tokens=8192, kv_quant="FP16")
            res_32k = evaluate_pair(model, hw, quant="Q4_K_M", ctx_tokens=32768, kv_quant="FP16")
            max_ctx_default = max_context_for_limit(model, hw, "Q4_K_M", "FP16", usable_limit)
            max_ctx_tuned = max_context_for_limit(model, hw, "Q4_K_M", "FP16", tuned_limit)
            score, reason = compute_worthiness_score(model, hw, res_8k, res_32k, max_ctx_tuned)

            row = {
                "slug": f"{model['slug']}-on-{hw['slug']}",
                "model_slug": model["slug"],
                "hardware_slug": hw["slug"],
                "quant": "Q4_K_M",
                "kv_quant": "FP16",
                "weights_gib": res_8k["weights_gib"],
                "kv_8k_gib": res_8k["kv_cache_gib"],
                "total_8k_gib": res_8k["total_mem_gib"],
                "fit_8k": res_8k["fit_status"],
                "decode_8k_tok_s": res_8k["decode_tok_s"],
                "prefill_8k_tok_s": res_8k["prefill_tok_s"],
                "ttft_8k_s": res_8k["ttft_s"],
                "total_32k_gib": res_32k["total_mem_gib"],
                "fit_32k": res_32k["fit_status"],
                "decode_32k_tok_s": res_32k["decode_tok_s"],
                "prefill_32k_tok_s": res_32k["prefill_tok_s"],
                "ttft_32k_s": res_32k["ttft_s"],
                "max_ctx_default": max_ctx_default,
                "max_ctx_tuned": max_ctx_tuned,
                "worthiness_score": score,
                "worthiness_reason": reason,
            }
            matrix_rows.append(row)
            if score >= 70:
                spokes_candidates.append(row)

    spokes_candidates.sort(key=lambda r: (-r["worthiness_score"], -r["decode_32k_tok_s"], r["slug"]))
    spokes_wave1 = spokes_candidates[:320]
    comparisons_wave2 = build_wave2_comparisons(hardware)

    # Write JSON exports
    (dist_dir / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    (dist_dir / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    (dist_dir / "quantizations.json").write_text(json.dumps(quantizations, indent=2) + "\n")
    (dist_dir / "spokes_wave1.json").write_text(json.dumps(spokes_wave1, indent=2) + "\n")
    (dist_dir / "comparisons_wave2.json").write_text(json.dumps(comparisons_wave2, indent=2) + "\n")

    # Build DuckDB database and export Parquet + CSV
    db_path = dist_dir / "llm_roofline.duckdb"
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        models_tmp = dist_dir / "models.json"
        hw_tmp = dist_dir / "hardware.json"
        matrix_tmp = dist_dir / "_matrix_tmp.json"
        matrix_tmp.write_text(json.dumps(matrix_rows) + "\n")

        con.execute(f"CREATE TABLE models AS SELECT * FROM read_json_auto('{models_tmp}')")
        con.execute(f"CREATE TABLE hardware AS SELECT * FROM read_json_auto('{hw_tmp}')")
        con.execute(f"CREATE TABLE roofline_matrix AS SELECT * FROM read_json_auto('{matrix_tmp}')")

        parquet_path = dist_dir / "roofline_matrix.parquet"
        csv_path = dist_dir / "roofline_matrix.csv"
        con.execute(f"COPY roofline_matrix TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.execute(f"COPY roofline_matrix TO '{csv_path}' (FORMAT CSV, HEADER true)")
        matrix_tmp.unlink(missing_ok=True)
    finally:
        con.close()

    print(
        f"Built DuckDB & exports: {len(models)} models × {len(hardware)} hardware = {len(matrix_rows)} pairs | "
        f"Wave 1 spokes={len(spokes_wave1)} (from {len(spokes_candidates)} >=70) | Wave 2 comparisons={len(comparisons_wave2)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
