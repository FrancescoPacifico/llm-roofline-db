#!/usr/bin/env python3
"""Deterministic LLM Roofline Sizing & Speed CLI (zero external dependencies)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

GIB = 1024**3

QUANT_BPW: dict[str, float] = {
    "FP16": 16.00,
    "Q8_0": 8.50,
    "Q6_K": 6.57,
    "Q5_K_M": 5.69,
    "Q4_K_M": 4.85,
    "MLX_4BIT": 4.50,
    "IQ4_XS": 4.25,
    "Q3_K_M": 3.91,
}

KV_BYTES: dict[str, float] = {
    "FP16": 2.0,
    "Q8_0": 1.0625,
    "Q4_0": 0.5625,
}


def compute_memory_limits(hw: dict[str, Any]) -> tuple[float, float]:
    """Return (usable_mem_gib, tuned_usable_mem_gib) for a hardware profile."""
    category = hw["category"]
    vram_gib = float(hw["vram_gib"])
    num_gpus = int(hw.get("num_gpus", 1))

    if hw.get("usable_mem_override_gib") is not None:
        usable = round(float(hw["usable_mem_override_gib"]), 2)
        tuned_raw = hw.get("tuned_usable_mem_override_gib")
        tuned = round(float(tuned_raw), 2) if tuned_raw is not None else usable
        return max(1.0, usable), max(1.0, tuned)

    if category in ("discrete_gpu", "cloud_gpu"):
        driver_overhead = 0.6 if category == "cloud_gpu" else 1.0
        usable = round(vram_gib * num_gpus - driver_overhead - max(0, num_gpus - 1) * 0.35, 2)
        return usable, usable

    if category == "apple_silicon":
        usable = round(min(vram_gib * 0.78, vram_gib - 6.0), 2)
        tuned = round(min(vram_gib * 0.88, vram_gib - 4.5), 2)
        return max(1.0, usable), max(1.0, tuned)

    # uma_minipc
    if vram_gib >= 128:
        return 96.0, 110.0
    if vram_gib >= 64:
        return 48.0, 54.0
    usable = round(vram_gib * 0.75, 2)
    tuned = round(vram_gib * 0.85, 2)
    return usable, tuned


def compute_kv_bytes(model: dict[str, Any], ctx_tokens: int, kv_quant: str, batch_size: int) -> float:
    """Return exact KV cache memory traffic/footprint in bytes."""
    kv_bytes = KV_BYTES[kv_quant]
    num_layers = int(model["num_layers"])
    sliding_window = int(model.get("sliding_window", 0))
    global_layer_ratio = float(model.get("global_layer_ratio", 1.0))
    attention_type = model.get("attention_type")

    if attention_type == "hybrid_linear":
        global_layers = round(num_layers * global_layer_ratio)
        linear_layers = num_layers - global_layers
        full_kv = (
            2
            * global_layers
            * ctx_tokens
            * int(model["num_kv_heads"])
            * int(model["head_dim"])
            * kv_bytes
            * batch_size
        )
        linear_elems = int(model.get("linear_state_elems_per_layer", 0) or 0)
        linear_state = linear_layers * linear_elems * 4.0 * batch_size
        return float(full_kv + linear_state)

    if sliding_window > 0 and global_layer_ratio < 1.0:
        global_layers = round(num_layers * global_layer_ratio)
        local_layers = num_layers - global_layers
        layer_tokens = global_layers * ctx_tokens + local_layers * min(ctx_tokens, sliding_window)
    else:
        layer_tokens = num_layers * ctx_tokens

    if attention_type == "mla":
        per_token_dim = int(model["kv_lora_rank"]) + int(model["qk_rope_head_dim"])
        return float(layer_tokens * per_token_dim * kv_bytes * batch_size)

    if sliding_window > 0 and global_layer_ratio < 1.0:
        global_layers = round(num_layers * global_layer_ratio)
        local_layers = num_layers - global_layers
        global_kv_heads = int(model.get("global_num_kv_heads") or model["num_kv_heads"])
        global_head_dim = int(model.get("global_head_dim") or model["head_dim"])
        global_bytes = 2 * global_layers * ctx_tokens * global_kv_heads * global_head_dim * kv_bytes * batch_size
        local_bytes = (
            2
            * local_layers
            * min(ctx_tokens, sliding_window)
            * int(model["num_kv_heads"])
            * int(model["head_dim"])
            * kv_bytes
            * batch_size
        )
        return float(global_bytes + local_bytes)

    return float(2 * layer_tokens * int(model["num_kv_heads"]) * int(model["head_dim"]) * kv_bytes * batch_size)


def evaluate_pair(
    model: dict[str, Any],
    hw: dict[str, Any],
    quant: str = "Q4_K_M",
    ctx_tokens: int = 8192,
    kv_quant: str = "FP16",
    prompt_tokens: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Compute exact Roofline memory, prefill/decode tok/s, TTFT, and fit status."""
    if quant not in QUANT_BPW:
        raise ValueError(f"Unsupported quantization: {quant}")
    if kv_quant not in KV_BYTES:
        raise ValueError(f"Unsupported KV cache precision: {kv_quant}")

    bpw = QUANT_BPW[quant]
    eff_prompt = min(ctx_tokens, 4096) if prompt_tokens is None else min(prompt_tokens, ctx_tokens)
    usable_mem_gib, tuned_usable_mem_gib = compute_memory_limits(hw)

    params_total_b = float(model["params_total_b"])
    params_active_b = float(model["params_active_b"])
    is_moe = bool(model["is_moe"])

    weights_gib = (params_total_b * 1e9 * (bpw / 8.0) * 1.02) / GIB
    kv_bytes_total = compute_kv_bytes(model, ctx_tokens, kv_quant, batch_size)
    kv_cache_gib = kv_bytes_total / GIB

    runtime_base_gib = 0.55 if hw["category"] in ("apple_silicon", "uma_minipc") else 0.75
    overhead_gib = runtime_base_gib + ((int(model["hidden_size"]) * min(ctx_tokens, 8192) * batch_size * 18.0) / GIB)
    total_mem_gib = weights_gib + kv_cache_gib + overhead_gib

    if total_mem_gib <= usable_mem_gib * 0.90:
        fit_status = "comfortable"
    elif total_mem_gib <= usable_mem_gib:
        fit_status = "tight"
    elif total_mem_gib <= tuned_usable_mem_gib:
        fit_status = "tunable_uma"
    else:
        fit_status = "oom"

    # Effective active parameters under batching
    if not is_moe or batch_size == 1:
        eff_params_b = params_active_b if is_moe else params_total_b
    else:
        shared_b = max(0.5, params_active_b * 0.15)
        routed_pool_b = params_total_b - shared_b
        p_single = float(model["num_experts_active"]) / float(model["num_experts_total"])
        eff_params_b = min(params_total_b, shared_b + routed_pool_b * (1.0 - (1.0 - p_single) ** batch_size))

    active_weight_bytes = eff_params_b * 1e9 * (bpw / 8.0) * 1.02
    step_bytes = active_weight_bytes + kv_bytes_total

    moe_penalty = 0.88 if is_moe else 1.0
    num_gpus = int(hw.get("num_gpus", 1))
    interconnect = hw.get("interconnect", "pcie")
    multi_gpu_eff = 1.0 if num_gpus == 1 else (0.92 if interconnect == "sxm" else 0.84)

    eff_bw_bytes_s = float(hw["mem_bw_gbs"]) * 1e9 * float(hw["bw_efficiency"]) * moe_penalty * multi_gpu_eff
    bw_bound_tok_s = (eff_bw_bytes_s / step_bytes) * batch_size
    compute_bound_tok_s = (
        float(hw["fp16_tflops"]) * 1e12 * float(hw["compute_efficiency"]) * multi_gpu_eff
    ) / (2.0 * params_active_b * 1e9)

    decode_tok_s = min(bw_bound_tok_s, compute_bound_tok_s)
    per_user_decode_tok_s = decode_tok_s / batch_size
    bottleneck = "memory_bandwidth" if bw_bound_tok_s <= compute_bound_tok_s else "compute_tflops"

    linear_flops_per_tok = 2.0 * params_active_b * 1e9
    attn_layers = (
        round(int(model["num_layers"]) * float(model.get("global_layer_ratio", 1.0)))
        if model.get("attention_type") == "hybrid_linear"
        else int(model["num_layers"])
    )
    attn_flops_per_tok = (
        2.0
        * attn_layers
        * int(model["num_q_heads"])
        * int(model["head_dim"])
        * eff_prompt
    )
    prefill_tok_s = (
        float(hw["fp16_tflops"]) * 1e12 * float(hw["compute_efficiency"]) * multi_gpu_eff
    ) / (linear_flops_per_tok + attn_flops_per_tok)
    ttft_s = 0.035 + (eff_prompt / prefill_tok_s)

    return {
        "model": model["slug"],
        "hardware": hw["slug"],
        "quant": quant,
        "kv_quant": kv_quant,
        "ctx_tokens": ctx_tokens,
        "prompt_tokens": eff_prompt,
        "batch_size": batch_size,
        "usable_mem_gib": round(usable_mem_gib, 2),
        "tuned_usable_mem_gib": round(tuned_usable_mem_gib, 2),
        "weights_gib": round(weights_gib, 2),
        "kv_cache_gib": round(kv_cache_gib, 2),
        "overhead_gib": round(overhead_gib, 2),
        "total_mem_gib": round(total_mem_gib, 2),
        "headroom_gib": round(usable_mem_gib - total_mem_gib, 2),
        "fit_status": fit_status,
        "eff_active_params_b": round(eff_params_b, 2),
        "bw_bound_tok_s": round(bw_bound_tok_s, 2),
        "compute_bound_tok_s": round(compute_bound_tok_s, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "per_user_decode_tok_s": round(per_user_decode_tok_s, 2),
        "prefill_tok_s": round(prefill_tok_s, 2),
        "ttft_s": round(ttft_s, 3),
        "bottleneck": bottleneck,
    }


def max_context_for_limit(
    model: dict[str, Any],
    hw: dict[str, Any],
    quant: str,
    kv_quant: str,
    limit_gib: float,
    batch_size: int = 1,
) -> int:
    """Compute maximum context length (multiple of 512) that fits within limit_gib."""
    max_ctx = int(model["max_context"])
    lo, hi = 0, max_ctx // 512
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        ctx = max(512, mid * 512) if mid > 0 else 512
        res = evaluate_pair(model, hw, quant=quant, ctx_tokens=ctx, kv_quant=kv_quant, batch_size=batch_size)
        if res["total_mem_gib"] <= limit_gib:
            best = ctx if mid > 0 else 512
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _load_catalog(base_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    dist_dir = base_dir / "dist"
    seed_dir = base_dir / "data" / "seed"
    models_path = dist_dir / "models.json" if (dist_dir / "models.json").exists() else seed_dir / "models_seed.json"
    hw_path = dist_dir / "hardware.json" if (dist_dir / "hardware.json").exists() else seed_dir / "hardware_seed.json"

    models = {item["slug"]: item for item in json.loads(models_path.read_text())}
    hardware = {item["slug"]: item for item in json.loads(hw_path.read_text())}
    return models, hardware


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM Roofline Deterministic Hardware & Speed CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    calc_p = sub.add_parser("calc", help="Evaluate a single model + hardware configuration")
    calc_p.add_argument("--model", required=True, help="Model slug (e.g. qwen3-30b-a3b)")
    calc_p.add_argument("--hw", required=True, help="Hardware slug (e.g. macbook-pro-m3-max-64gb)")
    calc_p.add_argument("--quant", default="Q4_K_M", choices=list(QUANT_BPW.keys()))
    calc_p.add_argument("--ctx", type=int, default=8192, help="Context length in tokens")
    calc_p.add_argument("--kv", default="FP16", choices=list(KV_BYTES.keys()))
    calc_p.add_argument("--prompt", type=int, default=None, help="Prompt length for prefill/TTFT")
    calc_p.add_argument("--batch", type=int, default=1, help="Batch concurrency")
    calc_p.add_argument("--json", action="store_true", help="Emit JSON")

    rank_p = sub.add_parser("rank-hw", help="Rank all hardware profiles for a model")
    rank_p.add_argument("--model", required=True, help="Model slug")
    rank_p.add_argument("--quant", default="Q4_K_M", choices=list(QUANT_BPW.keys()))
    rank_p.add_argument("--ctx", type=int, default=32768, help="Context length in tokens")
    rank_p.add_argument("--kv", default="FP16", choices=list(KV_BYTES.keys()))
    rank_p.add_argument("--json", action="store_true", help="Emit JSON")

    args = parser.parse_args(argv)
    base_dir = Path(__file__).resolve().parent.parent
    models, hardware = _load_catalog(base_dir)

    if args.model not in models:
        print(f"Unknown model slug: {args.model}", file=sys.stderr)
        return 2

    model = models[args.model]

    if args.command == "calc":
        if args.hw not in hardware:
            print(f"Unknown hardware slug: {args.hw}", file=sys.stderr)
            return 2
        res = evaluate_pair(
            model,
            hardware[args.hw],
            quant=args.quant,
            ctx_tokens=args.ctx,
            kv_quant=args.kv,
            prompt_tokens=args.prompt,
            batch_size=args.batch,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"{model['name']} on {hardware[args.hw]['name']} ({args.quant}, ctx={args.ctx}, kv={args.kv})")
            print(
                f"Memory: {res['total_mem_gib']:.2f} / {res['usable_mem_gib']:.2f} GiB "
                f"[{res['fit_status']}] (weights={res['weights_gib']:.2f}, kv={res['kv_cache_gib']:.2f}, overhead={res['overhead_gib']:.2f})"
            )
            print(
                f"Speed:  decode={res['decode_tok_s']:.1f} tok/s | prefill={res['prefill_tok_s']:.1f} tok/s | "
                f"TTFT={res['ttft_s']:.2f}s | bottleneck={res['bottleneck']}"
            )
        return 0

    if args.command == "rank-hw":
        rows = [
            evaluate_pair(model, hw, quant=args.quant, ctx_tokens=args.ctx, kv_quant=args.kv)
            for hw in hardware.values()
        ]
        rows = [r for r in rows if r["fit_status"] != "oom"]
        rows.sort(key=lambda r: r["decode_tok_s"], reverse=True)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"Compatible hardware for {model['name']} ({args.quant}, ctx={args.ctx}):")
            for r in rows:
                print(
                    f"  {r['hardware']:<28} {r['fit_status']:<12} {r['total_mem_gib']:>6.1f}/{r['usable_mem_gib']:<6.1f} GiB  "
                    f"decode={r['decode_tok_s']:>6.1f} tok/s  TTFT={r['ttft_s']:>5.2f}s"
                )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
