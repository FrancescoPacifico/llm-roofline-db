#!/usr/bin/env python3
"""Refresh live Hugging Face metadata/configs and OpenRouter serverless pricing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any

USER_AGENT = "llm-roofline-db/1.0 (+https://llmroofline.dev)"


def _fetch_json(url: str, timeout: int = 12) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_openrouter_pricing() -> dict[str, tuple[float, float]]:
    """Return {openrouter_id: (prompt_usd_per_1m, completion_usd_per_1m)}."""
    payload = _fetch_json("https://openrouter.ai/api/v1/models", timeout=15)
    prices: dict[str, tuple[float, float]] = {}
    for entry in payload.get("data", []):
        model_id = entry.get("id")
        pricing = entry.get("pricing") or {}
        if not model_id:
            continue
        try:
            prompt_per_tok = float(pricing.get("prompt", 0) or 0)
            comp_per_tok = float(pricing.get("completion", 0) or 0)
            if prompt_per_tok > 0 or comp_per_tok > 0:
                prices[model_id] = (
                    round(prompt_per_tok * 1_000_000, 4),
                    round(comp_per_tok * 1_000_000, 4),
                )
        except (TypeError, ValueError):
            continue
    return prices


def refresh_models(seed_models: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enrich seed models with live HF downloads, config checks, and OpenRouter prices."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict[str, Any] = {
        "refreshed_at": timestamp,
        "openrouter_models_loaded": 0,
        "hf_metadata_ok": 0,
        "hf_config_verified": 0,
        "warnings": [],
    }

    try:
        or_prices = fetch_openrouter_pricing()
        report["openrouter_models_loaded"] = len(or_prices)
    except Exception as exc:  # noqa: BLE001 - preserve offline resilience
        or_prices = {}
        report["warnings"].append(f"OpenRouter fetch skipped: {exc}")

    enriched: list[dict[str, Any]] = []
    for raw in seed_models:
        item = dict(raw)
        item["verified_at"] = timestamp

        or_id = item.get("openrouter_id") or ""
        if or_id and or_id in or_prices:
            p_in, p_out = or_prices[or_id]
            item["api_prompt_usd_1m"] = p_in
            item["api_completion_usd_1m"] = p_out
            item["api_price_source"] = "openrouter_live"
        else:
            item["api_price_source"] = "seed_reference"

        hf_repo = item["hf_repo"]
        try:
            meta = _fetch_json(f"https://huggingface.co/api/models/{hf_repo}", timeout=10)
            if isinstance(meta.get("downloads"), int):
                item["hf_downloads"] = meta["downloads"]
                report["hf_metadata_ok"] += 1
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append(f"HF metadata skipped for {hf_repo}: {exc}")

        cfg_repo = item.get("hf_config_repo") or hf_repo
        try:
            cfg = _fetch_json(f"https://huggingface.co/{cfg_repo}/raw/main/config.json", timeout=10)
            tc = cfg.get("text_config", cfg)
            # Only cross-check non-NAS architectures where top-level keys represent active layers
            if tc.get("model_type") != "nemotron-nas":
                if tc.get("num_hidden_layers") and tc["num_hidden_layers"] != item["num_layers"]:
                    report["warnings"].append(
                        f"Layer mismatch {item['slug']}: seed={item['num_layers']} hf={tc['num_hidden_layers']}"
                    )
                if tc.get("hidden_size") and tc["hidden_size"] != item["hidden_size"]:
                    report["warnings"].append(
                        f"Hidden size mismatch {item['slug']}: seed={item['hidden_size']} hf={tc['hidden_size']}"
                    )
            report["hf_config_verified"] += 1
        except urllib.error.HTTPError as exc:
            if exc.code not in (401, 403):
                report["warnings"].append(f"HF config HTTP {exc.code} for {cfg_repo}")
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append(f"HF config skipped for {cfg_repo}: {exc}")

        enriched.append(item)

    return enriched, report


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    seed_models = json.loads((root / "data" / "seed" / "models_seed.json").read_text())
    live_dir = root / "data" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    enriched, report = refresh_models(seed_models)
    (live_dir / "models_live.json").write_text(json.dumps(enriched, indent=2) + "\n")
    (live_dir / "refresh_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"Refreshed {len(enriched)} models | OpenRouter live={report['openrouter_models_loaded']} | "
        f"HF meta={report['hf_metadata_ok']} | HF config={report['hf_config_verified']} | "
        f"warnings={len(report['warnings'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
