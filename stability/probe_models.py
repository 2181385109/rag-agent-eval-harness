"""模型归属探针：对配置里的每个模型名发一条最小请求，把服务端实际响应记成产物。

    python -m stability.probe_models
    -> stability/raw/model_probe_{ts}.json

为什么要有它：请求名（config.MODEL_NAME / JUDGE_MODEL_NAME）与响应 `model` 字段可能不同——
旧名会被路由到别的后端。被测链路的响应已逐行记在 run_*.jsonl 里；裁判链路 2026-09-11
那份记录没存响应字段。这个探针把"同一时刻、同一 key 下两个请求名各自落到哪个后端"
固化成文件，报告的「裁判独立性」一节从这里读，不靠口头断言。

只发 max_tokens 很小的请求，成本可忽略；不重跑任何评测。记录的响应头只取无敏感信息的几项。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from src import config
from stability.analyze import RAW_DIR

HEADER_KEYS = ("date", "server", "via", "x-ds-trace-id", "x-amz-cf-pop", "x-cache")


def probe(client: OpenAI, requested: str, max_tokens: int) -> dict:
    raw = client.chat.completions.with_raw_response.create(
        model=requested,
        messages=[{"role": "user", "content": "回复一个字：好"}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    resp = raw.parse()
    usage = resp.usage
    extra = (getattr(usage, "model_extra", None) or {}) if usage is not None else {}
    return {
        "requested_model": requested,
        "http_status": raw.status_code,
        "response_model": resp.model,
        "system_fingerprint": resp.system_fingerprint,
        "response_id": resp.id,
        "created": resp.created,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None) or extra.get("prompt_cache_hit_tokens"),
        "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None) or extra.get("prompt_cache_miss_tokens"),
        "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
        "headers": {k: raw.headers.get(k) for k in HEADER_KEYS if raw.headers.get(k) is not None},
    }


def run_probe(names: list[str], max_tokens: int = 16) -> dict:
    client = OpenAI(api_key=config.get_api_key(), base_url=config.BASE_URL, max_retries=0, timeout=60)
    results = [probe(client, name, max_tokens) for name in names]
    try:
        listed = sorted(m.id for m in client.models.list().data)
    except Exception as exc:  # 列表接口失败不影响探针本身
        listed = [f"<models.list failed: {type(exc).__name__}>"]
    return {
        "probed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": config.BASE_URL,
        "config": {"MODEL_NAME": config.MODEL_NAME, "JUDGE_MODEL_NAME": config.JUDGE_MODEL_NAME},
        "results": results,
        "models_listed_by_endpoint": listed,
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="模型归属探针")
    parser.add_argument("--out-dir", default=str(RAW_DIR))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    payload = run_probe([config.MODEL_NAME, config.JUDGE_MODEL_NAME])
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / f"model_probe_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in payload["results"]:
        print(f"{r['requested_model']:20s} -> model={r['response_model']}  fingerprint={r['system_fingerprint']}  cache_hit={r['prompt_cache_hit_tokens']}")
    print(f"endpoint lists: {payload['models_listed_by_endpoint']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
