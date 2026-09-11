"""给重复运行里的**开放题**逐次打裁判分，让判定自洽率能覆盖开放题。

    python -m stability.judge_runs --raw stability/raw/run_20260912T010203Z.jsonl
    -> stability/raw/judge_20260912T020304Z.jsonl   一行一次 (题, 次)

为什么要逐次判：主评测的裁判分是对**那一次**答案打的；重复运行里 k 个答案各不相同，
把同一个分套到 k 次上就是张冠李戴（src/eval/metrics.consistency 里那条注释说的就是这个）。
所以这里对每个 (sample_id, run_index) 单独调裁判，并记下被打分答案的指纹
answer_sha1——analyze 回填时逐条核对，对不上的分作废。

裁判与判据完全复用 src/eval/judge（同一个 deepseek-reasoner、同一版要点级判据 v2），
不另起一套口径。成本：开放题 10 道 × k=5 = 50 次 reasoner 调用。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.eval import judge
from src.eval.datasets import load_golden_set
from stability.analyze import RAW_DIR, load_rows


def score_rows(rows, samples, *, call=None, progress=True) -> list[dict]:
    caller = call if call is not None else judge.DeepSeekJudge()
    model = getattr(caller, "model", config.JUDGE_MODEL_NAME)
    by_id = {s.id: s for s in samples}
    targets = [r for r in rows if r.answer_type == "open" and r.ok and r.sample_id in by_id]

    out: list[dict] = []
    for i, r in enumerate(targets, start=1):
        sample = by_id[r.sample_id]
        messages = judge.build_judge_messages(sample, r.answer)
        retried = False
        try:
            reply = caller(messages, model=model, sample_id=r.sample_id)
            score, reason = judge.parse_judge_reply(reply)
        except ValueError:
            retried = True
            reply = caller(messages, model=model, sample_id=r.sample_id)
            score, reason = judge.parse_judge_reply(reply)
        out.append(
            {
                "sample_id": r.sample_id,
                "run_index": r.run_index,
                "answer_sha1": r.answer_sha1,
                "judge_score": score,
                "reason": reason,
                "retried": retried,
                "judge_model": model,
                "rubric_version": judge.RUBRIC_VERSION,
                "judge_params": dict(getattr(caller, "params", {})),
                "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        if progress:
            print(f"  [{i}/{len(targets)}] {r.sample_id} pass {r.run_index} -> {score}", file=sys.stderr, flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="重复运行的开放题逐次裁判打分")
    parser.add_argument("--raw", type=str, required=True, help="run_*.jsonl")
    parser.add_argument("--out-dir", type=str, default=str(RAW_DIR))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    rows = load_rows(args.raw)
    samples = load_golden_set()
    n_open = sum(1 for r in rows if r.answer_type == "open" and r.ok)
    print(f"开放题运行 {n_open} 次，裁判 {config.JUDGE_MODEL_NAME}（判据 {judge.RUBRIC_VERSION}）", file=sys.stderr)

    scored = score_rows(rows, samples)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / f"judge_{ts}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"完成：{len(scored)} 行 -> {out}", file=sys.stderr)
    print(f"下一步：python -m stability.analyze --raw {args.raw} --judge {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
