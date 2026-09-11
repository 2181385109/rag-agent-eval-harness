# PERF_SPEC task B. On Windows (no make) run the python commands directly.
PY ?= python

.PHONY: stability stability-run stability-judge stability-analyze stability-gate gate test

# Full pipeline: 36 questions x k=5 (real DeepSeek calls), per-run judge for open
# questions (deepseek-reasoner), analysis, gate C. Costs money; ~30 min.
stability: stability-run stability-judge stability-analyze stability-gate

stability-run:
	$(PY) -m stability.run_repeat

stability-judge:
	$(PY) -m stability.judge_runs --raw $$(ls stability/raw/run_*.jsonl | tail -1)

stability-analyze:
	$(PY) -m stability.analyze --raw $$(ls stability/raw/run_*.jsonl | tail -1) --judge $$(ls stability/raw/judge_*.jsonl | tail -1)

stability-gate:
	$(PY) -m stability.gate

# All three offline gates (A, B, C) -- same command CI runs.
gate:
	$(PY) -m src.eval.report --gate

test:
	$(PY) -m pytest -m "not live"
