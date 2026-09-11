"""`python -m stability` = make stability：重复运行 -> 分析 -> 闸 C。

参数原样透传给 run_repeat（--runs / --limit / --ids）。
"""

from __future__ import annotations

import sys

from stability import analyze, gate, run_repeat


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    rc = run_repeat.main(args)
    if rc != 0:
        return rc
    rc = analyze.main([])
    if rc != 0:
        return rc
    return gate.main([])


if __name__ == "__main__":
    raise SystemExit(main())
