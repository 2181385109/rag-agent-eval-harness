"""push 前安全扫描（CLAUDE.md §13.4）：逐条打印命中原文，不做摘要。

    python -m scripts.security_scan              # 退出码 1 = 有未豁免的命中
    python -m scripts.security_scan --history    # 再扫全部历史提交里的 key 模式与 .env

扫描对象 = `git ls-files -co --exclude-standard`（已跟踪 + 未跟踪未忽略），即 `git add -A`
会带上的一切；.venv / index / .git 不在其中。覆盖：API key 模式与私钥块、本机绝对路径与
用户目录、邮箱、手机号、身份证号、`.env` 是否入库、本机用户名与主机名、（--history）
历史提交里的 key 模式与 .env。

已知豁免写在 EXEMPTIONS 里，每条带出处；命中豁免项照样打印（标「豁免」），
只是不计入失败。没有出处的豁免不许加。
"""

from __future__ import annotations

import argparse
import getpass
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ 模式
# (类别, 正则, 计入失败)
PATTERNS: tuple[tuple[str, re.Pattern, bool], ...] = (
    ("api_key", re.compile(r"sk-[A-Za-z0-9]{20,}"), True),
    ("api_key", re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password)\b\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}"), True),
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), True),
    ("api_key", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), True),
    ("api_key", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}"), True),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), True),
    # Windows 绝对路径（JSON 里反斜杠会翻倍，所以 \\+）；只吃 ASCII 路径字符，遇到反引号 / 中文即止
    ("local_path", re.compile(r"\b[A-Za-z]:\\+[A-Za-z0-9_.\-\\/]*"), True),
    ("local_path", re.compile(r"(?<![\w/])/(?:Users|home)/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*"), True),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), True),
    ("cn_mobile", re.compile(r"(?<![\w.])1[3-9]\d{9}(?![\w.])"), True),
    ("cn_id", re.compile(r"(?<![\w.])\d{17}[\dXx](?![\w.])"), True),
)

# ------------------------------------------------------------------ 已知豁免（每条必须带出处）
# token 比对前会把 JSON 的 \\ 归一成 \ 。
EXEMPTIONS: dict[str, dict[str, str]] = {
    "local_path": {
        # HANDOFF §三点五：cap_035 的一次真实 DeepSeek 调用逐字引用了语料原文；这 5 处快照 +
        # gate_traces 与 gate_judge_scores 的 answer_sha1 是一对冻结锚点，刻意保留。
        # 同一字符串也出现在 stability/raw/run_*.jsonl（同题）与 HANDOFF/LIMITATIONS 的脱敏对照表里。
        r"D:\xiangmu\credit-risk-mlops": "HANDOFF §三点五：语料原文被模型逐字引用，刻意保留（指纹锚点）",
        r"D:\xiangmu\flight-delay-scheduling": "HANDOFF §三点五：同上，出现在 gate_traces 的模型答案与脱敏对照表",
        r"D:\xiangmu": "HANDOFF §三点五：脱敏对照表里的原文（裸工作区路径）",
    },
    "backslash_relative": {
        # 09-06 快照里的三个 Windows 风格相对路径：ragas.reused_from（report*.md 照抄），以及
        # c95c4b5 脱敏时由绝对路径手改成的 backfill.source / judge_revision 来源。
        # 不是绝对路径、不含本机信息；写代码的一侧已改为 posix（report.repo_relative），旧快照不动。
        r"reports\eval_20260905T142507Z.json": "09-06 快照 ragas.reused_from 原样字符串（相对路径，非绝对）",
        r"reports\judge_scores.jsonl": "09-06 快照 backfill.source，c95c4b5 脱敏时手改的相对路径",
        r"reports\judge_scores_v1_conclusion_level.jsonl": "09-06 快照 judge_revision 来源，c95c4b5 脱敏时手改的相对路径",
    },
}

# 信息类：反斜杠相对路径本身不算泄露，只为把上面那条豁免落到扫描里，避免以后反复误报
INFO_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("backslash_relative", re.compile(r"(?<![A-Za-z]:)(?<![\\/\w])(?:reports|stability|data|tests)\\+[A-Za-z0-9_.\-\\]+")),
)


@dataclass
class Hit:
    category: str
    path: str
    line: int
    token: str
    exempt_reason: str | None = None
    fails: bool = True

    @property
    def is_violation(self) -> bool:
        return self.fails and self.exempt_reason is None


@dataclass
class ScanResult:
    files: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    env_tracked: bool | None = None
    env_ignored: bool | None = None
    identity_hits: list[Hit] = field(default_factory=list)
    history: dict | None = None

    @property
    def violations(self) -> list[Hit]:
        out = [h for h in self.hits if h.is_violation]
        if self.env_tracked:
            out.append(Hit("env_tracked", ".env", 0, ".env 已入库"))
        if self.history and (self.history.get("key_hits") or self.history.get("env_commits")):
            out.append(Hit("history", "<git history>", 0, "历史提交里有 key 模式或 .env"))
        return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def scannable_files() -> list[str]:
    files = [f for f in _git("ls-files", "-co", "--exclude-standard").split("\n") if f]
    return [f for f in files if (PROJECT_ROOT / f).is_file()]


def _normalize(token: str) -> str:
    return token.replace("\\\\", "\\").rstrip("\\/.")


def _exemption(category: str, token: str) -> str | None:
    return EXEMPTIONS.get(category, {}).get(_normalize(token))


def scan_text(path: str, text: str) -> list[Hit]:
    hits: list[Hit] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        for category, pat, fails in PATTERNS:
            for m in pat.finditer(line):
                token = m.group(0)
                hits.append(Hit(category, path, lineno, token, _exemption(category, token), fails))
        for category, pat in INFO_PATTERNS:
            for m in pat.finditer(line):
                token = m.group(0)
                hits.append(Hit(category, path, lineno, token, _exemption(category, token), fails=False))
    return hits


def scan(*, history: bool = False, identity: bool = True) -> ScanResult:
    result = ScanResult(files=scannable_files())
    for f in result.files:
        raw = (PROJECT_ROOT / f).read_bytes()
        if b"\x00" in raw[:4096]:
            continue  # 二进制
        text = raw.decode("utf-8", errors="replace")
        result.hits.extend(scan_text(f, text))
        if identity:
            result.identity_hits.extend(_identity_hits(f, text))
    result.env_tracked = bool(_git("ls-files", "--", ".env").strip())
    result.env_ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=PROJECT_ROOT).returncode == 0
    if history:
        result.history = scan_history()
    return result


def _identity_hits(path: str, text: str) -> list[Hit]:
    """本机用户名 / 主机名的字面命中。用户名可能是短数字串（会撞上 token 计数之类的数字），
    所以只打印供人工核对，不自动判失败；主机名足够特异，命中即失败。"""
    hits: list[Hit] = []
    user, host = getpass.getuser(), socket.gethostname()
    for lineno, line in enumerate(text.split("\n"), start=1):
        if host and host.lower() in line.lower():
            hits.append(Hit("hostname", path, lineno, host, None, fails=True))
        if user and re.search(rf"(?<![\w.]){re.escape(user)}(?![\w.])", line, re.IGNORECASE):
            hits.append(Hit("username", path, lineno, user, None, fails=False))
    return hits


def scan_history() -> dict:
    """全部提交里的 key 模式（git grep 跨所有 rev）与 .env 是否进过任何一次提交。"""
    revs = [r for r in _git("rev-list", "--all").split("\n") if r]
    key_hits: list[str] = []
    for i in range(0, len(revs), 50):
        out = subprocess.run(
            ["git", "grep", "-I", "-n", "-E", r"sk-[A-Za-z0-9]{20,}", *revs[i:i + 50]],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout
        key_hits.extend(l for l in out.split("\n") if l)
    env_commits = [l for l in _git("log", "--all", "--full-history", "--oneline", "--", ".env").split("\n") if l]
    return {"n_commits": len(revs), "key_hits": key_hits, "env_commits": env_commits}


def render(result: ScanResult) -> str:
    L: list[str] = [f"扫描对象：{len(result.files)} 个文件（git ls-files -co --exclude-standard）", ""]
    by_cat: dict[str, list[Hit]] = {}
    for h in result.hits:
        by_cat.setdefault(h.category, []).append(h)
    order = ["api_key", "private_key", "local_path", "email", "cn_mobile", "cn_id", "backslash_relative"]
    for cat in order:
        hs = by_cat.get(cat, [])
        n_v = sum(1 for h in hs if h.is_violation)
        n_e = sum(1 for h in hs if h.exempt_reason)
        tag = "计入失败" if cat not in ("backslash_relative",) else "信息类"
        L.append(f"## {cat}（{tag}）：命中 {len(hs)}，其中豁免 {n_e}，未豁免 {n_v}")
        for h in hs:
            mark = f"[豁免] {h.exempt_reason}" if h.exempt_reason else ("[命中]" if h.fails else "[信息]")
            L.append(f"  {h.path}:{h.line}: {h.token}    {mark}")
        L.append("")
    L.append(f"## .env：已入库={result.env_tracked}，被 gitignore 忽略={result.env_ignored}")
    L.append("")
    ident = result.identity_hits
    L.append(f"## 本机标识符（用户名 `{getpass.getuser()}`：信息类，需人工核对；主机名 `{socket.gethostname()}`：计入失败）：命中 {len(ident)}")
    for h in ident:
        L.append(f"  {h.path}:{h.line}: {h.category}={h.token}    {'[命中]' if h.fails else '[信息]'}")
    L.append("")
    if result.history is not None:
        hst = result.history
        L.append(f"## git 历史（{hst['n_commits']} 个提交）：key 模式命中 {len(hst['key_hits'])}；.env 进过的提交 {len(hst['env_commits'])}")
        for l in hst["key_hits"]:
            L.append(f"  {l}")
        for l in hst["env_commits"]:
            L.append(f"  {l}")
        L.append("")
    v = result.violations
    L.append(f"## 结论：未豁免命中 {len(v)} 条 -> {'FAIL' if v else 'PASS'}")
    for h in v:
        L.append(f"  {h.category}: {h.path}:{h.line}: {h.token}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="push 前安全扫描（CLAUDE.md §13.4）")
    parser.add_argument("--history", action="store_true", help="再扫全部历史提交")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    result = scan(history=args.history)
    print(render(result))
    return 1 if result.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
