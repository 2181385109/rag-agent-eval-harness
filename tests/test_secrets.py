"""密钥安全门禁（CLAUDE.md §1.7）：key 绝不能进仓库。

这条测试的价值在于它会一直跑下去——以后任何人不小心把 key 粘进代码，
提交前 pytest 就会红。
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,}")
SCAN_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".ini", ".txt", ".json", ".jsonl", ".cfg"}
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "node_modules"}


def _scannable_files():
    for path in PROJECT_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in SCAN_SUFFIXES:
            yield path


def test_no_api_key_literal_anywhere_in_repo():
    offenders = []
    for path in _scannable_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if KEY_PATTERN.search(text):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, f"这些文件里出现了疑似 API key：{offenders}"


def test_gitignore_excludes_env():
    """按行为断言而不是按字面量——规则怎么写无所谓，.env 进不去就行。"""
    assert _is_ignored(".env"), ".gitignore 必须排除 .env"


def test_env_example_has_no_value():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=" in text
    for line in text.splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            assert line.split("=", 1)[1].strip() == "", ".env.example 里不许填真实值"


# ---------------------------------------------------------------------------
# .gitignore 防线的回归测试
#
# 广谱规则（*api* / *key* / *secret* ...）威力大，副作用也大：它可能反过来
# 把源码静默吞掉——git add 不报错，文件就是进不去仓库。所以两个方向都要测：
#   1. 该挡的挡住了；2. 不该挡的没被误挡，尤其是 src/ 和 tests/ 下的真实文件。
# ---------------------------------------------------------------------------

import shutil
import subprocess

import pytest

_git_missing = shutil.which("git") is None
requires_git = pytest.mark.skipif(_git_missing, reason="环境里没有 git")

MUST_BE_IGNORED = [
    "api(Deepseek).txt",      # 带括号
    "my api key.txt",         # 带空格
    "API KEY 备份.txt",        # 大写 + 空格 + 中文
    "deepseek key(2).md",
    ".env",
    ".env.local",
    "src/.env",               # 白名单目录里的 .env 也要挡
    "src/api_key.txt",        # 白名单目录里的密钥文件也要挡
    "corpus/API KEY.txt",     # 语料目录放行 .txt，但名字带 API 仍要挡
    "notes secret.md",
    "DeepSeek Token.json",
    "my credential file",     # 无扩展名
    "server.pem",
    "id_rsa",
    "data/API(备份).jsonl",
    "深层/目录/里的 api key.txt",
]

MUST_NOT_BE_IGNORED = [
    "requirements.txt",
    ".env.example",
    "README.md",
    "pytest.ini",
    ".gitignore",
    "src/agent/rag.py",       # M2 会新建
    "src/agent/tools.py",     # M2 会新建
    "src/eval/metrics.py",    # M3 会新建
    "src/eval/judge.py",      # M5 会新建
    "tests/test_secrets.py",
    ".github/workflows/ci.yml",
    "corpus/机器学习基础.txt",
    "corpus/风控概念.md",
    "data/golden_capability.jsonl",
    "reports/2026-09-04.json",
]


def _is_ignored(path: str) -> bool:
    """--no-index：不看索引，纯按规则判定，这样已追踪的文件也能测出来。"""
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    return proc.returncode == 0


@requires_git
@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_secret_filenames_are_ignored(path):
    assert _is_ignored(path), f".gitignore 漏掉了疑似密钥文件：{path}"


@requires_git
@pytest.mark.parametrize("path", MUST_NOT_BE_IGNORED)
def test_legit_paths_are_not_ignored(path):
    assert not _is_ignored(path), f".gitignore 误挡了正常文件：{path}"


@requires_git
def test_gitignore_does_not_swallow_existing_source():
    """磁盘上真实存在的源码/测试/配置，一个都不许被 .gitignore 吞掉。

    这是广谱规则的安全网：以后谁新建了 src/agent/api_client.py 之类撞名的文件，
    这条测试会直接红，而不是让文件悄无声息地进不了仓库。
    """
    candidates = [
        p
        for d in ("src", "tests", ".github")
        for p in (PROJECT_ROOT / d).rglob("*")
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
    ]
    assert candidates, "没扫到任何源码文件，测试本身有问题"
    swallowed = [
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for p in candidates
        if _is_ignored(str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    ]
    assert not swallowed, (
        f".gitignore 把这些源码挡住了：{swallowed}\n"
        "要么改窄规则，要么在 .gitignore 第 4 段加一条 ! 豁免。"
    )


@requires_git
def test_no_tracked_file_looks_like_a_secret_file():
    """已经进了仓库的文件里，不许有名字像密钥的。"""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.split("\n")
    suspicious = [
        f
        for f in tracked
        if f.strip()
        and f != "tests/test_secrets.py"
        and re.search(r"(api|key|secret|token|credential|passwd|password)", f, re.I)
    ]
    assert not suspicious, f"这些已追踪文件名字像密钥：{suspicious}"


# ------------------------------------------------------------------ push 前安全扫描（CLAUDE.md §13.4）
def test_security_scan_has_no_unexempted_hits():
    """scripts/security_scan.py 在 git add -A 会带上的全部文件上零未豁免命中。

    豁免只认 EXEMPTIONS 里带出处的条目；本机用户名 / 主机名的字面命中不在这里断言
    （CI 机器上的用户名如 runner 会撞上正常词），由人在 push 前看扫描原文核对。
    """
    from scripts import security_scan

    result = security_scan.scan(history=False, identity=False)
    assert result.files, "扫描对象为空——git ls-files 没跑起来"
    assert result.env_ignored and not result.env_tracked
    offenders = [f"{h.category}: {h.path}:{h.line}: {h.token}" for h in result.violations]
    assert not offenders, "未豁免的安全扫描命中：\n" + "\n".join(offenders)


def test_security_scan_exemptions_all_carry_a_reason():
    from scripts import security_scan

    for category, items in security_scan.EXEMPTIONS.items():
        for token, reason in items.items():
            assert token and reason and len(reason) > 8, (category, token, reason)


def test_security_scan_catches_a_fresh_local_path_and_key():
    """扫描器本身要能抓到东西——否则零命中只是它瞎了。"""
    from scripts import security_scan

    # 用拼接构造样本，免得本测试文件自己被扫描器命中
    drive, bs = "D" + ":", "\\" * 2  # JSON 文本里反斜杠翻倍的写法
    text = f'x = "{drive}{bs}xiangmu{bs}rag-agent-eval-harness{bs}reports{bs}x.jsonl"\nk = "sk-' + "a" * 24 + '"\n'
    hits = security_scan.scan_text("fake.py", text)
    cats = {h.category for h in hits if h.is_violation}
    assert {"local_path", "api_key"} <= cats
    # 已知豁免的字符串命中但不算违规
    exempt = security_scan.scan_text("fake.json", f'a = "{drive}{bs}xiangmu{bs}credit-risk-mlops"')
    assert exempt and all(h.exempt_reason for h in exempt if h.category == "local_path")
