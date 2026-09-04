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
    lines = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert ".env" in lines, ".gitignore 必须排除 .env"


def test_env_example_has_no_value():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=" in text
    for line in text.splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            assert line.split("=", 1)[1].strip() == "", ".env.example 里不许填真实值"
