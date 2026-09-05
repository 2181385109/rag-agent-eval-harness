"""依赖锁版本的门禁（CLAUDE.md §2：依赖用 requirements.txt 锁版本）。

锁了不等于守得住：M4 装 ragas 时 pip 就把 openai 从 3.8.0 悄悄降到了 2.54.0。
这条测试让"清单写的"和"实际装的"对不上时立刻变红，而不是等某天行为诡异了才发现。
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.\-]+)==(.+)$")


def _pinned() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.split("#")[0].strip()
        match = PIN_PATTERN.match(stripped)
        if match:
            pins[match.group(1)] = match.group(2).strip()
    return pins


def test_requirements_pins_every_dependency():
    """不许出现 >=、~= 之类的浮动约束——评测结果要可复现。"""
    floating = []
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.split("#")[0].strip()
        if stripped and not PIN_PATTERN.match(stripped):
            floating.append(stripped)
    assert not floating, f"这些依赖没有锁死版本：{floating}"


def test_requirements_is_not_empty():
    assert len(_pinned()) >= 10


@pytest.mark.parametrize("package", sorted(_pinned()))
def test_installed_version_matches_the_pin(package):
    expected = _pinned()[package]
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError:
        pytest.fail(f"{package} 写在 requirements.txt 里却没装")
    assert actual == expected, (
        f"{package}: requirements.txt 锁 {expected}，实际装的是 {actual}。"
        "要么更新清单并说明原因，要么把环境装回锁定版本。"
    )
