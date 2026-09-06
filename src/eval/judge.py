"""LLM-as-Judge 与人工标注对照（CLAUDE.md §9 M5）。

本模块的第一步产物是**人工标注表**：闭合题已由规则判定，开放题需要人来打分，
再和自动裁判对照算 Cohen's kappa。

一条设计纪律：**标注表绝不写入自动裁判的结论**。人先看到机器打的分再标，
kappa 度量的就不是"人机是否独立地看法一致"，而是确认偏误。所以
build_labeling_sheet 显式忽略任何传进来的裁判分数。

命令行：
    python -m src.eval.judge --sheet     # 生成待人工标注的表 + 空白骨架
    python -m src.eval.judge --score     # 自动裁判打分（调 API），落盘 reports/judge_scores.jsonl
    python -m src.eval.judge --kappa     # 读回两边分数算一致率（纯离线，不花钱）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

from src import config
from src.agent.trace import AgentTrace
from src.eval.datasets import GoldenSample, load_golden_set

# 三级有序标度。为什么不用二值：开放题常见"方向对但要点有遗漏"，
# 二值会把它硬塞进对或错，人机分歧被人为放大。
#
# ---- 判据修订（2026-09-06，v1 结论级 -> v2 要点级）-------------------------
# v1 的 1/2 档写的是"要点有遗漏"/"要点齐全"，但没说清**什么算一个要点**。
# 结果人和机器各按各的理解办：M5 的分歧分析显示裁判比对的是**结论层**
# （三次的理由句都是"与参考答案一致"），人工比对的是**要点层**
# （把参考答案拆成具体限定逐条点数）。cap_017/032/035 三题人判 1、机判 2，
# 全是这个模式——不是随机噪声，是两把尺子的刻度不同，也是 kappa=0.216 的主因。
#
# v2 把"要点"定死成「参考答案里的每一项具体限定」，并写明缺一项即降档。
# **注意这是让裁判向人工的既有尺子靠拢**（人工本来就是这么标的，见
# data/agreement_interpretation.md，那份解读写在本次修订之前）。
# 所以 kappa 若上升，有一部分是构造性的，不是两个独立评分者自发趋同——
# 这一点必须在报告里说明，不能当成"评测可信度提高了"来卖。
RUBRIC_VERSION = "v2-要点级"

LABEL_SCALE: dict[int, str] = {
    0: "错误 / 编造 / 答非所问：与参考答案矛盾，或凭空捏造了知识库里没有的内容",
    1: "部分正确：结论方向对、无编造，但参考答案里的具体限定至少缺了一项，或有轻微不准确",
    2: "正确且有据：参考答案里的每一项具体限定都在 Agent 答案中出现，结论一致，且未编造",
}

# 打分细则。和 LABEL_SCALE 一样是**人机同源**的：标注表和裁判提示词都从这里拼，
# 各写一份必然漂移——v1 的教训就是判据留了解释空间，两边就各解释各的。
RUBRIC_NOTES: tuple[str, ...] = (
    "判的是「Agent 答案」相对「参考答案」的质量，不是判问题出得好不好，"
    "也不是判答案写得漂不漂亮。篇幅长、排版好不构成加分理由。",
    "**先把参考答案拆成一份「具体限定」清单，再逐项核对。**"
    "数字与阈值、比例、公式或指标定义、方法与算法名、明确的禁止或例外，各算一项。",
    "**缺一项即降到 1 分。**「结论方向一致」不足以给 2 分——"
    "2 分要求这份清单逐项都能在 Agent 答案里找到。",
    "标注为「幻觉诱饵题」的题目，正确行为是明说「知识库里没有」。"
    "如果 Agent 编出了一个看似合理的答案，那是 0 分，不是 2 分。",
)

SHEET_FILENAME = "human_labeling_sheet.md"


def _open_samples(samples: Sequence[GoldenSample]) -> list[GoldenSample]:
    """只有开放题需要人工标注；闭合题已由 answer_keys 规则判定。"""
    return [s for s in samples if s.answer_type == "open"]


def build_labeling_sheet(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    judge_scores: Mapping[str, int] | None = None,  # 刻意忽略，见模块 docstring
    include_chunk_text: bool = False,
) -> str:
    """生成给人看的标注表（Markdown）。"""
    del judge_scores  # 绝不写进表里：人必须独立于机器判定

    targets = _open_samples(samples)
    lines = [
        "# 人工标注表（开放题）",
        "",
        f"共 **{len(targets)}** 道开放题。闭合题已由 `answer_keys` 规则判定，不需要人工打分。",
        "",
        "## 怎么标",
        "",
        "给每道题的 **Agent 答案**打一个分，判据如下：",
        "",
        "| 分数 | 判据 |",
        "|---|---|",
    ]
    for level, description in LABEL_SCALE.items():
        lines.append(f"| **{level}** | {description} |")

    lines += [
        "",
        "打分细则（与自动裁判的提示词**同一段字**，见 `judge.RUBRIC_NOTES`）：",
        "",
    ]
    lines += [f"- {note}" for note in RUBRIC_NOTES]
    lines += [
        "",
        "标注流程上的两点：",
        "",
        "- 请**先标完再看自动裁判的结果**。先看机器分再标，"
        "kappa 度量的就不是人机独立看法是否一致，而是确认偏误。",
        f"- 样本量只有 {len(targets)} 条，算出来的 kappa 会很不稳定"
        "（置信区间宽），报告里会如实标注这一点，不要把它当硬结论。",
        "",
        "标完把分数告诉我（形如 `cap_007=2, cap_008=1, ...`），"
        f"或直接编辑 `{config.HUMAN_LABELS_PATH.name}` 里的 `human_score` 字段。",
        "",
        "---",
        "",
        "## 逐题",
        "",
    ]

    for index, sample in enumerate(targets, start=1):
        trace = traces.get(sample.id)
        answer = (trace.answer if trace else "") or "（本轮没有产出答案）"
        # 去重保序：Agent 多次检索时同一块可能重复出现
        seen_chunks: dict[str, object] = {}
        if trace:
            for chunk in trace.retrieved_chunks:
                seen_chunks.setdefault(chunk.doc_id, chunk)
        sources = [f"{c.doc_id}  {c.source}" for c in seen_chunks.values()]

        lines += [
            f"### {index}. `{sample.id}`"
            + (f"  ⚠ `{sample.failure_tag}`" if sample.failure_tag else ""),
            "",
            f"**问题**：{sample.question}",
            "",
            "**参考答案**：",
            "",
            f"> {sample.reference_answer}",
            "",
            "**Agent 答案**：",
            "",
        ]
        lines += [f"> {line}" if line.strip() else ">" for line in answer.splitlines()]
        lines += [
            "",
            f"<details><summary>检索到的片段（{len(sources)} 块）</summary>",
            "",
        ]
        if not seen_chunks:
            lines.append("- （未检索）")
        elif include_chunk_text:
            for chunk in seen_chunks.values():
                lines += [
                    f"**`{chunk.doc_id}`**  ·  {chunk.source}",
                    "",
                ]
                lines += [
                    f"> {ln}" if ln.strip() else ">" for ln in chunk.text.splitlines()
                ]
                lines.append("")
        else:
            lines += [f"- `{item}`" for item in sources]
        lines += [
            "",
            "</details>",
            "",
            "**我的评分**： `___`   （0 / 1 / 2）",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def write_labeling_sheet(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    path: Path | None = None,
    include_chunk_text: bool = False,
) -> Path:
    target = path or (config.REPORTS_DIR / SHEET_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_labeling_sheet(samples, traces, include_chunk_text=include_chunk_text),
        encoding="utf-8",
    )
    return target


def write_label_skeleton(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    path: Path | None = None,
) -> Path:
    """生成 human_labels.jsonl 骨架，human_score 留空等人填。

    已经填过的分数不会被覆盖——重新生成骨架不应该把人的劳动成果抹掉。
    """
    target = path or config.HUMAN_LABELS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, int | None] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row.get("human_score")

    with target.open("w", encoding="utf-8") as fh:
        for sample in _open_samples(samples):
            trace = traces.get(sample.id)
            payload = {
                "id": sample.id,
                "question": sample.question,
                "reference_answer": sample.reference_answer,
                "answer": (trace.answer if trace else "") or "",
                "failure_tag": sample.failure_tag,
                "human_score": existing.get(sample.id),
                "note": "",
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


def load_human_labels(path: Path | None = None) -> dict[str, int]:
    """读回已标注的分数，未标注的（human_score 为 null）跳过。"""
    target = Path(path) if path is not None else config.HUMAN_LABELS_PATH
    if not target.exists():
        raise FileNotFoundError(f"人工标注文件不存在：{target}")

    labels: dict[str, int] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("human_score")
        if score is None:
            continue
        if score not in LABEL_SCALE:
            raise ValueError(f"{row['id']} 的 human_score={score} 非法，只能是 0/1/2")
        labels[row["id"]] = int(score)
    return labels


# ===========================================================================
# 自动裁判（LLM-as-Judge）
#
# 三条刻意的设计，都由 tests/test_judge.py 钉住：
#   1. 判据文本与人工标注表**同源**（都从 LABEL_SCALE 拼），不许各写一份；
#   2. 提示词里**不给检索片段**——人标注时看的就是问题+参考答案+Agent答案，
#      输入不对等，kappa 就不再是"同一份材料下看法是否一致"；
#   3. 打分路径**不读** human_labels.jsonl。独立性要两个方向都锁死：
#      标注表不给人看机器分，裁判也不许看人的分。
# ===========================================================================

JUDGE_SCORES_FILENAME = "judge_scores.jsonl"

# 类别集固定为 [0, 1, 2]，绝不让 sklearn 从数据里推断（见 _kappa_pair 的注释）。
LABEL_ORDER = sorted(LABEL_SCALE)

# bootstrap 置信区间的参数。固定种子 = 同一份分数重算得同一个区间（可复现红线）。
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260906

BAIT_TAG = "hallucination_bait"


def _plain(text: str) -> str:
    """去掉 Markdown 强调星号。

    判据正文人机同源，但标注表是 Markdown、提示词是纯文本；
    排版壳子不该混进给模型看的那一份。
    """
    return text.replace("**", "")


def build_judge_system_prompt() -> str:
    """裁判的 system 提示词。

    判据三行由 LABEL_SCALE 拼、细则由 RUBRIC_NOTES 拼，都与标注表同源（测试钉住）。
    v2 起还写死了打分步骤：**先把参考答案拆成具体限定清单，再逐项核对**。
    v1 只说"要点齐全"，没说什么算一个要点，裁判就把它读成了结论层比对。
    """
    rubric = "\n".join(f"{level} = {_plain(text)}" for level, text in LABEL_SCALE.items())
    notes = "\n".join(f"- {_plain(note)}" for note in RUBRIC_NOTES)
    return (
        "你是一名严格的评测标注员。你的任务是给一个 RAG Agent 的回答打分。\n\n"
        "评分标度（三级，必须严格按此判据）：\n"
        f"{rubric}\n\n"
        "打分细则：\n"
        f"{notes}\n"
        f"- 「幻觉诱饵题」会在题面上标出 {BAIT_TAG}。\n"
        "- 你看不到 Agent 检索到的原文片段，这是刻意的。请只依据参考答案判断。\n\n"
        "打分步骤（必须按此顺序执行）：\n"
        "1. 先从【参考答案】里列出全部具体限定，逐条编号。\n"
        "2. 再逐条到【Agent 答案】里找，标记命中还是缺失。\n"
        "3. 有矛盾或编造判 0；逐条全部命中判 2；否则判 1。\n\n"
        "输出格式：仅输出一个 JSON 对象，不要有任何额外文字。\n"
        '{"score": 0 或 1 或 2, "reason": "50 字以内说明；若降档必须点名缺了哪一项"}'
    )


def build_judge_messages(sample: GoldenSample, answer: str) -> list[dict[str, str]]:
    """逐题一次独立调用：不批量塞进同一上下文，避免顺序效应与前题污染后题。"""
    header = ""
    if sample.failure_tag == BAIT_TAG:
        # 人标注时表里写着 ⚠ hallucination_bait，裁判也得看见，输入才对等。
        header = f"【题目标记】幻觉诱饵题（{BAIT_TAG}）\n\n"
    user = (
        f"{header}"
        f"【问题】\n{sample.question}\n\n"
        f"【参考答案】\n{sample.reference_answer}\n\n"
        f"【Agent 答案】\n{answer or '（本轮没有产出答案）'}"
    )
    return [
        {"role": "system", "content": build_judge_system_prompt()},
        {"role": "user", "content": user},
    ]


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def parse_judge_reply(text: str) -> tuple[int, str]:
    """从裁判回复里抠出 {score, reason}。

    解析不出来就抛错，**绝不静默兜底成某个分数**——那等于凭空造一个指标
    （CLAUDE.md §1.1）。上层负责重问一次，再失败就让整次运行失败。
    """
    raw = (text or "").strip()
    payload = None
    for candidate in (raw, *(m.group(1) for m in _FENCE_RE.finditer(raw))):
        try:
            payload = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    if not isinstance(payload, dict):
        match = _OBJECT_RE.search(raw)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        raise ValueError(f"裁判输出无法解析成 JSON 对象：{raw[:200]!r}")

    score = payload.get("score")
    if isinstance(score, str) and score.strip().lstrip("-").isdigit():
        score = int(score.strip())
    if isinstance(score, bool) or not isinstance(score, int) or score not in LABEL_SCALE:
        raise ValueError(f"裁判给出的 score 非法（只能是 0/1/2）：{payload!r}")

    reason = str(payload.get("reason") or "").strip()
    return int(score), reason


class DeepSeekJudge:
    """默认裁判调用器：走 config.JUDGE_MODEL_NAME（刻意与被测模型不同源）。

    deepseek-reasoner 对 `response_format` 与 `temperature` 的支持情况按官方文档
    会变，所以这里**逐级降档试**，并把最终真正生效的参数记在 self.params 里，
    由报告 meta 如实写出——不能提示词里写着 temperature=0、实际被忽略却不说。
    """

    def __init__(self, model: str | None = None, max_tokens: int | None = None,
                 timeout: float | None = None):
        self.model = model or config.JUDGE_MODEL_NAME
        self.max_tokens = max_tokens or config.JUDGE_MAX_TOKENS
        self.timeout = timeout if timeout is not None else config.JUDGE_TIMEOUT_S
        self.params: dict[str, object] = {
            "model": self.model,
            "json_mode": None,
            "temperature_sent": None,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout,
        }
        self._plan: list[dict] | None = None

    def _attempts(self) -> list[dict]:
        return [
            {"temperature": 0.0, "response_format": {"type": "json_object"}},
            {"temperature": 0.0},
            {},
        ]

    def __call__(self, messages, *, model: str | None = None, sample_id: str | None = None,
                 **_ignored) -> str:
        from openai import BadRequestError

        from src.agent import llm

        client = llm.get_client()
        plan = self._plan if self._plan is not None else self._attempts()
        last_error: Exception | None = None
        for extra in plan:
            try:
                resp = client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    **extra,
                )
            except BadRequestError as exc:  # 该档参数不被支持，降一档再试
                last_error = exc
                continue
            # 记住哪一档能用，后续题目直接走它，不再每题重试一遍
            self._plan = [extra]
            self.params["json_mode"] = "response_format" in extra
            self.params["temperature_sent"] = extra.get("temperature")
            content = resp.choices[0].message.content
            return content or ""
        raise RuntimeError(f"裁判调用全部参数组合都被拒绝（{sample_id}）：{last_error}")


def answer_fingerprint(answer: str) -> str:
    """被打分的那段答案的指纹，写进打分行。

    裁判分是**对某一批轨迹的答案**打的。重跑一次 Agent，答案就变了，
    旧分再套上去等于拿甲的答卷给乙判分——而任务成功率现在正是靠这些分
    回填开放题的，一次静默错配就能凭空造出一个指标。
    存下指纹，回填时逐条核对，对不上就不算数（见 report.load_judge_backfill）。

    只截 12 位十六进制：这是防错配，不是防篡改，够用且让打分文件保持可读。
    """
    return hashlib.sha1((answer or "").encode("utf-8")).hexdigest()[:12]


def score_samples(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    *,
    call=None,
) -> list[dict]:
    """让自动裁判给开放题打分。返回逐题结果，不落盘。

    只碰 samples 与 traces，**不读人工标注**（测试钉住）。
    """
    caller = call if call is not None else DeepSeekJudge()
    model = getattr(caller, "model", config.JUDGE_MODEL_NAME)

    rows: list[dict] = []
    for sample in _open_samples(samples):
        trace = traces.get(sample.id)
        answer = (trace.answer if trace else "") or ""
        messages = build_judge_messages(sample, answer)

        retried = False
        try:
            reply = caller(messages, model=model, sample_id=sample.id)
            score, reason = parse_judge_reply(reply)
        except ValueError:
            # 重问一次：原样重发。不追加"你格式错了"的纠正轮——
            # reasoner 对连续 user 消息与回填 assistant 内容都有限制，
            # 原样重发最稳，且不改变裁判看到的材料。
            retried = True
            reply = caller(messages, model=model, sample_id=sample.id)
            score, reason = parse_judge_reply(reply)  # 再失败就让它抛出去

        rows.append(
            {
                "id": sample.id,
                "judge_score": score,
                "reason": reason,
                "retried": retried,
                "failure_tag": sample.failure_tag,
                "judge_model": model,
                "rubric_version": RUBRIC_VERSION,
                "answer_sha1": answer_fingerprint(answer),
            }
        )
    return rows


def write_judge_scores(rows: Sequence[Mapping], path: Path | None = None) -> Path:
    target = Path(path) if path is not None else (config.REPORTS_DIR / JUDGE_SCORES_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return target


def load_judge_rows(path: Path | None = None) -> list[dict]:
    """读回裁判打分的**完整行**（含指纹与理由），不只是分数。

    任务成功率回填要用到 answer_sha1 来核对"这个分是不是给这批轨迹打的"，
    所以需要整行；只要分数的场景继续用 load_judge_scores。
    """
    target = Path(path) if path is not None else (config.REPORTS_DIR / JUDGE_SCORES_FILENAME)
    if not target.exists():
        raise FileNotFoundError(f"裁判打分文件不存在：{target}")
    rows: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("judge_score")
        if score is not None and score not in LABEL_SCALE:
            raise ValueError(f"{row['id']} 的 judge_score={score} 非法，只能是 0/1/2")
        rows.append(row)
    return rows


def load_judge_scores(path: Path | None = None) -> dict[str, int]:
    return {
        row["id"]: int(row["judge_score"])
        for row in load_judge_rows(path)
        if row.get("judge_score") is not None
    }


# ------------------------------------------------------------ 一致率 / kappa

def _kappa_pair(human: Sequence[int], auto: Sequence[int]) -> tuple[float | None, float | None]:
    """算两种口径的 kappa；任一方无变异则返回 (None, None)（数学上无定义）。"""
    from sklearn.metrics import cohen_kappa_score

    if len(set(human)) < 2 or len(set(auto)) < 2:
        return None, None
    # labels 必须显式固定：让 sklearn 从数据推断类别，某次裁判恰好没打过 0 分
    # 就会换一套类别集，两次运行的 kappa 不可比。
    plain = float(cohen_kappa_score(human, auto, labels=LABEL_ORDER))
    quad = float(cohen_kappa_score(human, auto, labels=LABEL_ORDER, weights="quadratic"))
    return plain, quad


def agreement_stats(
    human: Mapping[str, int],
    auto: Mapping[str, int],
    *,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """自动裁判 ↔ 人工标注的一致性。

    只在**两边都有分**的题上算，缺谁都如实列出（分母陷阱：RAGAS 那次
    NaN 被 mean() 静默跳过，差点报出一个不可比的漂亮数字）。
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    common = sorted(set(human) & set(auto))
    h = [int(human[i]) for i in common]
    a = [int(auto[i]) for i in common]
    n = len(common)

    exact = sum(x == y for x, y in zip(h, a)) / n if n else None
    adjacent = sum(abs(x - y) <= 1 for x, y in zip(h, a)) / n if n else None
    matrix = (
        confusion_matrix(h, a, labels=LABEL_ORDER).tolist()
        if n
        else [[0] * len(LABEL_ORDER) for _ in LABEL_ORDER]
    )

    kappa, kappa_q = _kappa_pair(h, a) if n else (None, None)
    degenerate = kappa is None
    note = ""
    if degenerate:
        which = []
        if len(set(h)) < 2:
            which.append("人工")
        if len(set(a)) < 2:
            which.append("裁判")
        note = (
            f"未定义：{'与'.join(which) or '某一方'}标注无变异（全打同一档），"
            "机遇一致率也等于 1，kappa 的分母为 0。"
            "此处**不落成 0.0**——0.0 读起来像「完全不一致」，那是编造。"
        )

    ci = ci_q = None
    used = dropped = 0
    if bootstrap and not degenerate and n:
        rng = np.random.default_rng(seed)
        plain_samples: list[float] = []
        quad_samples: list[float] = []
        for _ in range(bootstrap):
            idx = rng.integers(0, n, size=n)
            bh = [h[i] for i in idx]
            ba = [a[i] for i in idx]
            k, kq = _kappa_pair(bh, ba)
            if k is None:
                dropped += 1
                continue
            used += 1
            plain_samples.append(k)
            quad_samples.append(kq)
        if plain_samples:
            ci = [
                float(np.percentile(plain_samples, 2.5)),
                float(np.percentile(plain_samples, 97.5)),
            ]
            ci_q = [
                float(np.percentile(quad_samples, 2.5)),
                float(np.percentile(quad_samples, 97.5)),
            ]

    disagreements = [
        {"id": i, "human": human[i], "judge": auto[i], "gap": abs(human[i] - auto[i])}
        for i in common
        if human[i] != auto[i]
    ]

    return {
        "n": n,
        "labels": list(LABEL_ORDER),
        "exact_agreement": exact,
        "adjacent_agreement": adjacent,
        "confusion_matrix": matrix,
        "kappa": kappa,
        "kappa_quadratic": kappa_q,
        "kappa_ci": ci,
        "kappa_quadratic_ci": ci_q,
        "degenerate": degenerate,
        "kappa_note": note,
        "bootstrap_n": bootstrap,
        "bootstrap_used": used,
        "bootstrap_degenerate": dropped,
        "bootstrap_seed": seed,
        "missing_judge": sorted(set(human) - set(auto)),
        "missing_human": sorted(set(auto) - set(human)),
        "disagreements": disagreements,
        "human_distribution": {str(l): h.count(l) for l in LABEL_ORDER},
        "judge_distribution": {str(l): a.count(l) for l in LABEL_ORDER},
    }


def _fmt3(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def load_agreement_interpretation(path: Path | None = None) -> str:
    """读回人写的解读。没有就返回空串——不自动生成叙述，也不假装有。"""
    target = Path(path) if path is not None else config.AGREEMENT_INTERPRETATION_PATH
    return target.read_text(encoding="utf-8").strip() if target.exists() else ""


def render_agreement_markdown(stats: Mapping, interpretation: str | None = None) -> str:
    """报告里的「自动↔人工一致率」一节。

    数字全部自动算；`interpretation` 是人写的解读，单独成小节并标明出处，
    免得读的人把叙述当成机器结论。
    """
    n = stats["n"]
    ci, ci_q = stats.get("kappa_ci"), stats.get("kappa_quadratic_ci")

    def _with_ci(value, interval) -> str:
        if value is None:
            return "**未定义**"
        if not interval:
            return _fmt3(value)
        return f"{value:.3f}  （95% CI [{interval[0]:.3f}, {interval[1]:.3f}]）"

    lines = [
        "## 自动↔人工一致率（kappa）",
        "",
        f"两边都有分的题 **n={n}**"
        + (f"；人标了裁判没打分：{'、'.join(stats['missing_judge'])}" if stats["missing_judge"] else "")
        + (f"；裁判打了人没标：{'、'.join(stats['missing_human'])}" if stats["missing_human"] else "")
        + "。",
        "",
        "| 量 | 值 |",
        "|---|---|",
        f"| 完全一致率 | {_fmt3(stats['exact_agreement'])} |",
        f"| 相邻一致率（差 ≤1 档） | {_fmt3(stats['adjacent_agreement'])} |",
        f"| Cohen's kappa（unweighted） | {_with_ci(stats['kappa'], ci)} |",
        f"| Cohen's kappa（quadratic weighted） | {_with_ci(stats['kappa_quadratic'], ci_q)} |",
        "",
    ]

    if stats["degenerate"]:
        lines += [f"> ⚠ **kappa {stats['kappa_note']}**", ""]

    lines += [
        f"> ⚠ **样本量 n={n}，不作硬结论。** 置信区间很宽——一题翻档就能让 kappa 动 0.15~0.25。"
        f"引用这个数字时必须带 n。CI 由对这 {n} 个配对做 "
        f"{stats['bootstrap_used']}/{stats['bootstrap_n']} 次有效 bootstrap 重抽样得到"
        f"（{stats['bootstrap_degenerate']} 次因重抽样内某一方无变异被丢弃，种子 "
        f"{stats['bootstrap_seed']}）；n 这么小时 **bootstrap 本身也不可靠**，"
        "重抽样池就只有这几个点。",
        "",
        "### 混淆矩阵（行＝人工，列＝裁判）",
        "",
        "| 人工＼裁判 | 0 | 1 | 2 |",
        "|---|---|---|---|",
    ]
    for label, row in zip(stats["labels"], stats["confusion_matrix"]):
        lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")

    if stats["disagreements"]:
        lines += [
            "",
            "### 分歧题",
            "",
            "| id | 人工 | 裁判 | 差 |",
            "|---|---|---|---|",
        ]
        for d in stats["disagreements"]:
            lines.append(f"| `{d['id']}` | {d['human']} | {d['judge']} | {d['gap']} |")

    text = interpretation if interpretation is not None else stats.get("interpretation") or ""
    if text:
        lines += ["", "### 解读（人工撰写，非自动生成）", "", text.strip()]

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="裁判层与人工标注")
    parser.add_argument("--sheet", action="store_true", help="生成人工标注表与空白骨架")
    parser.add_argument(
        "--full", action="store_true", help="标注表里附上检索片段全文（篇幅会大很多）"
    )
    parser.add_argument("--score", action="store_true", help="自动裁判打分（调 API）")
    parser.add_argument(
        "--kappa", action="store_true", help="读回人工与裁判分数算一致率（纯离线，不调 API）"
    )
    parser.add_argument(
        "--traces",
        type=str,
        default=None,
        help="轨迹文件（默认 reports/traces_latest.jsonl）",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from src.eval.report import TRACES_FILENAME, load_traces

    if args.sheet:
        samples = load_golden_set()
        traces = load_traces(args.traces or (config.REPORTS_DIR / TRACES_FILENAME))
        sheet = write_labeling_sheet(
            samples,
            traces,
            path=config.REPORTS_DIR / (
                SHEET_FILENAME.replace('.md', '_full.md') if args.full else SHEET_FILENAME
            ),
            include_chunk_text=args.full,
        )
        skeleton = write_label_skeleton(samples, traces)
        print(f"标注表：{sheet}")
        print(f"空白骨架：{skeleton}")
        print(f"待标注开放题：{len(_open_samples(samples))} 条")
        return 0

    if args.score:
        samples = load_golden_set()
        traces = load_traces(args.traces or (config.REPORTS_DIR / TRACES_FILENAME))
        targets = _open_samples(samples)
        print(f"自动裁判：{len(targets)} 道开放题 × 1 次调用，模型 {config.JUDGE_MODEL_NAME}")
        caller = DeepSeekJudge()
        rows = score_samples(samples, traces, call=caller)
        for row in rows:
            row["judge_params"] = dict(caller.params)
        path = write_judge_scores(rows)
        print(f"裁判分数：{path}")
        for row in rows:
            flag = "（重问过一次）" if row["retried"] else ""
            print(f"  {row['id']} = {row['judge_score']}  {row['reason']}{flag}")
        print(f"实际生效的调用参数：{caller.params}")
        return 0

    if args.kappa:
        human = load_human_labels()
        auto = load_judge_scores()
        stats = agreement_stats(human, auto)
        print(render_agreement_markdown(stats, load_agreement_interpretation()))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
