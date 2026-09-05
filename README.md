# rag-agent-eval-harness

一个 **RAG + 工具调用 Agent** 的端到端评测流水线。
评测流水线是主角，Agent 是被测对象。模型能力由 **DeepSeek API** 提供，
语料 / 检索 / 指标 / 测试全部本机运行。

完整的范围约束、技术选型与验收口径见 [CLAUDE.md](CLAUDE.md)。

---

## 当前进度：M4 / M6（RAGAS 生成质量指标接入）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 目录骨架、依赖锁版本、config、`.env` 载入、DeepSeek 客户端封装、hello-world LangGraph 图 | ✅ 完成 |
| M2 | RAG（切分 → FAISS → 检索）+ 两个工具 + ReAct 图 + 轨迹记录 | ✅ 完成 |
| M3 | 36 条自建黄金集；recall@k / 任务成功率 / 工具调用准确率 / 多轮一致性 | ✅ 完成 |
| M4 | RAGAS：faithfulness / answer_relevancy / context_recall（+ context_precision） | ✅ 完成 |
| M5 | LLM-as-Judge + 人工标注对照，Cohen's kappa | ⬜ |
| M6 | GitHub Actions 指标回归门禁 + README 定稿 | ⬜ |

> **指标见 [reports/report.md](reports/report.md)，此处暂不摘抄。**
> 第一红线是「禁止编造任何指标」——README 里出现的每个数字都必须能由
> 仓库里的一条命令重新跑出来。M3 的第一份报告已经产出，但 M4/M5 的
> faithfulness 与 kappa 还没有，简历 bullet 要等 M6 定稿时一次性填真值。
>
> 复现命令：`python -m src.eval.report`（需要 DEEPSEEK_API_KEY，会产生费用）。
> 报告里记录了模型名、embedding 模型、top-k、切分参数与 git commit，
> 缺一样这个数字就不可复现。

---

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS

cp .env.example .env      # 然后填入你自己的 DEEPSEEK_API_KEY
```

跑测试（**不需要 API key，不发任何网络请求**）：

```bash
pytest -m "not live"
```

真实打通 DeepSeek 的冒烟测试（需要 key，会产生少量费用）：

```bash
pytest -m live
```

问一句话，走完整的 LangGraph 状态图：

```bash
python -m src.agent.rag --build                       # 建 FAISS 索引（首次会下载 BGE 模型）
python -m src.agent.rag --list                        # 列出 doc_id -> 出处，标注黄金集时用
python -m src.agent.graph --trace "PSI 告警阈值定在多少？"   # 跑 Agent 并打印完整轨迹

python -m src.eval.report --limit 5 --no-consistency   # 小子集试跑（省钱）
python -m src.eval.report --ragas                      # 全量评测 + RAGAS，出 reports/

# 轨迹会落盘到 reports/traces_latest.jsonl，可不重跑 Agent 只补算指标：
python -m src.eval.report --from-traces reports/traces_latest.jsonl --ragas
```

---

## 设计要点（会随里程碑补全）

- **密钥安全**：API key 只从环境变量 `DEEPSEEK_API_KEY` 读，`.env` 已被 gitignore，
  并且有一条常驻测试 (`tests/test_secrets.py`) 扫描全仓库，一旦有人把 key 粘进代码就直接红。
- **不外发数据**：除对 DeepSeek 的模型调用外没有任何外部请求；
  `langsmith`（langchain-core 的传递依赖）的追踪上报在 `src/config.py` 里被显式关闭。
- **CI 不调真实 API**：GitHub Actions 上不放 key，LLM 调用全部打桩，
  验证的是流水线和指标计算逻辑本身；真实全量评测在本机手动跑。
- **可复现优先**：默认温度固定为 0；多轮一致性另跑一档 `temperature>0` 的鲁棒性专测，
  两档并列报数、互不替代（temp=0 下一致比例天然接近 1.0，信息量有限）。
- **分母诚实**：指标无定义时返回 `None` 而不是当 0（纯算术题没有应检索文档、
  开放题在 M3 无法规则判定）；每个指标都带 `n/total`，报告正文必须写出分母。
- **失败如实入账**：Agent 选错工具、跑满步数没收口、RAGAS 裁判调用超时，
  全部记录并单列。任一 RAGAS 指标没在全部提交行上打出分就打 ⚠ 标记，
  因为 `mean()` 会跳过空值，不标出来就等于虚报覆盖面。
- **轨迹落盘**：主评测与两档一致性的轨迹都存进 `reports/`，
  后续加指标（M5 裁判）复用同一批轨迹，既省钱也保证指标之间对得上同一次运行。

## 环境说明

Python **3.12**（CLAUDE.md 原文锁 3.11，本机无 3.11，已确认后调整；其余选型不变）。
