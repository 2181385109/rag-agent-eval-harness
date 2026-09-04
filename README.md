# rag-agent-eval-harness

一个 **RAG + 工具调用 Agent** 的端到端评测流水线。
评测流水线是主角，Agent 是被测对象。模型能力由 **DeepSeek API** 提供，
语料 / 检索 / 指标 / 测试全部本机运行。

完整的范围约束、技术选型与验收口径见 [CLAUDE.md](CLAUDE.md)。

---

## 当前进度：M1 / M6（骨架）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 目录骨架、依赖锁版本、config、`.env` 载入、DeepSeek 客户端封装、hello-world LangGraph 图 | ✅ 完成 |
| M2 | RAG（切分 → FAISS → 检索）+ 两个工具 + ReAct 图 + 轨迹记录 | ⬜ |
| M3 | 30–50 条自建黄金集；recall@k / 任务成功率 / 工具调用准确率 / 多轮一致性 | ⬜ |
| M4 | RAGAS：faithfulness / answer_relevancy / context recall | ⬜ |
| M5 | LLM-as-Judge + 人工标注对照，Cohen's kappa | ⬜ |
| M6 | GitHub Actions 指标回归门禁 + README 定稿 | ⬜ |

> **指标区暂缺，是有意为之。** 本项目的第一红线是「禁止编造任何指标」——
> README 里出现的每个数字都必须能由仓库里的一条命令重新跑出来。
> 评测集和指标要到 M3 才存在，所以现在这里一个数字都没有。

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
python -m src.agent.graph "用一句话解释什么是过拟合。"
```

---

## 设计要点（会随里程碑补全）

- **密钥安全**：API key 只从环境变量 `DEEPSEEK_API_KEY` 读，`.env` 已被 gitignore，
  并且有一条常驻测试 (`tests/test_secrets.py`) 扫描全仓库，一旦有人把 key 粘进代码就直接红。
- **不外发数据**：除对 DeepSeek 的模型调用外没有任何外部请求；
  `langsmith`（langchain-core 的传递依赖）的追踪上报在 `src/config.py` 里被显式关闭。
- **CI 不调真实 API**：GitHub Actions 上不放 key，LLM 调用全部打桩，
  验证的是流水线和指标计算逻辑本身；真实全量评测在本机手动跑。
- **可复现优先**：默认温度固定为 0；需要制造多样性的地方（多轮一致性）显式覆盖。

## 环境说明

Python **3.12**（CLAUDE.md 原文锁 3.11，本机无 3.11，已确认后调整；其余选型不变）。
