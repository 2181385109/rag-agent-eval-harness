# 评测报告

- 时间（UTC）：2026-09-05T07:07:27+00:00
- git commit：`101bbf6`
- 被测模型：`deepseek-chat`（temperature=0.0）
- Embedding：`BAAI/bge-large-zh-v1.5`（查询指令前缀=True）
- 检索 top-k：4；切分 500/80
- 黄金集：36 条

## 指标

| 指标 | 值 | 分母 | 说明 |
|---|---|---|---|
| 检索召回率 recall@k | 0.968 | n=31/36 | 全部检索结果的并集 |
| recall@k（仅首次检索） | 0.774 | n=31/36 | 只算第一次检索；与上一行的差＝多次检索捞回了多少 |
| 任务成功率 | 1.000 | n=26/36 | 闭合题规则判定；开放题待 M5 裁判 |
| 工具调用准确率（严格） | 0.944 | n=36/36 | 折叠连续重复后逐项比对 |
| 工具调用准确率（宽松） | 0.944 | n=36/36 | 只看用了哪些工具，不看顺序 |

## RAGAS（生成质量）

- 裁判模型：`deepseek-reasoner`；embedding：`BAAI/bge-large-zh-v1.5`（本地，不外发）
- 提交评测：**35/36** 条（排除 1 条：无答案或无检索内容，见下节）

| 指标 | 值 | 实际打分行数 |
|---|---|---|
| faithfulness | 0.859 | 31/35  ⚠ |
| answer_relevancy | 0.816 | 35/35 |
| context_recall | 0.824 | 35/35 |
| context_precision | 0.699 | 34/35  ⚠ |

> ⚠ **有指标未在全部提交行上打出分**（裁判调用失败或超时，RAGAS 会把该行留空，而均值默认跳过空值）。带 ⚠ 的指标覆盖面小于分母，不能当作全量结果引用。

被排除的样本：`cap_009`（本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起）

### 裁判敏感性对照

同一批轨迹，换裁判模型重判一次：`deepseek-chat` → `deepseek-reasoner`。差值反映的是**评分标准本身有多依赖裁判模型**，与被测 Agent 无关。

| 指标 | deepseek-chat | deepseek-reasoner | 差值 | 打分行数 |
|---|---|---|---|---|
| faithfulness | 0.859 | 0.859 | **不可比** | 35/35 → 31/35 |
| answer_relevancy | 0.842 | 0.816 | -0.027 | 35/35 → 35/35 |
| context_recall | 0.905 | 0.824 | -0.081 | 35/35 → 35/35 |
| context_precision | 0.662 | 0.699 | **不可比** | 35/35 → 34/35 |

> ⚠ **faithfulness、context_precision 的差值不可比**：两次运行打分成功的行数不同，均值是在不同子集上算的。看着像「换裁判没影响」的 0.000 差值，很可能只是两个不同样本集碰巧接近。要得到可比的对照，必须两次都打满同样的行数。

## Token 消耗

- Agent 侧：prompt 155346 + completion 11256 = **166602**
- 注：**不含 RAGAS 裁判与多轮一致性重复运行的开销**——这两部分走的是独立调用，本计数只覆盖主评测每题一次的 Agent 运行。

## 逐题明细

| id | 题型 | 期望工具 | 实际工具 | 工具 | recall | 成功 | 停止原因 |
|---|---|---|---|---|---|---|---|
| cap_001 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_002 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_003 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_004 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_005 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_006 | closed | retrieve+calc | retrieve+calc | ✔ | 1.000 | ✔ | answered |
| cap_007 | open | retrieve | retrieve | ✔ | 1.000 | — | answered |
| cap_008 | open | retrieve | retrieve | ✔ | 1.000 | — | answered |
| cap_009 | closed | calc | calc | ✔ | — | ✔ | answered |
| cap_010 | open | retrieve | retrieve | ✔ | — | — | answered |
| cap_011 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_012 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_013 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_014 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_015 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_016 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_017 | open | retrieve | retrieve | ✔ | 1.000 | — | answered |
| cap_018 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_019 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_020 | open | retrieve | retrieve | ✔ | 1.000 | — | answered |
| cap_021 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_022 | open | retrieve | retrieve | ✔ | 1.000 | — | answered |
| cap_023 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_024 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_025 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_026 | open | retrieve | retrieve | ✔ | 1.000 | — | answered |
| cap_027 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_028 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_029 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_030 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_031 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_032 | open | retrieve | retrieve | ✔ | 1.000 | — | answered |
| cap_033 | closed | calc | retrieve+calc | ✘ | — | ✔ | answered |
| cap_034 | closed | calc | calc+retrieve | ✘ | — | ✔ | answered |
| cap_035 | open | retrieve | retrieve | ✔ | 0.000 | — | answered |
| cap_036 | open | retrieve | retrieve | ✔ | — | — | answered |
