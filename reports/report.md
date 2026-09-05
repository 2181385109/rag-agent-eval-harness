# 评测报告

- 时间（UTC）：2026-09-05T04:19:22+00:00
- git commit：`3f3eb46`
- 被测模型：`deepseek-chat`（temperature=0.0）
- Embedding：`BAAI/bge-large-zh-v1.5`（查询指令前缀=True）
- 检索 top-k：4；切分 500/80
- 黄金集：36 条

## 指标

| 指标 | 值 | 分母 | 说明 |
|---|---|---|---|
| 检索召回率 recall@k | 0.932 | n=31/36 | 全部检索结果的并集 |
| recall@k（仅首次检索） | 0.758 | n=31/36 | 只算第一次检索；与上一行的差＝多次检索捞回了多少 |
| 任务成功率 | 1.000 | n=26/36 | 闭合题规则判定；开放题待 M5 裁判 |
| 工具调用准确率（严格） | 0.944 | n=36/36 | 折叠连续重复后逐项比对 |
| 工具调用准确率（宽松） | 0.944 | n=36/36 | 只看用了哪些工具，不看顺序 |

## 多轮一致性

两档并存，互不替代：`temp=0` 是主评测的可复现基线（近乎确定，一致比例天然接近 1.0，信息量有限）；`temp>0` 才测得出模型在采样噪声下的鲁棒性。

| 档位 | temperature | K | 覆盖题数 | 判定一致比例 | 判定结果方差 | 工具路径一致比例 |
|---|---|---|---|---|---|---|
| 可复现基线 | 0.0 | 5 | 10 | 1.000 | 0.000 | 1.000 |
| 鲁棒性专测 | 0.7 | 5 | 10 | 1.000 | 0.000 | 1.000 |

> 判定一致比例 1.0 表示每题 K 次判定完全一致；方差是同题 K 次 0/1 判定的方差，按题平均。

## RAGAS（生成质量）

- 裁判模型：`deepseek-chat`；embedding：`BAAI/bge-large-zh-v1.5`（本地，不外发）
- 提交评测：**35/36** 条（排除 1 条：无答案或无检索内容，见下节）

| 指标 | 值 | 实际打分行数 |
|---|---|---|
| faithfulness | 0.859 | 35/35 |
| answer_relevancy | 0.842 | 35/35 |
| context_recall | 0.905 | 35/35 |
| context_precision | 0.662 | 35/35 |

被排除的样本：`cap_009`（本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起）

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
| cap_007 | open | retrieve | retrieve | ✔ | 0.400 | — | answered |
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
| cap_032 | open | retrieve | retrieve | ✔ | 0.500 | — | answered |
| cap_033 | closed | calc | retrieve+calc | ✘ | — | ✔ | answered |
| cap_034 | closed | calc | calc+retrieve | ✘ | — | ✔ | answered |
| cap_035 | open | retrieve | retrieve | ✔ | 0.000 | — | answered |
| cap_036 | open | retrieve | retrieve | ✔ | — | — | answered |
