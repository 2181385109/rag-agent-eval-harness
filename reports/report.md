# 评测报告

- 时间（UTC）：2026-09-12T08:17:18+00:00
- git commit：`56021c7`
- 被测模型：请求 `deepseek-chat`，响应 `deepseek-flash`×74（temperature=0.0）
- Embedding：`BAAI/bge-large-zh-v1.5`（查询指令前缀=True）
- 检索 top-k：4；切分 500/80
- 黄金集：36 条

## 指标

| 指标 | 值 | 分母 | 说明 |
|---|---|---|---|
| 检索召回率 recall@k | 0.871 | n=31/36 | 全部检索结果的并集 |
| recall@k（仅首次检索） | 0.774 | n=31/36 | 只算第一次检索；与上一行的差＝多次检索捞回了多少 |
| 任务成功率 | 0.750 | n=36/36 | 闭合题 26 题走 answer_keys；开放题 10 题走裁判分，满 2 分才算成功 |
| 工具调用准确率（严格） | 1.000 | n=36/36 | 折叠连续重复后逐项比对 |
| 工具调用准确率（宽松） | 1.000 | n=36/36 | 只看用了哪些工具，不看顺序 |

### 任务成功率的判定口径

- 闭合题 **26** 题：`answer_keys` 全部命中才算成功。
- 开放题 **10** 题：裁判（请求 `deepseek-reasoner`，响应 `deepseek-flash`×11）打分 **≥ 2** 才算成功——三级标度里 1 分是「方向对但要点有遗漏」，**不计成功**。
- 裁判分来源：`judge_scores_20260912T074455Z_full.jsonl`。

未通过的题：
- `cap_007`（开放/裁判） 裁判未给满分
- `cap_015`（闭合/规则） answer_keys 未全部命中
- `cap_016`（闭合/规则） answer_keys 未全部命中
- `cap_017`（开放/裁判） 裁判未给满分
- `cap_020`（开放/裁判） 裁判未给满分
- `cap_027`（闭合/规则） answer_keys 未全部命中
- `cap_032`（开放/裁判） 裁判未给满分
- `cap_035`（开放/裁判） 裁判未给满分
- `cap_036`（开放/裁判） 裁判未给满分

## RAGAS（生成质量）

- 裁判模型：请求 `deepseek-reasoner`，响应 `deepseek-flash`×426；embedding：`BAAI/bge-large-zh-v1.5`（本地，不外发）
- 提交评测：**33/36** 条（排除 3 条：无答案或无检索内容，见下节）

| 指标 | 值 | 实际打分行数 |
|---|---|---|
| faithfulness | 0.898 | 33/33 |
| answer_relevancy | 0.694 | 33/33 |
| context_recall | 0.727 | 33/33 |
| context_precision | 0.654 | 33/33 |

被排除的样本：`cap_009`（本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起）、`cap_033`（本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起）、`cap_034`（本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起）

## 自动↔人工一致率（kappa）

本快照不含此节：agreement 为空（未计算，或人工标注不对应本批答案——人工标注只对标注时的那批答案有效，答案变了 kappa 不可直接重算）。

## Token 消耗

- Agent 侧：prompt 93517 + completion 8055 = **101572**
- 注：**不含 RAGAS 裁判与多轮一致性重复运行的开销**——这两部分走的是独立调用，本计数只覆盖主评测每题一次的 Agent 运行。

## 逐题明细

| id | 题型 | 期望工具 | 实际工具 | 工具 | recall | 成功 | 判据 | 停止原因 |
|---|---|---|---|---|---|---|---|---|
| cap_001 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_002 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_003 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_004 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_005 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_006 | closed | retrieve+calc | retrieve+calc | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_007 | open | retrieve | retrieve | ✔ | 1.000 | ✘ | 裁判 | answered |
| cap_008 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_009 | closed | calc | calc | ✔ | — | ✔ | 规则 | answered |
| cap_010 | open | retrieve | retrieve | ✔ | — | ✔ | 裁判 | answered |
| cap_011 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_012 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_013 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_014 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_015 | closed | retrieve | retrieve | ✔ | 0.000 | ✘ | 规则 | answered |
| cap_016 | closed | retrieve | retrieve | ✔ | 0.000 | ✘ | 规则 | answered |
| cap_017 | open | retrieve | retrieve | ✔ | 1.000 | ✘ | 裁判 | answered |
| cap_018 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_019 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_020 | open | retrieve | retrieve | ✔ | 1.000 | ✘ | 裁判 | answered |
| cap_021 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_022 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_023 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_024 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_025 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_026 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_027 | closed | retrieve | retrieve | ✔ | 0.000 | ✘ | 规则 | answered |
| cap_028 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_029 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_030 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_031 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_032 | open | retrieve | retrieve | ✔ | 1.000 | ✘ | 裁判 | answered |
| cap_033 | closed | calc | calc | ✔ | — | ✔ | 规则 | answered |
| cap_034 | closed | calc | calc | ✔ | — | ✔ | 规则 | answered |
| cap_035 | open | retrieve | retrieve | ✔ | 0.000 | ✘ | 裁判 | answered |
| cap_036 | open | retrieve | retrieve | ✔ | — | ✘ | 裁判 | answered |
