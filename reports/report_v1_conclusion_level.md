# 评测报告

- 时间（UTC）：2026-09-06T07:22:14+00:00
- git commit：`e4583f0`
- 被测模型：`deepseek-chat`（temperature=0.0）
- Embedding：`BAAI/bge-large-zh-v1.5`（查询指令前缀=True）
- 检索 top-k：4；切分 500/80
- 黄金集：36 条

## 指标

| 指标 | 值 | 分母 | 说明 |
|---|---|---|---|
| 检索召回率 recall@k | 0.968 | n=31/36 | 全部检索结果的并集 |
| recall@k（仅首次检索） | 0.774 | n=31/36 | 只算第一次检索；与上一行的差＝多次检索捞回了多少 |
| 任务成功率 | 0.972 | n=36/36 | 闭合题 26 题走 answer_keys；开放题 10 题走裁判分，满 2 分才算成功 |
| 工具调用准确率（严格） | 0.944 | n=36/36 | 折叠连续重复后逐项比对 |
| 工具调用准确率（宽松） | 0.944 | n=36/36 | 只看用了哪些工具，不看顺序 |

### 任务成功率的判定口径

- 闭合题 **26** 题：`answer_keys` 全部命中才算成功。
- 开放题 **10** 题：裁判（`deepseek-reasoner`）打分 **≥ 2** 才算成功——三级标度里 1 分是「方向对但要点有遗漏」，**不计成功**。
- 裁判分来源：`judge_scores.jsonl`。
- ⚠ 未核验（打分行没存答案指纹，无从机器确认它是给这批轨迹打的）：10 行——重跑一次 `python -m src.eval.judge --score` 即可补上。

未通过的题：
- `cap_007`（开放/裁判） 裁判未给满分

## RAGAS（生成质量）

- 裁判模型：`deepseek-reasoner`；embedding：`BAAI/bge-large-zh-v1.5`（本地，不外发）
- ⚠ **本节为复用，不是本次重算**：分数原样取自 `reports\eval_20260905T142507Z.json`（2026-09-05T14:25:07+00:00），前提是轨迹未变。
- 提交评测：**35/36** 条（排除 1 条：无答案或无检索内容，见下节）

| 指标 | 值 | 实际打分行数 |
|---|---|---|
| faithfulness | 0.862 | 35/35 |
| answer_relevancy | 0.826 | 35/35 |
| context_recall | 0.824 | 35/35 |
| context_precision | 0.670 | 35/35 |

被排除的样本：`cap_009`（本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起）

## 自动↔人工一致率（kappa）

两边都有分的题 **n=10**。

| 量 | 值 |
|---|---|
| 完全一致率 | 0.600 |
| 相邻一致率（差 ≤1 档） | 0.900 |
| Cohen's kappa（unweighted） | 0.216  （95% CI [0.091, 0.750]） |
| Cohen's kappa（quadratic weighted） | 0.103  （95% CI [-0.000, 0.737]） |

> ⚠ **样本量 n=10，不作硬结论。** 置信区间很宽——一题翻档就能让 kappa 动 0.15~0.25。引用这个数字时必须带 n。CI 由对这 10 个配对做 1293/2000 次有效 bootstrap 重抽样得到（707 次因重抽样内某一方无变异被丢弃，种子 20260906）；n 这么小时 **bootstrap 本身也不可靠**，重抽样池就只有这几个点。

### 混淆矩阵（行＝人工，列＝裁判）

| 人工＼裁判 | 0 | 1 | 2 |
|---|---|---|---|
| **0** | 0 | 0 | 1 |
| **1** | 0 | 1 | 3 |
| **2** | 0 | 0 | 5 |

### 分歧题

| id | 人工 | 裁判 | 差 |
|---|---|---|---|
| `cap_010` | 0 | 2 | 2 |
| `cap_017` | 1 | 2 | 1 |
| `cap_032` | 1 | 2 | 1 |
| `cap_035` | 1 | 2 | 1 |

### 解读（人工撰写，非自动生成）

本节数字由 `python -m src.eval.judge --kappa` 自动算出；以下解读是**人写的**，
随分歧内容变化需手工更新（源文件：`data/agreement_interpretation.md`）。

**1. cap_010 的 2 档差距部分源于信息不对等，不是纯判断差异。**
裁判按设计盲看，只拿到问题、参考答案、Agent 答案；人工标注者用的是
`reports/human_labeling_sheet_full.md`，其中带**检索片段全文**（该题 20 块），
并另外对 `corpus/` 做过 grep 确认「Redis」全库出现 0 次。
该题 Agent 的结论正确（规格书确无 Redis），但给出的第 4 条依据
「检索结果中出现的 Redis 相关内容来自航班延误预测项目」是凭空断言——
检索片段与全语料中都不存在含 Redis 的块。人工据此判 0（编造），
裁判在只看参考答案的信息条件下**无从判伪**，反而把该句当作「未编造」的佐证判 2。
所以这条分歧至少一部分要记在信息量差异上，而非两者判断力的差异。
保持裁判盲看是有意的取舍：给了片段，裁判会拿原文替 Agent 找补，
判的就不再是答案本身。代价就是这类需要回查语料才能识别的编造，它抓不到。

**2. 其余三条分歧是系统性的：裁判做结论级比对，人工做要点级比对。**
cap_017、cap_032、cap_035 三题人工判 1、裁判判 2，模式一致——
Agent 的结论方向都对，但都漏掉了参考答案里的一个具体限定：
KS 题缺 `max(TPR−FPR)` 的公式表述；基线题缺「标准化 + L2」；
K8s 题把规格书「明确禁止、不引入 K8s/Airflow/云服务」弱化成了「全程未提及」。
裁判三次的理由句都是「与参考答案一致」，即它比对的是**结论层**；
人工把参考答案拆成要点逐条点数，比对的是**要点层**。
这不是随机噪声，是两把尺子的刻度不同，也是本轮 kappa 偏低的主因。
要收敛，得在判据里写死「参考答案中的每个具体限定缺失即降档」，
而不是靠换模型或调温度。

**3. kappa=0.216 在 n=10 下不构成硬结论。**
95% CI 覆盖 [0.091, 0.750]，横跨「几乎无一致」到「相当一致」。
完全一致率 0.600、相邻一致率 0.900 这两个不做机遇校正的量在此样本量下更稳，
读的时候应以它们为主、kappa 为辅。引用 kappa 必须连带 n 与区间一起给。
另注意 quadratic 加权反而更低（0.103）：cap_010 那条差 2 档被 4 倍惩罚，
且裁判边缘分布极偏（10 题里 9 题打 2），机遇一致率被推高，校正后自然更低。

## Token 消耗

- Agent 侧：prompt 155346 + completion 11256 = **166602**
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
| cap_015 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_016 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_017 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_018 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_019 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_020 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_021 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_022 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_023 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_024 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_025 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_026 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_027 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_028 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_029 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_030 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_031 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | 规则 | answered |
| cap_032 | open | retrieve | retrieve | ✔ | 1.000 | ✔ | 裁判 | answered |
| cap_033 | closed | calc | retrieve+calc | ✘ | — | ✔ | 规则 | answered |
| cap_034 | closed | calc | calc+retrieve | ✘ | — | ✔ | 规则 | answered |
| cap_035 | open | retrieve | retrieve | ✔ | 0.000 | ✔ | 裁判 | answered |
| cap_036 | open | retrieve | retrieve | ✔ | — | ✔ | 裁判 | answered |
