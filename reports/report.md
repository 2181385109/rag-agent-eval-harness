# 评测报告

- 时间（UTC）：2026-09-06T09:28:35+00:00
- git commit：`65bf140`
- 被测模型：`deepseek-chat`（temperature=0.0）
- Embedding：`BAAI/bge-large-zh-v1.5`（查询指令前缀=True）
- 检索 top-k：4；切分 500/80
- 黄金集：36 条

## 指标

| 指标 | 值 | 分母 | 说明 |
|---|---|---|---|
| 检索召回率 recall@k | 0.968 | n=31/36 | 全部检索结果的并集 |
| recall@k（仅首次检索） | 0.774 | n=31/36 | 只算第一次检索；与上一行的差＝多次检索捞回了多少 |
| 任务成功率 | 0.917 | n=36/36 | 闭合题 26 题走 answer_keys；开放题 10 题走裁判分，满 2 分才算成功 |
| 工具调用准确率（严格） | 0.944 | n=36/36 | 折叠连续重复后逐项比对 |
| 工具调用准确率（宽松） | 0.944 | n=36/36 | 只看用了哪些工具，不看顺序 |

### 任务成功率的判定口径

- 闭合题 **26** 题：`answer_keys` 全部命中才算成功。
- 开放题 **10** 题：裁判（`deepseek-reasoner`）打分 **≥ 2** 才算成功——三级标度里 1 分是「方向对但要点有遗漏」，**不计成功**。
- 裁判分来源：`judge_scores.jsonl`。

未通过的题：
- `cap_007`（开放/裁判） 裁判未给满分
- `cap_032`（开放/裁判） 裁判未给满分
- `cap_035`（开放/裁判） 裁判未给满分

## RAGAS（生成质量）

- 裁判模型：`deepseek-reasoner`；embedding：`BAAI/bge-large-zh-v1.5`（本地，不外发）
- ✅ **已验证轨迹指纹一致（35 条），复用成立**：分数原样取自 `reports\eval_20260905T142507Z.json`（2026-09-05T14:25:07+00:00）。核验口径：answer 文本 + retrieved_doc_ids 集合的指纹逐题比对（corpus 自 M2 后未再变动，doc_id 相同即 context 必然相同）。
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
| 完全一致率 | 0.800 |
| 相邻一致率（差 ≤1 档） | 0.900 |
| Cohen's kappa（unweighted） | 0.623  （95% CI [0.231, 1.000]） |
| Cohen's kappa（quadratic weighted） | 0.324  （95% CI [0.000, 1.000]） |

> ⚠ **样本量 n=10，不作硬结论。** 置信区间很宽——一题翻档就能让 kappa 动 0.15~0.25。引用这个数字时必须带 n。CI 由对这 10 个配对做 1933/2000 次有效 bootstrap 重抽样得到（67 次因重抽样内某一方无变异被丢弃，种子 20260906）；n 这么小时 **bootstrap 本身也不可靠**，重抽样池就只有这几个点。

### 混淆矩阵（行＝人工，列＝裁判）

| 人工＼裁判 | 0 | 1 | 2 |
|---|---|---|---|
| **0** | 0 | 0 | 1 |
| **1** | 0 | 3 | 1 |
| **2** | 0 | 0 | 5 |

### 分歧题

| id | 人工 | 裁判 | 差 |
|---|---|---|---|
| `cap_010` | 0 | 2 | 2 |
| `cap_017` | 1 | 2 | 1 |

### 解读（人工撰写，非自动生成）

本节数字由 `python -m src.eval.judge --kappa` 自动算出；以下解读是**人写的**，
随分歧内容变化需手工更新（源文件：`data/agreement_interpretation.md`）。

**0. 本轮的判据是 v2（要点级），与上一轮的 v1（结论级）不是同一把尺子。**
v1 的 1/2 档写的是「要点有遗漏」/「要点齐全」，却没定义**什么算一个要点**。
人和机器就各按各的理解办：裁判比对结论层（三次的理由句都是「与参考答案一致」），
人工把参考答案拆成具体限定逐条点数。cap_017/032/035 三题人判 1、机判 2 全是这个模式——
不是随机噪声，是刻度不同，也是 v1 那轮 kappa=0.216 的主因。

v2 把「要点」定死成**参考答案里的每一项具体限定**（数字与阈值、比例、公式或指标定义、
方法与算法名、明确的禁止或例外，各算一项），并写明**缺一项即降到 1 分**、
「结论方向一致」不足以给 2 分；提示词里还加了三步打分流程，强制先列清单再逐项核对。
判据正文与打分细则和人工标注表**同一段字**（`judge.LABEL_SCALE` / `judge.RUBRIC_NOTES`，
有测试钉住），改一处两边同时生效。

改前那批分留档在 `reports/judge_scores_v1_conclusion_level.jsonl`，
报告的「裁判判据修订对照」一节给出改前改后两版的全部数字。
轨迹、人工标注、被测模型三者均未变动，**唯一的变量是判据**。

**1. kappa 从 0.216 升到 0.623，但这个上升有相当一部分是构造性的，不能当作「评测更可信了」来卖。**
v2 做的事情，本质是**让裁判向人工既有的尺子靠拢**——人工本来就是要点级标注的
（这一点写在 v1 那轮的解读里，早于本次修订，可在 git 历史中查证）。
把一方的判据改成另一方的做法，两者当然更一致。真正独立的双盲一致性，
需要人工在 v2 判据下重新盲标一遍才能测得，本轮**没有**做这件事：
人工标注仍是 v1 时期的那一份，没有重标，也没有依据机器分调整过。
所以诚实的说法是「判据澄清后，自动裁判能复现人工的判定口径」，
而不是「两个独立评分者达成了实质一致」。

**2. 改判的两题（cap_032、cap_035）都改对了方向，且都是判据直接命中的类型。**
- `cap_032` 2→1：Agent 只写了「逻辑回归」和「常数基线」，漏了参考答案里的
  「标准化 + L2」这一限定，也把「常数/热门基线」缩成了「常数基线」。裁判 v2 两项都点了名。
- `cap_035` 2→1：规格书是「明确禁止、不引入 K8s/Airflow/云服务」，
  Agent 弱化成了「全程未提及」。裁判 v2 明确指出缺「明确禁止并写入范围纪律」这一限定。

两题改后都与人工判定相同。裁判 v2 的理由句也从「与参考答案一致」变成了逐项点名缺失，
说明它确实在执行清单核对，而不是换了个说法给同样的分。

**3. cap_017 没有改判，这是本轮残余分歧里唯一值得存疑的一条——存疑的对象是人工，不是裁判。**
参考答案写的是「好坏样本累积分布之差的最大值（max(TPR−FPR)）」。
Agent 给出了「好坏样本累积分布之差的最大值」，但没写 `max(TPR−FPR)` 这个公式。
人工把公式记作一项独立的具体限定，据此判 1；裁判 v2 认为口径定义已经命中，判 2。
两种读法都讲得通——**`max(TPR−FPR)` 与「累积分布之差的最大值」在数学上是同一件事**，
括号里那串更像是同一限定的符号写法，而不是第二项限定。
按这个理解，这题裁判判 2 更贴合 v2 判据，人工那 1 分是比判据更严。
**但本轮不动它**：判据只改一次，改完认结果。要修正只能通过重新盲标人工侧，
不能挑一道题手工翻档——那就成了朝着数字调判定。

**4. cap_010 的 2 档差距仍在，性质与 v1 时相同：信息不对等，判据改不动它。**
裁判按设计盲看，只拿到问题、参考答案、Agent 答案；人工用的是
`reports/human_labeling_sheet_full.md`，其中带**检索片段全文**（该题 20 块），
并另外对 `corpus/` 做过 grep 确认「Redis」全库出现 0 次。
该题 Agent 结论正确（规格书确无 Redis），但第 4 条依据
「检索结果中出现的 Redis 相关内容来自航班延误预测项目」是凭空断言——
检索片段与全语料中都不存在含 Redis 的块。人工据此判 0（编造），
裁判在只看参考答案的信息条件下**无从判伪**。
保持裁判盲看是有意的取舍：给了片段，裁判会拿原文替 Agent 找补，
判的就不再是答案本身。代价就是这类需要回查语料才能识别的编造，它抓不到。
这是判据层面解决不了的问题，属 v2 的可观测性范畴，本轮不动。

**5. kappa=0.623 在 n=10 下同样不构成硬结论——区间反而更宽了。**
95% CI `[0.231, 1.000]`，上界顶到了 1.0。样本量没变，分歧从 4 题减到 2 题，
bootstrap 重抽样里更容易抽到「零分歧」的样本，kappa 分布因此被推向上界并被截断。
换句话说，**这个区间的收窄程度配不上点估计的涨幅**。
完全一致率 0.800、相邻一致率 0.900 这两个不做机遇校正的量在此样本量下更稳，
读的时候应以它们为主、kappa 为辅。引用 kappa 必须连带 n 与区间一起给。

另注意 quadratic 加权仍然低得多（0.324）：cap_010 那条差 2 档被 4 倍惩罚，
且裁判边缘分布依然偏（10 题里 7 题打 2），机遇一致率被推高，校正后自然更低。
两个口径并列报，不挑好看的那个。

**6. 连带后果：任务成功率同时下降。**
开放题的裁判分既进 kappa，也进任务成功率（满分 2 才算成功）。
cap_032、cap_035 掉到 1 分，于是任务成功率 **0.972 → 0.917（n=36/36）**，
失败题从 1 道变成 3 道（cap_007、cap_032、cap_035）。
判据收紧让一致率上升、让成功率下降，这两件事是同一次改动的两面，必须一起报。

## 裁判判据修订对照（改前 / 改后）

- 判据：`未标注（判据版本字段之前）` → `v2-要点级`
- 改前的分留档在 `judge_scores_v1_conclusion_level.jsonl`；轨迹、人工标注、被测模型三者**均未变动**，唯一的变量是判据本身。

| 量 | 改前 | 改后 | 差 | 分母 |
|---|---|---|---|---|
| 任务成功率 | 0.972 | 0.917 | -0.056 | n=36/36 |
| 完全一致率（人↔裁判） | 0.600 | 0.800 | +0.200 | n=10 |
| 相邻一致率（差 ≤1 档） | 0.900 | 0.900 | +0.000 | n=10 |
| Cohen's kappa（unweighted） | 0.216 | 0.623 | +0.407 | n=10 |
| Cohen's kappa（quadratic） | 0.103 | 0.324 | +0.222 | n=10 |

### 改判的题（2 道）

| id | 改前 | 改后 | 人工 |
|---|---|---|---|
| `cap_032` | 2 | 1 | 1 |
| `cap_035` | 2 | 1 | 1 |

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
| cap_032 | open | retrieve | retrieve | ✔ | 1.000 | ✘ | 裁判 | answered |
| cap_033 | closed | calc | retrieve+calc | ✘ | — | ✔ | 规则 | answered |
| cap_034 | closed | calc | calc+retrieve | ✘ | — | ✔ | 规则 | answered |
| cap_035 | open | retrieve | retrieve | ✔ | 0.000 | ✘ | 裁判 | answered |
| cap_036 | open | retrieve | retrieve | ✔ | — | ✔ | 裁判 | answered |
