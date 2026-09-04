# 评测报告

- 时间（UTC）：2026-09-04T18:59:15+00:00
- git commit：`06e157d`
- 被测模型：`deepseek-chat`（temperature=0.0）
- Embedding：`BAAI/bge-large-zh-v1.5`（查询指令前缀=True）
- 检索 top-k：4；切分 500/80
- 黄金集：36 条

## 指标

| 指标 | 值 | 分母 | 说明 |
|---|---|---|---|
| 检索召回率 recall@k | 0.919 | n=31/36 | 全部检索结果的并集 |
| recall@k（仅首次检索） | 0.758 | n=31/36 | 只算第一次检索；与上一行的差＝多次检索捞回了多少 |
| 任务成功率 | 1.000 | n=26/36 | 闭合题规则判定；开放题待 M5 裁判 |
| 工具调用准确率（严格） | 0.972 | n=36/36 | 折叠连续重复后逐项比对 |
| 工具调用准确率（宽松） | 0.972 | n=36/36 | 只看用了哪些工具，不看顺序 |

## 多轮一致性

- 同题重复次数 K = 5，覆盖 10 题
- 判定一致比例：1.000（1.0 表示每题 K 次判定完全一致）
- 判定结果方差：0.000（同题 K 次 0/1 判定的方差，按题平均）
- 工具路径一致比例：1.000

## Token 消耗

- prompt 142052 + completion 10547 = **152599**

## 逐题明细

| id | 题型 | 期望工具 | 实际工具 | 工具 | recall | 成功 | 停止原因 |
|---|---|---|---|---|---|---|---|
| cap_001 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_002 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_003 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_004 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_005 | closed | retrieve | retrieve | ✔ | 1.000 | ✔ | answered |
| cap_006 | closed | retrieve+calc | retrieve+calc | ✔ | 1.000 | ✔ | answered |
| cap_007 | open | retrieve | retrieve | ✔ | 0.000 | — | answered |
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
| cap_034 | closed | calc | calc | ✔ | — | ✔ | answered |
| cap_035 | open | retrieve | retrieve | ✔ | 0.000 | — | answered |
| cap_036 | open | retrieve | retrieve | ✔ | — | — | max_steps |
