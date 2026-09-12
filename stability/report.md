# 稳定性与延迟分解报告（脚本生成，勿手改）

> **告警：请求模型名与服务端响应模型名不一致。**
> 请求 `deepseek-chat`，全部 180 次运行的响应 `model` 字段为 `deepseek-flash`×180（system_fingerprint `aeb56401ca74e127821c4f9126dcb669`×180）。
> DeepSeek 官方已于 2026-07-24 弃用 deepseek-chat / deepseek-reasoner 两个模型名；现由旧名路由至 deepseek-flash（V4.1 Flash）。本仓库 src/config.py 仍硬编码旧名。
> 本报告中所有「被测模型」的数字，实际服务模型均为上述响应模型名。

> 本文件由 `python -m stability.analyze` 从 `stability/raw/run_20260911T162531Z.jsonl` 生成；
> 原始产物 sha256 = `bdf6b3018bfbe633adae7ac39f0a82078fdb6e0b02a5cf7579e5107fde4c8240`；分析时间 2026-09-12T03:28:44+00:00。
> 每个数字旁都带样本量 n、重复次数 k 与环境标识（PERF_SPEC §0.3）。

## 环境与随机性来源

| 项 | 值 |
|---|---|
| 请求模型名（config.MODEL_NAME） | `deepseek-chat` |
| 实际响应模型名（响应 `model` 字段，按行计数） | `deepseek-flash`×180  **← 与请求名不一致** |
| system_fingerprint（按行计数） | `aeb56401ca74e127821c4f9126dcb669`×180 |
| 单次运行内响应模型一致性 | 逐条记录 0 次运行：一致 0、不一致 0；未逐条记录 180 次（旧格式（2026-09-11）只存每次运行首条响应，无法判断运行内一致性；逐条记录自 2026-09-12 起） |
| temperature | 0.0 |
| seed | none — 本机侧无随机数：FAISS 精确内积搜索、规则判定均确定；唯一随机源是 API 侧解码（已发 temperature=0） |
| embedding | BAAI/bge-large-zh-v1.5（query 指令前缀：True） |
| 检索参数 | top_k=4，chunk=500/80，max_steps=6 |
| git commit | 78b5122 |
| 运行时间 | 2026-09-11T16:25:31+00:00 → 2026-09-11T16:33:52+00:00 |
| 顺序 | pass-major（先把所有题跑完第 0 次，再第 1 次……） |
| 硬件 | AMD Ryzen 7 7840H  w/ Radeon 780M Graphics × 16 核，RAM 15.3 GB |
| 软件 | Windows 11 (10.0.26200)；Python 3.12.3；openai 2.54.0；sentence-transformers 6.0.1 |
| 网络 | 压测客户端与 DeepSeek API 之间为公网；LLM 延迟含网络往返与服务端排队 |

## 延迟分解

n=180（36 题 × k=5），temperature=0.0，model=请求 deepseek-chat / 响应 deepseek-flash；有效行 n_ok=180，报错行 n_error=0，错误率 0.000。

| 段 | n | P50 | P95 | P99 | 均值 | 最大 | 占比中位数 |
|---|---|---|---|---|---|---|---|
| 端到端 | 180 | 2.52s | 3.72s | 6.96s | 2.65s | 7.23s | 1.00 |
| LLM 调用 | 180 | 2.29s | 3.44s | 5.96s | 2.42s | 6.41s | 0.91 |
| 检索（BGE+FAISS） | 180 | 0.27s | 0.32s | 0.82s | 0.23s | 1.02s | 0.09 |
| 工具（calc） | 180 | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00 |
| 编排开销 | 180 | 0.00s | 0.00s | 0.00s | 0.00s | 0.01s | 0.00 |

单次 LLM 调用（n=373 次）：P50 1.06s / P95 2.11s / P99 3.05s；每次运行 LLM 调用次数 P50 2.0、最大 4。

### 按服务端缓存命中拆分（端到端）

本批数据无法按缓存命中拆分：180 次有效运行的运行级 cache_hit 全部 > 0，未命中组 n=0。分组依据是运行级的和：同一运行内第二次调用起会命中第一次写入的前缀，所以含 ≥2 次 LLM 调用的运行几乎必然 > 0；逐次调用的缓存命中 2026-09-12 起才逐条记录。上文端到端分位数即为全部有效行。

### 按 pass 看单次 LLM 调用延迟、轨迹长度与 token

轨迹长度众数 n_llm_calls = 2；「控制长度」列只取 n_llm_calls 等于众数的运行。

| pass | n | 端到端 P50 | 端到端 P50（控制长度） | n（控制长度） | 单次 LLM 调用 P50 | 单次 P95 | 调用数 | 平均 n_llm_calls | 平均 token | 平均缓存命中 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 2.38s | 2.38s | 34 | 1.07s | 2.05s | 74 | 2.06 | 2793 | 1070 |
| 1 | 36 | 2.69s | 2.65s | 34 | 1.09s | 2.10s | 75 | 2.08 | 3073 | 1632 |
| 2 | 36 | 2.58s | 2.45s | 34 | 1.03s | 2.08s | 75 | 2.08 | 2956 | 1898 |
| 3 | 36 | 2.44s | 2.41s | 34 | 1.01s | 2.24s | 74 | 2.06 | 2752 | 1728 |
| 4 | 36 | 2.55s | 2.50s | 34 | 1.08s | 1.91s | 75 | 2.08 | 3041 | 1781 |

pass 0（无跨轮缓存）端到端 P50 2.38s，后续 pass 的 P50 为 pass 1 2.69s、pass 2 2.58s、pass 3 2.44s、pass 4 2.55s；pass 0 低于全部后续 pass。单次 LLM 调用 P50 各 pass 接近（各 pass 单次 LLM 调用 P50 相对 pass 0 偏差 ≤ 10%）；控制轨迹长度后 pass 0 仍低于全部后续 pass。—— **未解释现象**。

**结论**：LLM 调用占端到端延迟的 91.4%（占比中位数），检索占 8.6%，工具占 0.0%，编排开销占 0.1%。

## 稳定性（三个层级，逐级放宽）

| 指标 | 值 | 分母 | 口径 |
|---|---|---|---|
| 轨迹自洽率（严格） | **0.778** | n=36/36，k=5 | k 次运行的工具调用序列逐项完全一致的题占比（不折叠连续重复） |
| 轨迹自洽率（折叠连续重复） | 1.000 | n=36/36，k=5 | 参考口径 |
| 判定自洽率（合并） | **0.944** | n=36/36（规则 26、裁判 10），k=5 | k 次运行的任务成功判定全部相同的题占比；闭合题走 answer_keys 规则，开放题走逐次裁判分（>= 2 为成功），没有逐次裁判分的开放题不计入分母；报错的运行计为失败 |
| 判定自洽率 · 规则路（闭合题） | **0.962** | n=26，k=5 | answer_keys 全命中；不一致：['cap_016'] |
| 判定自洽率 · 裁判路（开放题） | **0.900** | n=10，k=5 | 裁判分 ≥ 2；不一致：['cap_017'] |
| 裁判分跨次标准差（裁判路） | 均值 **0.049**，最大 **0.490** | n=10 题，每题 k=5 次 | 每题 k 次裁判分（0/1/2）的总体标准差（ddof=0）；mean/max 为跨题的均值与最大值 |
| 答案相似度 | 均值 **0.948**，最小 **0.453**（cap_006） | n=36/36 题，k=5 | 同题 k 个答案两两余弦相似度（BGE 向量）；mean = 题内均值的均值，min = 题内最小值的最小值；模型 BAAI/bge-large-zh-v1.5 |
| 成功率跨次标准差 | **0.023**（样本，ddof=1）；0.021（总体） | k=5，各 pass 分母 [36, 36, 36, 36, 36] | 各 pass 成功率 [0.778, 0.806, 0.806, 0.778, 0.75] |

轨迹不自洽的题：cap_001, cap_002, cap_007, cap_010, cap_013, cap_022, cap_035, cap_036。
- `cap_001`：[['retrieve', 'retrieve'], ['retrieve', 'retrieve'], ['retrieve'], ['retrieve'], ['retrieve', 'retrieve']]
- `cap_002`：[['retrieve', 'retrieve'], ['retrieve', 'retrieve'], ['retrieve'], ['retrieve', 'retrieve'], ['retrieve', 'retrieve']]
- `cap_007`：[['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve']]
- `cap_010`：[['retrieve'], ['retrieve'], ['retrieve', 'retrieve'], ['retrieve'], ['retrieve', 'retrieve']]
- `cap_013`：[['retrieve'], ['retrieve', 'retrieve'], ['retrieve', 'retrieve'], ['retrieve', 'retrieve'], ['retrieve', 'retrieve']]
- `cap_022`：[['retrieve'], ['retrieve'], ['retrieve'], ['retrieve'], ['retrieve', 'retrieve']]
- `cap_035`：[['retrieve', 'retrieve'], ['retrieve', 'retrieve'], ['retrieve'], ['retrieve'], ['retrieve']]
- `cap_036`：[['retrieve'], ['retrieve', 'retrieve'], ['retrieve', 'retrieve'], ['retrieve'], ['retrieve', 'retrieve']]
判定不自洽的题：cap_016, cap_017。
- `cap_016`：[False, True, True, True, False]；偏离多数派的 pass：[0, 4]
- `cap_017`：[True, True, True, False, False]；偏离多数派的 pass：[3, 4]（全部落在最后两个 pass）
翻转位置与运行顺序的关系：1/2 题的偏离全部落在最后两个 pass；k=5 不足以判定翻转是否与运行顺序相关，需更大的 k。
裁判路逐题 k 次裁判分（0/1/2）与标准差：
- `cap_007`：[1, 1, 1, 1, 1]，std=0.000
- `cap_008`：[2, 2, 2, 2, 2]，std=0.000
- `cap_010`：[2, 2, 2, 2, 2]，std=0.000
- `cap_017`：[2, 2, 2, 1, 1]，std=0.490
- `cap_020`：[1, 1, 1, 1, 1]，std=0.000
- `cap_022`：[2, 2, 2, 2, 2]，std=0.000
- `cap_026`：[2, 2, 2, 2, 2]，std=0.000
- `cap_032`：[1, 1, 1, 1, 1]，std=0.000
- `cap_035`：[1, 1, 1, 1, 1]，std=0.000
- `cap_036`：[1, 1, 1, 1, 1]，std=0.000

### 裁判独立性核验

| 链路 | 请求模型名 | 响应 model | system_fingerprint | 证据 |
|---|---|---|---|---|
| 被测（Agent） | `deepseek-chat` | `deepseek-flash`×180 | `aeb56401ca74e127821c4f9126dcb669`×180 | 本次 180 行运行记录 |
| 裁判（judge_runs） | `deepseek-reasoner` | 未记录 | 未记录 | 本次裁判记录未存响应字段（2026-09-12 起才记录） |
| 探针 | `deepseek-chat` | `deepseek-flash` | `aeb56401ca74e127821c4f9126dcb669` | `stability/raw/model_probe_20260912T025754Z.json`（2026-09-12T02:57:54+00:00，HTTP 200） |
| 探针 | `deepseek-reasoner` | `deepseek-flash` | `aeb56401ca74e127821c4f9126dcb669` | `stability/raw/model_probe_20260912T025754Z.json`（2026-09-12T02:57:54+00:00，HTTP 200） |

探针同一时刻两个请求名是否落在同一 (响应 model, 指纹)：True；端点 `/models` 列出的模型：['deepseek-flash', 'deepseek-v4-pro']。

**本次运行无法确认裁判与被测模型的独立性，开放题判定结果（n=10）应据此折价。**（判断依据：probe）

裁判链路缓存：客户端 none（src/eval/judge.py DeepSeekJudge.__call__ 每次直接 client.chat.completions.create，无本地缓存层）；服务端：裁判记录未存 usage（2026-09-12 起 judge_runs.py 才记录），服务端缓存是否命中无法确认。

### 解读

轨迹自洽率（0.778）低于判定自洽率（0.944）：Agent 走的路径不稳，但结果凑到了同一个判定上——路径层面的非确定性被判定层面掩盖了。
temperature=0 **没有**给出确定性输出：轨迹自洽率 < 1.0 不是 bug，它就是结论本身（PERF_SPEC B3；见 LIMITATIONS.md）。

### 多步检索行为对照（快照单次 vs 本次 k 次）

口径：retrieve 调用次数 = 工具序列里 retrieve 的出现次数（不折叠）；快照为单次运行，本次为 k 次；首次检索 = 快照第一次 retrieve 的 doc_id 列表 vs 本次去重并集的前 4 个。可对照 n=36 题。

| | 快照（2026-09-06，单次） | 本次（k 次运行） |
|---|---|---|
| retrieve 调用次数中位数 | 1.0（n=36 题） | 2.0（n=180 次运行）；题内中位数再取中位数 2.0 |
| 次数分布（次数: 题/运行数） | {0: 1, 1: 18, 2: 11, 3: 2, 4: 1, 6: 1, 8: 1, 9: 1} | {0: 15, 1: 61, 2: 99, 4: 1, 5: 1, 6: 2, 7: 1} |

| 题 | 快照 retrieve 次数 | 本次各 pass 次数 | 本次中位数 | 首次检索 doc_id 相同 | 快照判定 | 本次判定 |
|---|---|---|---|---|---|---|
| cap_001 | 1 | [2, 2, 1, 1, 2] | 2.0 | 否 | True | [True, True, True, True, True] |
| cap_002 | 1 | [2, 2, 1, 2, 2] | 2.0 | 否 | True | [True, True, True, True, True] |
| cap_003 | 2 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_004 | 1 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_005 | 1 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_006 | 1 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_007 | 8 | [5, 7, 6, 4, 6] | 6.0 | 是 | False | [False, False, False, False, False] |
| cap_008 | 2 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_009 | 0 | [0, 0, 0, 0, 0] | 0.0 | 否 | True | [True, True, True, True, True] |
| cap_010 | 6 | [1, 1, 2, 1, 2] | 1.0 | 否 | True | [True, True, True, True, True] |
| cap_011 | 3 | [2, 2, 2, 2, 2] | 2.0 | 否 | True | [True, True, True, True, True] |
| cap_012 | 1 | [1, 1, 1, 1, 1] | 1.0 | 是 | True | [True, True, True, True, True] |
| cap_013 | 1 | [1, 2, 2, 2, 2] | 2.0 | 否 | True | [True, True, True, True, True] |
| cap_014 | 1 | [1, 1, 1, 1, 1] | 1.0 | 是 | True | [True, True, True, True, True] |
| cap_015 | 3 | [1, 1, 1, 1, 1] | 1.0 | 是 | True | [False, False, False, False, False] |
| cap_016 | 2 | [2, 2, 2, 2, 2] | 2.0 | 否 | True | [False, True, True, True, False] |
| cap_017 | 1 | [2, 2, 2, 2, 2] | 2.0 | 否 | True | [True, True, True, False, False] |
| cap_018 | 2 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_019 | 1 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_020 | 4 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [False, False, False, False, False] |
| cap_021 | 1 | [1, 1, 1, 1, 1] | 1.0 | 是 | True | [True, True, True, True, True] |
| cap_022 | 1 | [1, 1, 1, 1, 2] | 1.0 | 否 | True | [True, True, True, True, True] |
| cap_023 | 1 | [1, 1, 1, 1, 1] | 1.0 | 是 | True | [True, True, True, True, True] |
| cap_024 | 1 | [1, 1, 1, 1, 1] | 1.0 | 是 | True | [True, True, True, True, True] |
| cap_025 | 2 | [2, 2, 2, 2, 2] | 2.0 | 否 | True | [True, True, True, True, True] |
| cap_026 | 1 | [1, 1, 1, 1, 1] | 1.0 | 是 | True | [True, True, True, True, True] |
| cap_027 | 2 | [1, 1, 1, 1, 1] | 1.0 | 否 | True | [False, False, False, False, False] |
| cap_028 | 1 | [1, 1, 1, 1, 1] | 1.0 | 否 | True | [True, True, True, True, True] |
| cap_029 | 2 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_030 | 2 | [2, 2, 2, 2, 2] | 2.0 | 否 | True | [True, True, True, True, True] |
| cap_031 | 2 | [2, 2, 2, 2, 2] | 2.0 | 是 | True | [True, True, True, True, True] |
| cap_032 | 2 | [2, 2, 2, 2, 2] | 2.0 | 是 | False | [False, False, False, False, False] |
| cap_033 | 1 | [0, 0, 0, 0, 0] | 0.0 | 否 | True | [True, True, True, True, True] |
| cap_034 | 1 | [0, 0, 0, 0, 0] | 0.0 | 否 | True | [True, True, True, True, True] |
| cap_035 | 2 | [2, 2, 1, 1, 1] | 1.0 | 是 | False | [False, False, False, False, False] |
| cap_036 | 9 | [1, 2, 2, 1, 2] | 2.0 | 否 | True | [False, False, False, False, False] |

首次检索 doc_id 相同但判定不同的题（n=2）：cap_015, cap_020。这类题的检索输入没有变化，差异出在首次检索之后的行为。
- `cap_015`：快照 3 次 retrieve、判定 True；本次 [1, 1, 1, 1, 1] 次、判定 [False, False, False, False, False]；首次检索命中标注证据：False
- `cap_020`：快照 4 次 retrieve、判定 True；本次 [2, 2, 2, 2, 2] 次、判定 [False, False, False, False, False]；首次检索命中标注证据：False

逐题方向（本次题内中位数 vs 快照次数）：减少 10 题 ['cap_007', 'cap_010', 'cap_011', 'cap_015', 'cap_020', 'cap_027', 'cap_033', 'cap_034', 'cap_035', 'cap_036']；不变 18 题；增加 8 题 ['cap_001', 'cap_002', 'cap_004', 'cap_005', 'cap_006', 'cap_013', 'cap_017', 'cap_019']。

**同一代码与同一索引下，retrieve 调用次数中位数由 1.0 变为 2.0；1 题因首次检索未命中后未再检索而失败。**
（「因首次检索未命中后未再检索而失败」的判定条件：有标注证据、快照首次检索未命中、失败的 pass 里 retrieve 次数为 1，且首次检索 doc_id 与快照相同；命中：['cap_015']。放宽「首次检索相同」这一条后命中：['cap_015', 'cap_027']。）

### 对照主评测快照（逐题，同一判定口径）

主评测快照 `reports/latest.json`（2026-09-06T09:28:35+00:00，commit 65bf140）为单次运行，任务成功率 0.917（n=36）；本次 k=5 轮各 pass 成功率均值 0.783。可对照 36 题中，30 题 k 次判定与快照全部一致。

| 题 | 类型 | 快照判定 | 本次 k 次判定 | 快照工具序列 | 本次工具序列（折叠） |
|---|---|---|---|---|---|
| cap_015 | closed | True | [False, False, False, False, False] | ['retrieve'] | [['retrieve'], ['retrieve'], ['retrieve'], ['retrieve'], ['retrieve']] |
| cap_016 | closed | True | [False, True, True, True, False] | ['retrieve'] | [['retrieve'], ['retrieve'], ['retrieve'], ['retrieve'], ['retrieve']] |
| cap_017 | open | True | [True, True, True, False, False] | ['retrieve'] | [['retrieve'], ['retrieve'], ['retrieve'], ['retrieve'], ['retrieve']] |
| cap_020 | open | True | [False, False, False, False, False] | ['retrieve'] | [['retrieve'], ['retrieve'], ['retrieve'], ['retrieve'], ['retrieve']] |
| cap_027 | closed | True | [False, False, False, False, False] | ['retrieve'] | [['retrieve'], ['retrieve'], ['retrieve'], ['retrieve'], ['retrieve']] |
| cap_036 | open | True | [False, False, False, False, False] | ['retrieve'] | [['retrieve'], ['retrieve'], ['retrieve'], ['retrieve'], ['retrieve']] |

两边判定口径相同（判定代码与黄金集在两个 commit 间逐字相同）。快照未记录响应模型名；本次响应模型名见顶部。差异的归因不在本报告范围内；正式比对需重跑主评测（`python -m src.eval.report`）走闸 B。

## 成本

| 项 | 值 | 分母 |
|---|---|---|
| 每次运行平均 token | 2923 | n=180 次运行 |
| 全部 5 轮总 token | 526161（prompt 486068 / completion 40093） | n=180 |
| 跑完一轮完整评测的 token | 105232 | 36 题，每题取 k=5 次均值 |
| 命中服务端缓存的 prompt token | 291944（占 prompt 的 0.601） | n=180 |

按 pass 看缓存与延迟（run_index=0 是首次，之后大概率命中 DeepSeek 服务端 KV 缓存）：

| pass | 平均 token | 平均缓存命中 | 端到端 P50 |
|---|---|---|---|
| 0 | 2793 | 1070 | 2.38s |
| 1 | 3073 | 1632 | 2.69s |
| 2 | 2956 | 1898 | 2.58s |
| 3 | 2752 | 1728 | 2.44s |
| 4 | 3041 | 1781 | 2.55s |

## 门禁（闸 C）

- 成功率跨次标准差 ≤ 0.05：当前 0.023
- 轨迹自洽率 ≥ 0.727：当前 0.778

阈值在 `src/config.py`，首次由本次测量结果确定后固定；未达标只记录、不调低（PERF_SPEC §1）。
