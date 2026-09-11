# 稳定性与延迟分解报告（脚本生成，勿手改）

> 本文件由 `python -m stability.analyze` 从 `stability/raw/run_20260911T162531Z.jsonl` 生成；
> 原始产物 sha256 = `bdf6b3018bfbe633adae7ac39f0a82078fdb6e0b02a5cf7579e5107fde4c8240`；分析时间 2026-09-11T16:53:30+00:00。
> 每个数字旁都带样本量 n、重复次数 k 与环境标识（PERF_SPEC §0.3）。

## 环境与随机性来源

| 项 | 值 |
|---|---|
| 被测模型 | 请求 `deepseek-chat`；响应 model 字段：['deepseek-flash']；system_fingerprint：['aeb56401ca74e127821c4f9126dcb669'] |
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

n=180（36 题 × k=5），temperature=0.0，model=deepseek-chat；有效行 n_ok=180，报错行 n_error=0，错误率 0.000。

| 段 | n | P50 | P95 | P99 | 均值 | 最大 | 占比中位数 |
|---|---|---|---|---|---|---|---|
| 端到端 | 180 | 2.52s | 3.72s | 6.96s | 2.65s | 7.23s | 1.00 |
| LLM 调用 | 180 | 2.29s | 3.44s | 5.96s | 2.42s | 6.41s | 0.91 |
| 检索（BGE+FAISS） | 180 | 0.27s | 0.32s | 0.82s | 0.23s | 1.02s | 0.09 |
| 工具（calc） | 180 | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00 |
| 编排开销 | 180 | 0.00s | 0.00s | 0.00s | 0.00s | 0.01s | 0.00 |

单次 LLM 调用（n=373 次）：P50 1.06s / P95 2.11s / P99 3.05s；每次运行 LLM 调用次数 P50 2.0、最大 4。

**结论**：LLM 调用占端到端延迟的 91.4%（占比中位数），检索占 8.6%，工具占 0.0%，编排开销占 0.1%。

## 稳定性（三个层级，逐级放宽）

| 指标 | 值 | 分母 | 口径 |
|---|---|---|---|
| 轨迹自洽率（严格） | **0.778** | n=36/36，k=5 | k 次运行的工具调用序列逐项完全一致的题占比（不折叠连续重复） |
| 轨迹自洽率（折叠连续重复） | 1.000 | n=36/36，k=5 | 参考口径 |
| 判定自洽率 | **0.944** | n=36/36（规则 26、裁判 10），k=5 | k 次运行的任务成功判定全部相同的题占比；闭合题走 answer_keys 规则，开放题走逐次裁判分（>= 2 为成功），没有逐次裁判分的开放题不计入分母；报错的运行计为失败 |
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
- `cap_016`：[False, True, True, True, False]
- `cap_017`：[True, True, True, False, False]

### 解读

轨迹自洽率（0.778）低于判定自洽率（0.944）：Agent 走的路径不稳，但结果凑到了同一个判定上——路径层面的非确定性被判定层面掩盖了。
temperature=0 **没有**给出确定性输出：轨迹自洽率 < 1.0 不是 bug，它就是结论本身（PERF_SPEC B3；见 LIMITATIONS.md）。

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

判定口径两边相同，差异只能来自被测模型的行为变化或裁判波动——这正是回归闸（闸 B）要抓的漂移；要确认，需重跑一次主评测（`python -m src.eval.report`）让闸 B 正式比对。

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
