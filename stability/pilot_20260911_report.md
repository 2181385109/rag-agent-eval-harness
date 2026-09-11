# 稳定性与延迟分解报告（脚本生成，勿手改）

> 本文件由 `python -m stability.analyze` 从 `stability/raw/run_20260911T161706Z.jsonl` 生成；
> 原始产物 sha256 = `c1e274d7edfaf815b4a4a4cbd6300976ddb1039508567fc325f1d1f2f7afd2d8`；分析时间 2026-09-11T16:22:03+00:00。
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
| 运行时间 | 2026-09-11T16:17:06+00:00 → 2026-09-11T16:18:58+00:00 |
| 顺序 | pass-major（先把所有题跑完第 0 次，再第 1 次……） |
| 硬件 | AMD Ryzen 7 7840H  w/ Radeon 780M Graphics × 16 核，RAM 15.3 GB |
| 软件 | Windows 11 (10.0.26200)；Python 3.12.3；openai 2.54.0；sentence-transformers 6.0.1 |
| 网络 | 压测客户端与 DeepSeek API 之间为公网；LLM 延迟含网络往返与服务端排队 |

## 延迟分解

n=25（5 题 × k=5），temperature=0.0，model=deepseek-chat；有效行 n_ok=25，报错行 n_error=0，错误率 0.000。

| 段 | n | P50 | P95 | P99 | 均值 | 最大 | 占比中位数 |
|---|---|---|---|---|---|---|---|
| 端到端 | 25 | 2.76s | 7.92s | 10.55s | 3.50s | 11.31s | 1.00 |
| LLM 调用 | 25 | 2.50s | 7.12s | 9.12s | 3.18s | 9.68s | 0.92 |
| 检索（BGE+FAISS） | 25 | 0.26s | 0.82s | 1.43s | 0.32s | 1.62s | 0.08 |
| 工具（calc） | 25 | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00 |
| 编排开销 | 25 | 0.00s | 0.01s | 0.01s | 0.00s | 0.01s | 0.00 |

单次 LLM 调用（n=66 次）：P50 1.07s / P95 2.85s / P99 3.79s；每次运行 LLM 调用次数 P50 2.0、最大 6。

**结论**：LLM 调用占端到端延迟的 91.8%（占比中位数），检索占 8.1%，工具占 0.0%，编排开销占 0.1%。

## 稳定性（三个层级，逐级放宽）

| 指标 | 值 | 分母 | 口径 |
|---|---|---|---|
| 轨迹自洽率（严格） | **0.400** | n=5/5，k=5 | k 次运行的工具调用序列逐项完全一致的题占比（不折叠连续重复） |
| 轨迹自洽率（折叠连续重复） | 1.000 | n=5/5，k=5 | 参考口径 |
| 判定自洽率 | **1.000** | n=5/5（规则 3、裁判 2），k=5 | k 次运行的任务成功判定全部相同的题占比；闭合题走 answer_keys 规则，开放题走逐次裁判分（>= 2 为成功），没有逐次裁判分的开放题不计入分母；报错的运行计为失败 |
| 答案相似度 | 均值 **0.895**，最小 **0.453**（cap_006） | n=5/5 题，k=5 | 同题 k 个答案两两余弦相似度（BGE 向量）；mean = 题内均值的均值，min = 题内最小值的最小值；模型 BAAI/bge-large-zh-v1.5 |
| 成功率跨次标准差 | **0.000**（样本，ddof=1）；0.000（总体） | k=5，各 pass 分母 [5, 5, 5, 5, 5] | 各 pass 成功率 [0.6, 0.6, 0.6, 0.6, 0.6] |

轨迹不自洽的题：cap_001, cap_007, cap_035。
- `cap_001`：[['retrieve'], ['retrieve'], ['retrieve', 'retrieve'], ['retrieve', 'retrieve'], ['retrieve', 'retrieve']]
- `cap_007`：[['retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve'], ['retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve', 'retrieve']]
- `cap_035`：[['retrieve'], ['retrieve', 'retrieve'], ['retrieve'], ['retrieve', 'retrieve'], ['retrieve']]
判定不自洽的题：无。

### 解读

轨迹自洽率（0.400）低于判定自洽率（1.000）：Agent 走的路径不稳，但结果凑到了同一个判定上——路径层面的非确定性被判定层面掩盖了。
temperature=0 **没有**给出确定性输出：轨迹自洽率 < 1.0 不是 bug，它就是结论本身（PERF_SPEC B3；见 LIMITATIONS.md）。

## 成本

| 项 | 值 | 分母 |
|---|---|---|
| 每次运行平均 token | 6122 | n=25 次运行 |
| 全部 5 轮总 token | 153051（prompt 145110 / completion 7941） | n=25 |
| 跑完一轮完整评测的 token | 30610 | 5 题，每题取 k=5 次均值 |
| 命中服务端缓存的 prompt token | 91520（占 prompt 的 0.631） | n=25 |

按 pass 看缓存与延迟（run_index=0 是首次，之后大概率命中 DeepSeek 服务端 KV 缓存）：

| pass | 平均 token | 平均缓存命中 | 端到端 P50 |
|---|---|---|---|
| 0 | 3757 | 1843 | 2.29s |
| 1 | 10814 | 6938 | 2.83s |
| 2 | 5298 | 3021 | 3.14s |
| 3 | 5426 | 3174 | 2.45s |
| 4 | 5315 | 3328 | 2.16s |

## 门禁（闸 C）

- 成功率跨次标准差 ≤ 0.05：当前 0.000
- 轨迹自洽率 ≥ 0.8：当前 0.400

阈值在 `src/config.py`，首次由本次测量结果确定后固定；未达标只记录、不调低（PERF_SPEC §1）。
