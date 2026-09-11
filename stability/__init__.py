"""性能 / 稳定性测量（PERF_SPEC §3，任务 B）。

不做并发压测（LLM API 有速率限制且花钱）。测的是**同一输入重复调用的一致性**
与端到端延迟的三段分解（检索 / LLM / 工具）。

    python -m stability.run_repeat   # 真实调 API：36 题 × k=5，落 stability/raw/run_*.jsonl
    python -m stability.analyze      # 离线：原始产物 -> summary.json + report.md
    python -m stability.gate         # 离线：闸 C（阈值 + 可复现）
    python -m stability              # 以上三步串起来（= make stability）

铁律（PERF_SPEC §1）：报告里不许出现手敲数字；不许只报均值；不许因为结果不好看而
重跑、删除或挑选运行记录；指标未达标不许调低门槛。
"""
