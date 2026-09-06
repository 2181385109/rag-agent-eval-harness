# HANDOFF — v1 封版交接（2026-09-06）

给下一轮对话（或下一个人）的接手说明。**最高约束仍是 [CLAUDE.md](CLAUDE.md)**，
本文件只补充"当前停在哪、什么已定、什么还没定"。

---

## 一、状态：v1（M1–M6）已封版

| 里程碑 | 状态 |
|---|---|
| M1 骨架 / M2 被测 Agent / M3 黄金集+自定义指标 / M4 RAGAS | ✅ |
| M5 LLM-as-Judge + 人工盲标 + kappa | ✅ |
| M6 两道离线回归门禁 + README 定稿 | ✅ |

- `pytest -m "not live"` **309 项全绿**
- `python -m src.eval.report --gate` **PASS**（闸 A 口径闸 + 闸 B 回归闸）
- **尚未推到 GitHub。** 推之前必须先问用户是否愿意公开 `corpus/` 里的私有技术文档。
- v2（安全红队 / FastAPI+Streamlit demo / MLflow / rerank）**一律不碰**，见 CLAUDE.md §11。

### 封版的三个 commit

```
4a52f79  docs: README 定稿（M1–M6 全部完成，指标区留占位）
0282801  feat(eval): 两道离线回归门禁接进 GitHub Actions
b0aec2d  feat(eval): LLM-as-Judge 打分与自动↔人工一致率（kappa）
```

（此前的基线：`daf6152 feat(eval): 标注表支持附上检索片段全文（--full）`）

---

## 二、当前真实指标（全部可复现，不许改动数字）

出处：[reports/report.md](reports/report.md)，对应快照 `reports/eval_20260906T072214Z.json`。
轨迹是同一份 `reports/traces_latest.jsonl`（2026-09-06T03:38 那次真实运行），
之后的两次报告都是 `--from-traces` 重算，没有重跑过 Agent——
所以除任务成功率外的数字与上一份快照逐位相同。

| 指标 | 值 | 分母 / 必须连带说明的限定 |
|---|---|---|
| 黄金集规模 | 36 条 | 自建，不复用公开 benchmark |
| 工具调用准确率（严格） | 0.944 | n=36/36 |
| 检索召回率 recall@k | 0.968 | n=31/36（5 题无 `expected_doc_ids`，不计入） |
| recall@k（仅首次检索） | 0.774 | n=31/36 |
| **任务成功率** | **0.972** | **n=36/36**（闭合题 26 走规则 + 开放题 10 走裁判，门槛=2；唯一失败 cap_007） |
| faithfulness | 0.862 | 35/35 行打满（1 条无检索内容被排除） |
| answer_relevancy | 0.826 | 35/35 |
| context_recall | 0.824 | 35/35 |
| context_precision | 0.670 | 35/35 |
| 完全一致率（人↔裁判） | 0.600 | n=10 |
| 相邻一致率（差 ≤1 档） | 0.900 | n=10 |
| **Cohen's kappa（unweighted）** | **0.216** | **n=10，95% CI [0.091, 0.750]，见待决 ②** |
| Cohen's kappa（quadratic） | 0.103 | n=10，95% CI [-0.000, 0.737] |

RAGAS 那一段在最新报告里是**复用**的（`--reuse-ragas`，同一批轨迹省 40 分钟），
报告里已打「本节为复用，不是本次重算」标记并指向 `eval_20260905T142507Z.json`。

---

## 三、待决问题（②③ 未决，再填简历 bullet）

### ① 任务成功率回填开放题 —— ✅ 已完成（2026-09-06）

口径已拍板并落地：**闭合题 `answer_keys` 全命中；开放题裁判分 ≥2（即满分）才算成功**，
1 分「方向对但要点有遗漏」不计成功。门槛在 `config.OPEN_SUCCESS_THRESHOLD`。

同一批轨迹下：**1.000 (n=26/36) → 0.972 (n=36/36)**。
唯一失败是 `cap_007`（裁判判 1 分：信贷侧缺 70/15/15、航班侧漏了时间序切分与
tail_id GroupKFold）。**这 -0.028 全部来自口径，不含任何模型变化。**

连带做的三件事（细节见 CLAUDE.md §6 修订记录 6/7）：

1. 裁判打分行新增 `answer_sha1` 指纹，回填时逐条核对"这个分是不是给这批轨迹打的"，
   对不上就作废。**现有的 10 行是本机制之前写的，没有指纹**，报告里标为「未核验」——
   下次跑 `judge --score` 就会自动补上（② 若要重跑裁判，这一项顺带就解决了）。
2. 闸 A 的冻结输入新增 `tests/fixtures/gate_judge_scores.jsonl`；
   闸 A 基线 `task_success_rate` 由 `1.000/n=5` 改为 `0.900/n=10`（已重写并提交）。
3. 闸 B 新增「分母变动即 incomparable」。这次 n 从 26 变到 36，
   旧逻辑会打印 `ok -0.0278` 把口径变动当噪声放过去；现在如实标 incomparable。

### ② kappa=0.216 偏低，且 n=10 不构成硬结论（**下一步**）

95% CI `[0.091, 0.750]` 横跨"几乎无一致"到"相当一致"。
三种诚实写法（用户尚未选定）：
1. 直接报 `kappa=0.216 (n=10, 95% CI [0.09, 0.75])`；
2. 改报更稳的一致率（完全 0.600 / 相邻 0.900，n=10）；
3. **先改判据再重跑**（Claude 当时的倾向）——见待决 ③ 的根因，把判据收紧后重跑裁判，
   kappa 大概率上升。这是真做工作不是调数字，但**必须留档改前改后两版**，
   并在报告里写明判据改了什么、为什么。成本：10 次 reasoner 调用，几分钱。

**红线提醒**：不许为了数字好看反复改判据直到 kappa 变漂亮。
CLAUDE.md §6 那条"血的教训"（`expected_doc_ids` 改了三次、recall 每次都升）同样适用于这里。

### ③ 分歧根因：裁判做**结论级**比对，人工做**要点级**比对

四道分歧题里有三道是同一个模式（cap_017 / cap_032 / cap_035，人工判 1、裁判判 2）：
Agent 结论方向都对，但都漏掉参考答案里的一个具体限定——
KS 题缺 `max(TPR−FPR)`；基线题缺"标准化 + L2"；K8s 题把规格书的
"明确禁止、不引入 K8s/Airflow/云服务"弱化成了"全程未提及"。
裁判三次的理由句都是"与参考答案一致"。

**这是两把尺子刻度不同，不是随机噪声**，也是本轮 kappa 偏低的主因。
要收敛就在判据里写死"参考答案中的每个具体限定缺失即降档"，
而不是换模型或调温度。

第四道 `cap_010`（人工 0 / 裁判 2）性质不同：**信息不对等**。
裁判按设计盲看（只给问题+参考答案+Agent 答案），人工标注时手里有检索片段全文、
还 grep 过语料确认「Redis」全库 0 命中。该题 Agent 结论对但第 4 条依据是凭空断言，
裁判在其信息条件下**无从判伪**。这条限制已写进报告的一致率一节。

完整叙述见 [data/agreement_interpretation.md](data/agreement_interpretation.md)（随报告落盘）。

---

## 四、接手常用命令

```bash
# 门禁（离线，不花钱）——改任何指标代码后第一件事
python -m src.eval.report --gate

# 全量评测（要 key、要钱、约 6 分钟；加 --ragas 约 40 分钟）
python -m src.eval.report --ragas

# 只补算指标，不重跑 Agent（复用轨迹，省钱）
python -m src.eval.report --from-traces reports/traces_latest.jsonl \
    --reuse-ragas reports/eval_20260905T142507Z.json

# 裁判打分 / 算一致率
python -m src.eval.judge --score      # 10 次 reasoner 调用
python -m src.eval.judge --kappa      # 纯离线

# 有意变更指标口径时（会在提交历史里留痕）
python -m src.eval.report --write-gate-baseline

# 任务成功率退回"只算闭合题"的旧口径（对照用，不改基线）
python -m src.eval.report --from-traces reports/traces_latest.jsonl --no-judge-backfill
```

**本机注意**：跑测试要用 `./.venv/Scripts/python.exe -m pytest`（不能用全局 `python`）；
中文 CLI 需 `PYTHONIOENCODING=utf-8`（Windows 控制台 cp936）。

---

## 五、容易踩的坑（都已踩过一次）

- **`reports/traces_*.jsonl` 被 gitignore**，所以 CI 门禁不能依赖它，
  必须走 `tests/fixtures/gate_traces.jsonl` 那份冻结轨迹。
- **分母陷阱**：RAGAS 单个 job 失败会留 NaN 而 `mean()` 默认跳过它。
  看到 0.000 这种漂亮差值，先查"实际打分行数"。
- **闸 A 是等值不是容差**，涨了也拦——输入冻结时数字没理由变。
- **裁判分与轨迹必须配对**：重跑 Agent 后旧裁判分就作废了（指纹对不上会被丢弃并在
  报告里列出）。看到"已作废 N 条"就是该重跑 `judge --score` 了，别忽略它。
- **kappa 退化情形打印「未定义」，绝不落成 0.0**。
- **标注纪律**：改 `expected_doc_ids` 只能依据 `data/annotation_criteria.md` 的书面判据，
  不许看着检索结果反推；确需修订就对全部题目重扫一遍并在报告里写明差值与归因。
- **README 与简历 bullet 的指标区仍是占位**，等上面三个待决拍板后一次性填真值。
