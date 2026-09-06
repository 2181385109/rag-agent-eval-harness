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

出处：[reports/report.md](reports/report.md)，对应快照 `reports/eval_20260906T092835Z.json`
（commit `65bf140` 生成，git_commit 字段与生成它的代码所在提交逐位一致，已核对）。
轨迹是同一份 `reports/traces_latest.jsonl`（2026-09-06T03:38 那次真实运行），
之后的报告都是 `--from-traces` 重算，没有重跑过 Agent——
所以除任务成功率、RAGAS 复用标记外的数字与上一份快照逐位相同。

| 指标 | 值 | 分母 / 必须连带说明的限定 |
|---|---|---|
| 黄金集规模 | 36 条 | 自建，不复用公开 benchmark |
| 工具调用准确率（严格） | 0.944 | n=36/36 |
| 检索召回率 recall@k | 0.968 | n=31/36（5 题无 `expected_doc_ids`，不计入） |
| recall@k（仅首次检索） | 0.774 | n=31/36 |
| **任务成功率** | **0.917** | **n=36/36**（闭合题 26 走规则 + 开放题 10 走裁判 v2，门槛=2；失败 cap_007/032/035） |
| faithfulness | 0.862 | 35/35 行打满（1 条无检索内容被排除） |
| answer_relevancy | 0.826 | 35/35 |
| context_recall | 0.824 | 35/35 |
| context_precision | 0.670 | 35/35 |
| 完全一致率（人↔裁判） | 0.800 | n=10 |
| 相邻一致率（差 ≤1 档） | 0.900 | n=10 |
| **Cohen's kappa（unweighted）** | **0.623** | **n=10，95% CI [0.231, 1.000]；上升部分是构造性的，见下** |
| Cohen's kappa（quadratic） | 0.324 | n=10，95% CI [0.000, 1.000] |

RAGAS 那一段在最新报告里是**复用**的（`--reuse-ragas`，同一批轨迹省 40 分钟），
且复用前提已**机器核验**（不再是口头断言）：`report.verify_ragas_reuse` 对基线快照
`eval_20260905T142507Z.json` 的 per_question（answer + retrieved_doc_ids）与当前
`traces_latest.jsonl` 逐题算指纹比对，**35/35 条一致**。报告里相应从
「⚠ 前提是轨迹未变」改成了「✅ 已验证轨迹指纹一致（35 条），复用成立」。

---

## 三、三个待决问题 —— 全部已处理（①②③）

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

### ② kappa 偏低 —— ✅ 已处理（2026-09-06，走了 ③ 那条路）

选的是"先改判据再重跑"。判据由 **v1 结论级** 收紧为 **v2 要点级**，重跑裁判，
改前改后两版全部留档。详见 CLAUDE.md §6 修订记录 8/9 与
[reports/report.md](reports/report.md) 的「裁判判据修订对照」一节。

| 量 | 改前(v1) | 改后(v2) | 差 |
|---|---|---|---|
| Cohen's kappa（unweighted） | 0.216 | **0.623** | +0.407 |
| Cohen's kappa（quadratic） | 0.103 | 0.324 | +0.222 |
| 完全一致率 | 0.600 | **0.800** | +0.200 |
| 相邻一致率 | 0.900 | 0.900 | ±0 |
| **任务成功率**（连带） | 0.972 | **0.917** | **-0.056** |

- 改判 2 题：`cap_032`、`cap_035` 由 2 降到 1，**两题改后都与人工判定相同**。
- `cap_017` 未改判 —— 残余分歧，且存疑的是**人工**那一侧
  （`max(TPR−FPR)` 与"累积分布之差的最大值"数学上是同一件事，
  人工把它当成独立的第二项限定，比 v2 判据更严）。**本轮不动它。**
- `cap_010` 的 2 档差距仍在，是信息不对等（裁判盲看，抓不到需回查语料才能识别的编造），
  判据层面解决不了，属 v2 范畴。

**引用 kappa 时必须连带说的两条限定（不许省）：**

1. **上升有相当一部分是构造性的。** v2 本质是让裁判向人工的既有尺子靠拢——
   人工本来就是要点级标注的（写在 v1 那轮解读里，早于本次修订，git 可查证）。
   诚实的说法是「判据澄清后自动裁判能复现人工的判定口径」，
   **不是**「两个独立评分者达成了实质一致」。要测后者得让人工在 v2 判据下重新盲标，本轮没做。
2. **区间反而更宽了。** 95% CI 由 `[0.091, 0.750]` 变成 `[0.231, 1.000]`，上界顶到 1.0。
   n 没变、分歧从 4 题减到 2 题，bootstrap 更容易抽到"零分歧"样本。
   **区间的收窄程度配不上点估计的涨幅**，n=10 仍不构成硬结论。

**红线已执行**：判据只改这一轮，改完认结果。任务成功率跌了 0.056 也照实记，没有回调判据。
往后若要再动判据，必须是新的、写得出书面理由的问题，不能是"kappa 还不够好看"。

### ③ 分歧根因 —— ✅ 已处理（随 ② 一并解决）

根因就是 ② 里那条：**裁判做结论级比对、人工做要点级比对**，两把尺子刻度不同。
v2 判据把「要点」定死成"参考答案里的每一项具体限定"，并强制"先列清单再逐项核对"，
`cap_032`/`cap_035` 因此归位。裁判的理由句也从"与参考答案一致"变成逐项点名缺失，
说明它确实在执行清单核对，而不是换个说法给同样的分。

余下两条不是判据问题：`cap_017` 是"什么算一项独立限定"的真实歧义，
`cap_010` 是信息不对等。两条都写进了 [data/agreement_interpretation.md](data/agreement_interpretation.md)。

---

## 三点五、公开上架前的安全排查与脱敏决定（2026-09-06）

### 排查结果

- 全仓 + 全部历史提交扫 `sk-[a-zA-Z0-9]{20,}`：**零命中**。真实 key 只存在于未跟踪的
  `.env`，且 `git log --all --full-history -- .env` 为空——**从未进入过任何一次提交**。
- 已跟踪文件逐一比对 `.gitignore`：零文件命中忽略规则，无需 `git rm --cached`。
- 邮箱 / 手机号 / 身份证：零命中（此前"疑似手机号"实为指标小数，如 `0.7741935483870968`）。

### 已脱敏（commit `87fdada`）

本机路径统一替换为占位符：

| 原文 | 替换为 |
|---|---|
| `D:\xiangmu\credit-risk-mlops` | `<project-root>（credit-risk-mlops）` |
| `D:\xiangmu\flight-delay-scheduling` | `<project-root>（flight-delay-scheduling）` |
| 裸 `D:\xiangmu` | `<workspace-root>` |

覆盖：`corpus/` 两份规格书（**源头**，今后重建索引 / 重跑 Agent 都不会再带出该路径）、
`reports/human_labeling_sheet*.md`、`data/human_labels.jsonl` 的 answer 字段、
以及两份报告 JSON 里 `backfill.source` 的绝对路径（改为相对路径）。

### 刻意保留未脱敏的 5 处 —— 这是决定，不是遗漏

`reports/eval_20260905T041922Z.json`、`eval_20260905T142507Z.json`、
`eval_20260906T033855Z.json`、`latest.json`、`tests/fixtures/gate_traces.jsonl`
里仍有 `D:\xiangmu\credit-risk-mlops` 字符串。它来自 **cap_035 的一次真实 DeepSeek 调用**
——模型在答案里逐字引用了语料原文。保留的三条理由：

1. **良性、非密钥。** 它是一个本机开发路径，且 `credit-risk-mlops` 这个仓库名本就公开，
   公开的边际风险≈0。
2. **`gate_traces.jsonl` 与 `gate_judge_scores.jsonl` 的 `answer_sha1` 是一对冻结锚点，
   价值就在冻结。** 那套指纹是专为防止"裁判分套错轨迹"而加的。为一个无害字符串去改
   fixture 文本、再回改指纹让它重新对上，**形状上等同于"朝着想要的结果调指纹"**——
   正是本项目通篇在证明自己不做的事（见 §6 那条"血的教训"）。红线不破。
3. 若强行脱敏而不同步改指纹，闸 A 会因指纹失配丢弃该条裁判分、改变
   `task_success_rate` 的分母，`--gate` 相对已提交基线直接 FAIL。

**给后来者**：看到公开仓库里这 5 处路径，不必"顺手修掉"。要动它，先读懂上面第 2 条。

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
python -m src.eval.judge --score      # 10 次 reasoner 调用（会覆盖 judge_scores.jsonl，先留档！）
python -m src.eval.judge --kappa      # 纯离线

# 改判据后出改前/改后对照（离线；--compare-judge 指向留档的那份旧分）
python -m src.eval.report --from-traces reports/traces_latest.jsonl     --reuse-ragas reports/eval_20260905T142507Z.json     --compare-judge reports/judge_scores_v1_conclusion_level.jsonl

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
- **README 与简历 bullet 的指标区已填真值**（2026-09-06）。CLAUDE.md §12 与 README
  「## 指标」都已按 `reports/eval_20260906T092835Z.json` 填入，且逐项标注了对应
  report.md 的哪一节——多轮一致性那格**照实留白**（当前快照未含一致性重跑），
  没有为了填满表格去凑一个不在这份快照里的数字。
