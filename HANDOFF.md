# HANDOFF — 交接说明

给下一轮对话（或下一个人）的接手说明。**最高约束是 [CLAUDE.md](CLAUDE.md)**（含 2026-09-12
追加的 §13 测量与发布纪律），本文件只补充"当前停在哪、什么已定、什么还没定"。

**先读 [§零（2026-09-12 现状）](#零2026-09-12-现状先读这一节)**——它描述工作区里未提交的
改动与本次调查的结论。§一～§五是 2026-09-06 v1 封版时的记录，其中的指标数字（0.917 等）
是**当时**的快照值，已被 §零 的事实覆盖，不要直接引用。

---

## 零、2026-09-12 现状（先读这一节）

### 0.0 已提交的部分（HEAD = `56021c7`）

在 v1（§一～§五）之上，2026-09-11/12 已提交四个 commit（`0aa2d57`…`56021c7`）：PERF_SPEC 任务 B 的
稳定性/延迟测量包 `stability/`（`run_repeat` 每题 k=5 重复运行、`judge_runs` 开放题逐次裁判、
`analyze` 出 `stability/summary.json` + `report.md`、`gate` 闸 C）、其原始产物
`stability/raw/run_20260911T162531Z.jsonl`（36 题 × 5，180 行）与 `judge_20260911T163709Z.jsonl`
（50 行）、`LIMITATIONS.md`、`Makefile`、README 的「闸 C」一节。入口：`python -m stability`
（= `make stability`，本机无 make）；只重算不重跑：`python -m stability.analyze --raw … --judge …`。
规格文件在仓库外（本机 `PERF_SPEC.md`，路径不入库）。

### 0.1 本轮改动清单（已于 2026-09-12 提交并推送）

下表是 2026-09-12 这一轮在 `56021c7` 之上的全部改动，已拆成三个 commit 推到 `origin/main`：
`6f7fc94 feat(eval)` 工具与机制、`467c3b5 data(eval)` 本次调查的产物与结论、`b23014a docs` 文档。
拆开是为了不让 5511 行的 latest.json 淹没机制改动。每个 commit 都在干净 worktree 里单独验证过
（pytest 全绿、`--gate` PASS、安全扫描 PASS）。表中「状态」列是相对 `56021c7` 的 M / 新：

| 文件 | 状态 | 改了什么 |
|---|---|---|
| `CLAUDE.md` | M | 追加 §13：报告数字脚本生成、不得因结果重跑/换口径、不得调低阈值、push 前安全扫描原文 |
| `HANDOFF.md` | M | 本文；旧 §六（PERF 任务 B 交接）已删，内容并入本节 |
| `LIMITATIONS.md` | M | 第 4/4a/4b/5 条改写：模型归属、裁判独立性、裁判缓存未记录、缓存拆分 n=0；补 09-12 重跑事实 |
| `src/eval/judge.py` | M | `DeepSeekJudge` 新增 `last_response_meta`（响应 model / 指纹 / 缓存命中），供调用方逐条记录；加 `_usage_field` |
| `src/eval/report.py` | M | `find_previous_snapshot` 只认 `eval_YYYYMMDDTHHMMSSZ.json`，另存的对照快照不再被当成闸 B 基线；加 `SNAPSHOT_NAME_RE`；加 `repo_relative`，`backfill.source` / `reused_from` 记仓库相对路径（posix），不再写本机绝对路径 |
| `stability/records.py` | M | `Latency` 加逐次调用缓存命中/未命中列表；`RunRecord` 加 `response_models` / `system_fingerprints`（逐条，旧记录为空列表） |
| `stability/instrument.py` | M | 探针逐条记录每次 LLM 调用的响应 model / 指纹 / 缓存命中（此前只记首条） |
| `stability/run_repeat.py` | M | 把上述逐条字段写进 RunRecord |
| `stability/judge_runs.py` | M | 裁判行记录响应 model / 指纹 / usage（此前没有） |
| `stability/analyze.py` | M | 新增：模型归属段+顶部告警、裁判独立性核验段、缓存拆分（n=0 时改一句话）、按 pass 单次调用表+两项检查、判定自洽率分路+裁判分标准差、翻转位置、多步检索行为对照段；标题改"请求名/响应名"；快照对照段去掉归因句 |
| `stability/gate.py` | M | 可复现闸纳入 `model_attribution` 与探针文件；`retrieval_persistence` 与 `main_snapshot_comparison` 不重算（依赖 gitignore 的 traces） |
| `stability/probe_models.py` | 新 | 模型归属探针：对 `MODEL_NAME` / `JUDGE_MODEL_NAME` 各发一条最小请求，落 `stability/raw/model_probe_*.json` |
| `stability/rerun_main_eval.py` | 新 | 主评测对照重跑：单次、另存 `reports/eval_{name}.json`、不写 latest.json；`--ragas` 时给 RAGAS 的 ChatOpenAI 挂探针；`meta.served` 逐条记被测 / 裁判 / RAGAS 三条链路的响应 model / 指纹；RAGAS 前先落中间快照 |
| `stability/rerun_report.py` | 新 | 对照重跑报告（脚本生成 markdown）：头部指标并列、RAGAS 原始口径与交集口径并列（逐题分缺失即「无法重算」）、三条事实（分母耦合 / 检索次数 vs recall / 独立性范围）、逐题判定变化。基线轨迹默认取与基线快照同名的 `traces_{name}.jsonl` 并逐题核对（答案 + retrieved_doc_ids），对不上即「未记录」——不再默认 `traces_latest.jsonl`（第 5 步同步 latest 时曾因此把 09-12 轨迹当成基线，靠 sha256 对比发现，已修） |
| `scripts/security_scan.py` / `scripts/__init__.py` | 新 | CLAUDE.md §13.4 的 push 前安全扫描：逐条打印命中原文，`EXEMPTIONS` 里的豁免每条带出处；`--history` 扫全部提交 |
| `stability/raw/model_probe_20260912T025754Z.json` | 新 | 探针产物（两个请求名 → 同一响应 model / 指纹；`/models` 列表） |
| `stability/summary.json` / `stability/report.md` | M | 由同一批 09-11 原始产物 + 09-11 裁判分 + 探针重新生成（**没有重跑 180 次**） |
| `reports/eval_20260912_rerun.json` | 新 | 09-12 主评测对照重跑快照（见 0.2）；`backfill.source` 已由本机绝对路径改为 `reports/judge_scores_20260912_rerun.jsonl` |
| `reports/eval_20260912T074455Z_full.json` | 新 | 09-12 **全量**重跑快照（36 题单次 + 裁判 + RAGAS 四项；三条链路响应 model / 指纹逐条在 `meta.served`）；`backfill.source` 同样已改为相对路径。名字带 `_full`，按设计不作闸 B 基线 |
| `reports/judge_scores_20260912T074455Z_full.jsonl` | 新 | 该重跑的 10 道开放题裁判分（cap_020 解析失败重问一次，共 11 次调用；指纹 10/10 核验） |
| `reports/report_20260912T074455Z_full.md` | 新 | 上述快照 vs 09-06 快照的对照报告，`python -m stability.rerun_report --snapshot reports/eval_20260912T074455Z_full.json` 生成，重跑逐字节相同 |
| `reports/eval_20260912T081718Z.json` | 新 | 全量重跑快照的规范名复制件（与 `_full` 原件、`latest.json` 三者 sha256 相同），闸 B 据此工作 |
| `reports/latest.json` / `reports/report.md` | M | latest 指向 09-12 全量重跑；report.md 由其重新渲染（见 0.6） |
| `reports/gate_fail_20260912T081718Z.txt` | 新 | 闸 B 首次拦下真实落差的 `--gate` 原文（不得删除或覆盖）；`.gitignore` 第 2 段加白名单 `!reports/gate_fail_*.txt` |
| `reports/gate_accepted_regressions.json` | 新 | 闸 B 已接受的回归登记（1 条：task_success_rate 0.917→0.750，2026-09-12，归因不可行） |
| `.gitignore` | M | 上述白名单一行 |
| `README.md` / `CLAUDE.md` | M | 指标区 / §12：所有历史数字带测量日期与复现状态（09-06 vs 09-12 两列），出处列改为快照字段；README 加「脱敏与历史」「拦下之后怎么办」；CLAUDE.md §13.4 corpus 已公开 |
| `reports/judge_scores_20260912_rerun.jsonl` | 新 | 该重跑的 10 道开放题裁判分（带响应字段，指纹 10/10 核验） |
| `tests/test_stability_analyze.py` | M | +14 条口径测试（缓存拆分、pass 0 规则、分路、翻转位置、归属、独立性措辞、检索对照） |
| `tests/test_gate.py` | M | +1 条：非流水线命名的快照不作闸 B 基线；已接受回归机制 +16 条（登记只对那一对快照/指标/数值生效、缺日期/原因/接受人即报错、登记项指向存在的快照、门禁输出打印原因）；回归断言改为「无新增 dropped」 |
| `tests/test_rerun_report.py` | 新 | +10 条：交集口径缺逐题分即无法重算、分母耦合、检索次数 vs recall、独立性范围、渲染措辞 |
| `tests/test_report.py` | M | +3 条：溯源路径记仓库相对路径；+3 条：请求名/响应名并列打印、未记录不回填、agreement 缺失要说出来 |
| `tests/test_secrets.py` | M | +3 条：安全扫描零未豁免命中、豁免必须带出处、扫描器能抓到样本 |
| `reports/traces_20260912_rerun.jsonl` / `reports/traces_20260912T074455Z_full.jsonl` | gitignore | 两次重跑的完整轨迹（不入库） |

验证状态：`pytest -m "not live"` **395 passed**；`python -m scripts.security_scan` 未豁免命中 0；`python -m src.eval.report --gate` 见第 3 步（本轮尚未重跑）。

### 0.2 本次调查的事实结论（只列事实）

**模型归属**
- 本仓库所有 LLM 调用的 `model` 参数都来自 `src/config.py` 的硬编码常量（被测 `deepseek-chat`、
  裁判 `deepseek-reasoner`），无环境变量覆盖；我方没有传错。
- 这两个请求名已于 2026-07-24 被 DeepSeek 官方弃用，现由旧名路由至 `deepseek-flash`（V4.1 Flash）。
  09-11 的 180 次运行、试点 25 次、09-12 的重跑 36 次与探针 2 次，响应 `model` 字段**全部**为
  `deepseek-flash`，`system_fingerprint` 全部为 `aeb56401ca74e127821c4f9126dcb669`。
  端点 `GET /models` 只列出 `deepseek-flash` 与 `deepseek-v4-pro`。
- 裁判链路请求 `deepseek-reasoner`，09-12 探针与重跑记录的响应 model / 指纹与被测链路**相同**
  （`deepseek-reasoner` 的响应仍含 `reasoning_tokens`）。固定措辞：
  **「本次运行无法确认裁判与被测模型的独立性，开放题判定结果（n=10）应据此折价。」**

**指标不可复现**
- 2026-09-06 快照（`reports/latest.json` = `eval_20260906T092835Z.json`）：任务成功率 0.917。
  2026-09-12 同一代码、同一磁盘索引单次重跑（`reports/eval_20260912_rerun.json`）：**0.750**（n=36/36）。
- 判定变化恰好 6 题、全部 True→False：`cap_015/016/017/020/027/036`，与 09-11 稳定性运行里判定
  变化的是同一组题。recall 并集 0.968 → 0.871；**仅首次检索的 recall 不变（0.774）**；
  tool_accuracy 0.944 → 1.000；retrieve 调用次数中位数 1.0 → 2.0（升）。
- 已排除的因素：两个 commit（`65bf140`↔`78b5122`/`56021c7`）间 `src/`、黄金集、判定规则、裁判判据
  逐字相同（md5 相同）；`index/`（09-04 建）与 `65bf140` 的语料切分逐块相同且未重建，
  两次运行的检索输入相同；判定口径相同。
- **归因不可能**：09-06 快照及其轨迹只记了请求名，没有记录响应 model / 指纹，
  无法知道当时落在哪个后端。这是记录缺口，不是可以推断的事。

### 0.3 三道闸：当前阈值与来历

| 闸 | 判据 | 阈值 / 基线 | 怎么定的 | 本次是否新定 |
|---|---|---|---|---|
| A 口径闸 | 冻结子集（`tests/fixtures/`，10 题）重算五个指标，与 `reports/gate_baseline.json` **逐位相等** | 基线：recall 0.833(6/10)、首次 recall 0.667、任务成功率 0.8(10/10)、工具 0.8、宽松工具 0.8；commit `5521864`，2026-09-06 | 输入冻结，数字没理由变；改口径必须显式 `--write-gate-baseline` | 否 |
| B 回归闸 | `reports/latest.json` vs 上一份 `eval_YYYYMMDDTHHMMSSZ.json`；`task_success_rate` / `tool_accuracy` / `faithfulness` 跌超容差即 fail；分母/判据版本/题数变则 incomparable | `METRIC_DROP_TOLERANCE = 0.05` | CLAUDE.md §7 给定 | 否（本次只收紧了"上一份快照"的文件名匹配） |
| C 稳定性闸 | `stability/summary.json`：成功率跨 k 次样本标准差 ≤ 阈值；轨迹自洽率（严格口径）≥ 阈值；且 summary 除答案相似度/快照对照/检索对照三段外必须能由 `stability/raw/` + 裁判分 + 探针重算逐位相等 | `STABILITY_MAX_SUCCESS_RATE_STD = 0.05`（规格给定）；`STABILITY_MIN_TRAJECTORY_CONSISTENCY = 0.727` | 0.05 来自 PERF_SPEC；0.727 = 首次全量实测 0.7778 − 0.05（与闸 B 同容差），规则在看到全量数字**之前**拍板，规格提议的 0.8 未达标已记 LIMITATIONS | **是**（2026-09-12 建，已冻结，不许调低） |

2026-09-12 更新：全量重跑快照已复制为规范名 `eval_20260912T081718Z.json`（名字取自其 `meta.timestamp_utc`，
与原件逐字节一致）并写入 `latest.json`；闸 B 随即报 `task_success_rate dropped (-0.1667)`，原文存档
`reports/gate_fail_20260912T081718Z.txt`（不得删除或覆盖）。该下跌已在 `reports/gate_accepted_regressions.json`
登记为已接受的回归（日期 / 原因 / 接受人，只对 `eval_20260912T081718Z.json` vs `eval_20260906T092835Z.json`
这一对生效），`--gate` 与 `tests/test_gate.py` 改为「相对已接受基线无新增 dropped」。闸 B 对 RAGAS 分母
不敏感的问题见 LIMITATIONS 第 19 条。`eval_20260912T074455Z_full.json` 与 `eval_20260912T081718Z.json` 内容相同
（sha256 一致）为**有意保留**：前者是原始产物名（`rerun_main_eval` 按启动时刻命名），后者是流水线规范名
（按 `meta.timestamp_utc`）供闸 B 识别；删任一份都会断掉一条可追溯链路。

### 0.4 已知未解释现象（禁止编解释）

09-11 稳定性运行里，pass 0（同题首轮、无跨轮服务端缓存）端到端 P50 = 2.38s，**低于**后续四个
pass（2.44–2.69s）。已核查：各 pass 单次 LLM 调用 P50 接近（相对 pass 0 偏差 ≤ 6%）；
平均 n_llm_calls 相同（2.06–2.08）；把轨迹长度固定为 2 次调用后 pass 0 仍最快
（2.38s vs 2.41–2.65s）。所以"轨迹长度"和"缓存"都解释不了它。报告标为「未解释现象」，
`analyze.pass0_observation` 的判定规则写在代码里；不要在任何文档里给它编一个原因。

另一处不可测：运行级 `cache_hit == 0` 的未命中组 n=0（每次运行 ≥2 次调用，第二次必命中
前缀），按缓存命中拆延迟在这批数据上做不到；逐次调用的缓存命中自 09-12 起才记录。

### 0.5 措辞红线

- 不得写「静默换模型」「厂商未告知」之类对厂商行为的推断；只写「请求名已弃用、实际服务模型为
  deepseek-flash」。
- 不得写「裁判即被测模型」；只写「本次运行无法确认裁判与被测模型的独立性，开放题判定结果
  （n=10）应据此折价」（`analyze.JUDGE_INDEPENDENCE_UNCONFIRMED`，有测试钉住）。
- 不得写「模型变笨」或任何对 0.917→0.750 的归因；检索对照一节的结论句只填数字。
- 报告里的数字一律脚本生成；HANDOFF / LIMITATIONS 引用时以生成物为准。

### 0.6 尚未完成

- faithfulness 已在当前服务模型下重跑（09-12，`reports/eval_20260912T074455Z_full.json`：
  **0.898，n=33/33**；09-06 为 0.862，n=35/35——分母不同，按闸 B 口径 incomparable；
  `cap_033/034` 本次 0 次检索被 RAGAS 排除，同一行为使 tool_accuracy 0.944→1.000）。
  对照见脚本生成的 `reports/report_20260912T074455Z_full.md`。闸 B 里的 faithfulness 仍是 09-06 值。
- **待办：`ragas_runner._evaluate_rows` 需把逐行分数一并落盘**（当前只返回列均值，本轮未改）。
  含 RAGAS 段的全部快照都没有逐题分，09-06 vs 09-12 的 33 题交集口径因此「无法重算」
  （LIMITATIONS 第 16 条）。改完要重跑一次 RAGAS 才有逐题分，旧快照补不回来。
- 重跑结果已进入 latest.json / 闸 B（见 0.3）；`reports/report.md` 已由新 latest 重新渲染（由此**不再含**
  「自动↔人工一致率」与「裁判判据修订对照」两节——人工标注是对 09-06 答案做的，对新答案没有 kappa），
  `reports/traces_latest.jsonl` 已同步为 09-12 全量重跑轨迹，09-06 轨迹保留在 `reports/traces_20260906T092835Z.jsonl`
  （两者都 gitignore，仅本机）。README 与 CLAUDE.md §12 的简历 bullet 仍写着
  0.917 等 09-06 数字——2026-09-12 已改：每个数字带测量日期与复现状态（09-06 vs 09-12 重跑两列），
  「出处」列改为快照 JSON 字段。`render_markdown` 已改为并列打印请求名 / 响应名（按次计数，来自
  `meta.served`，没有就「未记录」），agreement 为空时打印「本快照不含此节」；report.md 已按新渲染重生成。
- `index/` 早于语料脱敏（见 §五），重建会移动 doc_id，需先全量重扫标注；未做。
- 09-11 的 50 条裁判分没有响应字段（`judge_runs.py` 现已记录，下次运行起生效）。
- 行尾：`stability/instrument.py` / `stability/records.py` 工作区为 LF（git autocrlf 提示），留待单独 commit。

### 0.7 下一步（建议顺序）

1. ~~由项目负责人决定是否接受本次工作区改动并提交~~ 已提交并推送（见 0.1 与 §一）。
2. 决定被测模型名的处理：`config.MODEL_NAME` / `JUDGE_MODEL_NAME` 改为端点实际列出的名字，
   还是保留旧名并在报告里持续标注响应名。改名会触发闸 A 之外的一切口径讨论，先问再动。
3. ~~若要让 0.917→0.750 进入门禁~~ 已做（见 0.3）：闸 B 拦下并存档，下跌登记为已接受的回归；
   闸 A 基线未动。`gate_accepted_regressions.json` 的接受人：姚尹杰（2026-09-12 确认）。
4. ~~RAGAS 在当前服务模型下重跑（`--ragas`），补 faithfulness。~~ 已做（09-12 全量重跑，见 0.6）。
   剩余：`_evaluate_rows` 落逐行分数后再跑一次，才有跨快照的交集口径。
5. 裁判独立性：换一个响应 model / 指纹**不同**的裁判端点或模型，重跑 10 道开放题裁判，
   再谈 kappa；否则简历里的 kappa 必须带 0.2 的限定。
6. 修 README / CLAUDE.md §12 的数字为当前生成物的值，并注明快照与服务模型名。
7. push 前：`python -m scripts.security_scan --history`（CLAUDE.md §13.4），把输出原文贴出
   （`corpus/` 已随此前提交公开，见 §一；历史里的本机路径按 LIMITATIONS 第 17 条不改写）。

---

## 一、状态：v1（M1–M6）已封版

| 里程碑 | 状态 |
|---|---|
| M1 骨架 / M2 被测 Agent / M3 黄金集+自定义指标 / M4 RAGAS | ✅ |
| M5 LLM-as-Judge + 人工盲标 + kappa | ✅ |
| M6 两道离线回归门禁 + README 定稿 | ✅ |

- `pytest -m "not live"` **309 项全绿**
- `python -m src.eval.report --gate` **PASS**（闸 A 口径闸 + 闸 B 回归闸）
- **已推送到 GitHub，本地与远程一致，无领先 commit**：`origin` = `github.com/2181385109/rag-agent-eval-harness`。
  2026-09-12 push 后 `git ls-remote` 确认 `origin/main` = `b23014a`（tag `v0.2-attribution`）；本句所在的
  docs 提交随后推送，push 后再次核对本地 `main` 与 `origin/main` 相同。两个 tag 已推到远程：
  `v0.1-pre-attribution` = `78b5122`（模型归属核验建立之前，指标无响应模型记录），
  `v0.2-attribution` = `b23014a`（模型归属核验建立、历史指标复现性调查完成、回归闸首次拦下真实落差）。
  `corpus/` 两份规格书已在 `origin/main` 里，即已公开。历史脱敏状态见 LIMITATIONS 第 17 条。
  （本条原写「尚未推到 GitHub」，2026-09-12 更正。）
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

**本机注意**：`traces_latest.jsonl` 自 2026-09-12 起是 09-12 全量重跑的轨迹，与 `eval_20260905T142507Z.json` 的 RAGAS 段
不再同源——上面 `--reuse-ragas` 那条示例现在会被指纹核验拦下（这是对的）；09-06 轨迹在 `traces_20260906T092835Z.jsonl`。
跑测试要用 `./.venv/Scripts/python.exe -m pytest`（不能用全局 `python`）；
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

---
