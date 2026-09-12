# LIMITATIONS — 稳定性 / 延迟测量的已知局限（PERF_SPEC 任务 B）

本文件记录 2026-09-11 那次测量（`stability/raw/run_20260911T162531Z.jsonl`，n=36×5）
的全部已知局限。数字本身在 [stability/report.md](stability/report.md)（脚本生成），
这里只写"这些数字**不能**说明什么"。本文提到的数字全部抄自那份报告，以报告为准；
报告重新生成后本文需要人工核对一遍。

## 测量设计上的局限

1. **temperature=0 不保证确定性，本次直接测到了。** 36 题里 8 题的工具调用序列在
   5 次运行间不一致（严格口径轨迹自洽率 0.778），2 题的任务成功判定在 5 次间翻转。
   这不是 bug、没有去修，它就是结论本身。PERF_SPEC 提议的「轨迹自洽率 ≥ 0.8」
   在这个 Agent + 这个模型上**未达标**；按事先定下的规则（首次全量实测值 − 0.05）
   把阈值定为 0.727 并冻结，未达标事实在此记录，不调低第二次。
2. **k=5 很小。** 每题只有 5 个样本，"5 次全一致"对一个 0.8 概率一致的题也有 33% 的
   机会出现。轨迹自洽率的分辨率是 1/36≈0.028，成功率跨次标准差只有 5 个点。
   这些数字能说明"有非确定性、大致多大"，不能支撑更细的比较。
3. **单日、单批次。** 180 次运行在 12 分钟内跑完，测的是那 12 分钟内 DeepSeek 服务端
   的行为。跨日、跨时段的漂移没有测；"服务端换了后端"这类变化只有再跑一次才看得到
   （见下一条）。
4. **请求名已弃用，实际服务模型为 `deepseek-flash`。** 请求的是 `deepseek-chat`，
   180 次运行的响应 `model` 字段全部为 `deepseek-flash`，`system_fingerprint` 只有一个值
   （`aeb56401ca74e127821c4f9126dcb669`）。DeepSeek 官方已于 2026-07-24 弃用
   `deepseek-chat` / `deepseek-reasoner` 两个模型名，现由旧名路由至 `deepseek-flash`（V4.1 Flash）；
   本仓库 `src/config.py` 仍硬编码旧名。DeepSeek 不暴露权重版本，能记的只有这两个字段；
   09-11 的记录只存每次运行首条响应，运行内后续调用是否同一后端未记录（09-12 起逐条记录）。
   2026-09-06 主评测快照（0.917）**未记录**响应模型名，只记了请求名 `deepseek-chat`。
   本次与该快照的逐题对照显示 6 题判定变化（`cap_015/016/017/020/027/036`）；
   两个 commit 间 Agent 代码、检索参数、黄金集、judge 判据逐字相同，索引未变；
   差异的归因（服务端后端变化 / 采样波动 / 其它）本次无法确定。
   2026-09-12 用当前代码与同一磁盘索引单次重跑主评测（`reports/eval_20260912_rerun.json`，
   由 `stability/rerun_main_eval.py` 生成，未覆盖 `latest.json`）：任务成功率 0.750（n=36/36），
   判定变化的正是同一组 6 题（`cap_015/016/017/020/027/036`，全部 True→False）；被测与裁判
   两条链路的响应 `model` 均为 `deepseek-flash`、同一 fingerprint；仅首次检索的 recall 不变（0.774），
   并集 recall 由 0.968 降到 0.871；retrieve 调用次数中位数由 1.0 变为 2.0（升，非降）。
   若以该重跑作 current 走闸 B：`task_success_rate` 跌 0.167 → dropped。
4a. **本次运行无法确认裁判与被测模型的独立性，开放题判定结果（n=10）应据此折价。**
   被测链路请求 `deepseek-chat`、裁判链路请求 `deepseek-reasoner`，设计上是两个模型。
   但 2026-09-12 的探针（`stability/raw/model_probe_20260912T025754Z.json`）显示两个请求名
   在同一时刻返回相同的响应 `model`（`deepseek-flash`）与相同的 `system_fingerprint`
   （`aeb56401ca74e127821c4f9126dcb669`）；端点 `/models` 只列出 `deepseek-flash` 与 `deepseek-v4-pro`。
   09-11 的 50 条裁判记录没有存响应字段，裁判链路当时落在哪个后端只有探针这一项间接证据。
   `deepseek-reasoner` 请求仍返回 `reasoning_tokens`，与 `deepseek-chat` 的行为是否等同，本次没有测。
4b. **裁判链路的服务端缓存命中未记录。** 客户端无缓存层（代码可查）；服务端 prompt cache
   是否命中，09-11 的裁判记录没存 usage，无法确认；`judge_runs.py` 自 09-12 起记录。
5. **同题重复调用命中服务端 KV 缓存，且"未命中组"为空。** 60% 的 prompt token 命中了
   DeepSeek 的缓存。报告按运行级 `cache_hit == 0 / > 0` 拆了两组，但 180 次运行的未命中组
   n=0：每次运行都有 ≥2 次 LLM 调用，第二次起必然命中第一次写入的系统提示词前缀，
   运行级的和不可能为 0。所以"按未命中组给主结论"在这批数据上做不到；逐次调用的缓存
   命中 09-12 起才记录，下一批数据才能按首次调用拆。另外 pass 0（无跨轮缓存）的端到端
   P50 反而低于后续四个 pass——报告标为**未解释现象**，本文不给解释。
   顺序已改成 pass-major（同题两次调用之间隔一整轮）以减轻、但消除不了缓存影响。

## 延迟数字的局限

6. **"LLM 调用延迟"是 HTTP 往返，不是模型推理时间。** 它包含公网 RTT、TLS、服务端排队、
   推理、流式传输。本机无法拆开。所以"LLM 占端到端 91%"说的是"等 API 的时间占 91%"。
7. **检索延迟是本机 CPU 上的 BGE-large 编码 + FAISS 精确搜索**（AMD Ryzen 7 7840H，无 GPU）。
   换硬件、换 embedding 模型、换 GPU 这个数就变。首次 encode 触发的模型加载（约 11 s）
   已在预热里排除，预热耗时记在 meta 里。
8. **openai SDK 的自动重试（max_retries=2）被计入 LLM 延迟。** 某次调用若被限流后重试，
   它的耗时会包含退避等待。本次 180 次运行 0 报错，但无法区分"一次成功"和"重试后成功"。
9. **分位数样本量 n=180，P99 只由 2 个点决定**，P99 数字的置信度很低，报出来是为了不只报均值。

## 判定与相似度的局限

10. **开放题的逐次判定靠 deepseek-reasoner 裁判，裁判本身也是 LLM。** 判定翻转的
    `cap_017` 是开放题，无法区分是 Agent 答案变了还是裁判打分抖了（两者都记录了：
    答案指纹 + 裁判理由）。裁判用的是 v2 要点级判据，与主评测同源；人工没有对这 50 个
    答案重新标注，所以这里没有 kappa。
11. **答案相似度用 BGE 余弦，它对措辞长度敏感。** `cap_006` 五次都答对（0.15），
    只是一次 23 字、一次 95 字，相似度就掉到 0.453。这个指标衡量的是"表述稳不稳"，
    不是"对错稳不稳"——后者看判定自洽率。
12. **判定自洽率的分母含 10 道开放题，但它们的"成功"门槛是裁判 ≥2 分。** 裁判给 1 分
    的题（如 `cap_007/035`）五次都判失败，于是"自洽"——自洽不等于答对。

## 产物与流程的局限

13. **FAISS 索引（2026-09-04 建）早于语料脱敏（2026-09-06）。** 索引里的 chunk 仍含本机
    路径 `D:\xiangmu\credit-risk-mlops`，模型在 `cap_035` 的答案里逐字引用了它（全量运行 3 次、试点 2 次）
    （与 HANDOFF §三点五刻意保留的 5 处是同一字符串、同一性质）。没有重建索引：
    重建会移动 chunk 边界与 doc_id，黄金集的 `expected_doc_ids` 需要按
    `data/annotation_criteria.md` 全量重扫才能跟上——那是独立的一次口径变更，不混在本次里。
14. **答案相似度不在离线可复现闸的范围内**（要加载 1.3 GB 的 BGE），闸 C 只核对它记录了
    所用 embedding 模型名。其余所有数字都由 CI 从 `stability/raw/` 重算并逐位比对。
15. **本机无 `make`。** `Makefile` 按规格提供；Windows 上等价命令是 `python -m stability`。
16. **产物记录粒度细于报告粒度这一要求未被满足；「响应 model 未记录」与「逐题 RAGAS 分未存」
    是同一根因。** 09-06 主评测快照及其轨迹只记了请求模型名（没有响应 `model` / `system_fingerprint`），
    09-11 的 50 条裁判记录没有响应字段；含 RAGAS 段的全部快照（09-05 源快照、09-06、09-12 全量重跑）
    都只存了均值、`scored_counts` 与 `evaluated_ids`，没有逐题分数（`ragas_runner._evaluate_rows`
    只返回列均值）。报告按题、按链路下结论，产物却只到运行级 / 均值级，于是两类事后分析都做不到：
    - 归因：不知道 09-06 那次落在哪个后端，0.917→0.750 的差异不能归因（见第 4 条）；
    - 重新切分：09-06 与 09-12 的 RAGAS 分母不同（35 vs 33，`cap_033/034` 本次无检索内容被排除），
      要在 33 题交集上并列对比需要逐题分，两侧都没有，只能记「无法重算」
      （`reports/report_20260912T074455Z_full.md`，脚本生成；不估算、不用全量均值近似）。
    响应字段自 09-12 起逐条记录（被测 / 裁判 / RAGAS 三条链路，`stability/rerun_main_eval.py`）；
    逐题 RAGAS 分数本轮**未改**，待办见 HANDOFF §0.6。已有快照的缺口补不回来。
17. **git 历史里含本机路径；脱敏自 `c95c4b5` 起，此前提交保留原样。** 截至 2026-09-12，
    `origin/main`（`78b5122`）共 24 个已推送提交，其中 21 个的文件内容含本机路径 `D:\xiangmu`
    （脱敏前的语料、派生文档与快照，自 `f149a6c` 起）；`c95c4b5` 在源头（`corpus/`）与派生文档
    做了脱敏，之后只剩 HANDOFF §三点五刻意保留的 5 处与后续产物里的同一字符串。
    `python -m scripts.security_scan --history` 对全部 28 个提交扫描：API key 模式 0 命中，
    `.env` 从未进入任何提交。
    处置决定：**不改写历史**。理由：保全提交时间线的可追溯性（快照的 `git_commit` 字段、闸 A 基线、
    判据修订记录都指向具体 commit，改写会让这些引用失效）；该路径只含盘符与目录名，不含凭据
    与个人身份信息。
18. **闸 B 的保护范围取决于 `latest.json` 何时更新，而这由人工决定；未纳入 latest 的运行不受回归闸保护。**
    09-12 的对照重跑（`eval_20260912_rerun.json`，任务成功率 0.750）与全量重跑（`eval_20260912T074455Z_full.json`）
    都是另存快照，在被复制为规范名 `eval_20260912T081718Z.json` 并写入 `latest.json` 之前，`--gate` 一直是 PASS。
    闸 B 第一次真正拦下 0.917→0.750 的原文存档在 `reports/gate_fail_20260912T081718Z.txt`（不得删除或覆盖）；
    该下跌随后以 `reports/gate_accepted_regressions.json` 登记为已接受的回归（日期、原因、接受人，只对这一对快照
    生效），测试改为「相对已接受基线无新增 dropped」。门禁只看它被喂进的那份 latest；这一路上的三次同类失效
    与设计原则见第 21 条。
19. **闸 B 不检查 RAGAS 段的分母，该类指标的回归保护当前失效。** `report._metric_n` 只读指标块自己的 `n`，
    RAGAS 段没有这个字段而返回 None，于是 faithfulness 的 `scored_counts` 35→33（`cap_033/034` 本次无检索内容
    被排除）没有触发「分母变动即 incomparable」，被以 `ok +0.0361` 放过（见 `reports/gate_fail_20260912T081718Z.txt`）；
    `stability/rerun_report.py` 标了「分母不同」，闸 B 标不了。同理，faithfulness 若因分母缩小而**上升**，闸 B 也看不见。
    待办：`_metric_n` 需支持 RAGAS 段（取 `scored_counts[metric]`）；本轮未改。
20. **`latest.json` / `eval_20260912T081718Z.json` 的 `meta.notes` 自述「另存快照，不写 latest.json」，
    `meta.rerun_of` 指向 `reports/latest.json`；成为 latest 之后这两句失真。** 它们是 `stability/rerun_main_eval.py`
    写入原件 `eval_20260912T074455Z_full.json` 时的自述。为保持复制件与原件逐字节一致（sha256
    `ac4fe0cd574a0d83a3430705761e40a9076c012874776ad656acfc601942e563`，三份相同），不修改产物内容；以本条为准。
21. **设计原则：基线必须由不可变标识指定（快照文件名 / commit / sha256），禁止任何 latest 语义的可变指针。**
    本轮同类失效出现三次，共同根因都是拿可变指针当基线，失效时静默返回「无差异」而不是报错：
    - 实例一：`find_previous_snapshot` 原用 `glob("eval_*.json")` 选基线，另存的对照快照 `eval_20260912_rerun.json`
      也在候选里，闸 B 会拿另一份重跑当基线。修法：只认 `SNAPSHOT_NAME_RE.fullmatch`（`eval_YYYYMMDDTHHMMSSZ.json`），
      测试 `test_find_previous_snapshot_ignores_non_pipeline_snapshot_names`。
    - 实例二：快照复制件若按启动时刻命名为 `eval_20260912T074455Z.json`（而不是 `meta.timestamp_utc` 对应的
      `081718Z`），闸 B 排除不到它自己，会把它选成基线——latest 与自己比，三项全 ok，−0.167 的下跌被吞掉。
      落盘前用临时目录模拟发现。修法：复制件按 `snapshot_filename(meta.timestamp_utc)` 命名；`run_gates` 新增守卫，
      基线与 latest 字节相同即 ERROR 不放行，测试 `test_gate_b_refuses_a_baseline_whose_content_equals_latest`。
    - 实例三：`stability/rerun_report.py` 原默认把 `reports/traces_latest.jsonl` 当基线轨迹；同步 latest 后它变成
      09-12 轨迹，重生成的对照报告把本次检索次数冒充成基线（cap_015 `3→1` 变成 `1→1`，中位数 `1.0→2.0`
      变成 `2.0→2.0`），所有数字仍「合理」、无任何报错，靠重生成前后 sha256 不同才发现——**逐字节可复现
      是检出这类静默失效的手段**。修法：默认取与基线快照同名的 `traces_{name}.jsonl`，并逐题核对答案与
      retrieved_doc_ids，对不上按「未记录」处理并在报告写明；09-06 轨迹保留为 `traces_20260906T092835Z.jsonl`。
    正面例子：`--reuse-ragas` 复用前逐题核对轨迹指纹（`report.verify_ragas_reuse`）；裁判分带 `answer_sha1`。
    尚存的同类用法（待办）：`stability/analyze.py` 的快照对照与检索对照两段读 `reports/latest.json` +
    `reports/traces_latest.jsonl`，两者靠同时写出配对、未核对内容；HANDOFF §四 `--from-traces reports/traces_latest.jsonl`
    的示例。应改为显式传快照名并核对。
22. **Cohen's kappa 0.623 不可复现，且不可直接重算。** `data/human_labels.jsonl` 的 10 条人工标注是对 2026-09-06
    那批答案（`eval_20260906T092835Z.json`）做的；latest 自 2026-09-12 起是另一批答案（10 道开放题里
    `cap_017/020/036` 的裁判分都变了），人工标注与当前答案不再配对——kappa 既不能用当前快照复现，也不能拿旧标注
    对新答案直接重算。当前 `reports/report.md` 的「自动↔人工一致率」与「裁判判据修订对照」两节因此消失；
    `render_markdown` 在 agreement 为空时原本直接跳过，属静默失效、不会报错，2026-09-12 起改为打印「本快照不含此节」。
    待办：重新人工标注（对当前答案盲标），或让标注版本与答案快照绑定（标注行带 `answer_sha1`，同裁判分的做法，
    回填时逐条核对、对不上即作废）。
