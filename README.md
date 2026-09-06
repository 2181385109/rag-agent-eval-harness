# rag-agent-eval-harness

一个 **RAG + 工具调用 Agent** 的端到端评测流水线。
评测流水线是主角，Agent 是被测对象。模型能力由 **DeepSeek API** 提供，
语料 / 检索 / 指标 / 测试全部本机运行。

完整的范围约束、技术选型与验收口径见 [CLAUDE.md](CLAUDE.md)。

---

## 进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 目录骨架、依赖锁版本、config、`.env` 载入、DeepSeek 客户端封装、hello-world LangGraph 图 | ✅ |
| M2 | RAG（切分 → FAISS → 检索）+ 两个工具 + ReAct 图 + 轨迹记录 | ✅ |
| M3 | 36 条自建黄金集；recall@k / 任务成功率 / 工具调用准确率 / 多轮一致性 | ✅ |
| M4 | RAGAS：faithfulness / answer_relevancy / context_precision / context_recall | ✅ |
| M5 | LLM-as-Judge（`deepseek-reasoner`）+ 10 题人工盲标，Cohen's kappa | ✅ |
| M6 | GitHub Actions 两道回归门禁 + README 定稿 | ✅ |

---

## 指标

> **📌 待填。** 本节的数字等门禁跑通、全量重跑一次后再一次性填入真值。
> 在那之前请直接看 **[reports/report.md](reports/report.md)**——那是自动生成的，
> 每个数字都带分母、带 git commit、带模型与检索参数。

| 指标 | 值 | 出处 |
|---|---|---|
| 检索召回率 recall@k | _（见报告）_ | `reports/report.md` · 指标 |
| 任务成功率 | _（见报告）_ | 同上 |
| 工具调用准确率 | _（见报告）_ | 同上 |
| 多轮一致性（temp=0 / temp>0） | _（见报告）_ | `reports/report.md` · 多轮一致性 |
| faithfulness / answer_relevancy / context_precision / context_recall | _（见报告）_ | `reports/report.md` · RAGAS |
| 自动↔人工一致率 kappa | _（见报告）_ | `reports/report.md` · 自动↔人工一致率 |

**第一红线是「禁止编造任何指标」**：README 与简历里出现的每个数字，
都必须能由仓库里的一条命令重新跑出来。填不出来就说明那块还没做完，不许提前填。

复现全量评测（需要 `DEEPSEEK_API_KEY`，会产生费用）：

```bash
python -m src.eval.report --ragas          # 36 题 + 一致性 + RAGAS，写 reports/
python -m src.eval.judge --score           # 自动裁判给 10 道开放题打分
python -m src.eval.judge --kappa           # 离线算一致率，不花钱
```

报告里记录了模型名、裁判模型、embedding 模型、top-k、切分参数与 git commit——
缺任何一样，这个数字就不可复现。

---

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS

cp .env.example .env      # 然后填入你自己的 DEEPSEEK_API_KEY
```

跑测试（**不需要 API key，不发任何网络请求**）：

```bash
pytest -m "not live"
```

真实打通 DeepSeek 的冒烟测试（需要 key，会产生少量费用）：

```bash
pytest -m live
```

问一句话，走完整的 LangGraph 状态图：

```bash
python -m src.agent.rag --build                       # 建 FAISS 索引（首次会下载 BGE 模型）
python -m src.agent.rag --list                        # 列出 doc_id -> 出处，标注黄金集时用
python -m src.agent.graph --trace "PSI 告警阈值定在多少？"   # 跑 Agent 并打印完整轨迹

python -m src.eval.report --limit 5 --no-consistency   # 小子集试跑（省钱）
python -m src.eval.report --ragas                      # 全量评测 + RAGAS，出 reports/

# 轨迹会落盘到 reports/traces_latest.jsonl，可不重跑 Agent 只补算指标：
python -m src.eval.report --from-traces reports/traces_latest.jsonl --ragas
```

---

## 回归门禁（M6）

思路直接搬自信贷风控项目里的 **PSI 触发告警**：定一条基线，指标漂出容差就报警。
这里落成 CI 的两道闸，**都不调真实 API**（GitHub Actions 上不放 key）：

```bash
python -m src.eval.report --gate     # 本地跑两道闸，等价于 CI 里那一步
```

| | 闸 A · 口径闸 | 闸 B · 回归闸 |
|---|---|---|
| 输入 | 冻结：`tests/fixtures/` 里的 10 题子集 + 当时的轨迹 + 当时的开放题裁判分 | 已提交进 git 的报告 JSON |
| 基线 | `reports/gate_baseline.json` | `reports/` 里上一份 `eval_*.json` 快照 |
| 判据 | **逐位相等** | 关键指标跌超 `METRIC_DROP_TOLERANCE`（0.05） |
| 盯的指标 | recall@k（含仅首次检索档）、任务成功率、工具准确率（严格/宽松） | `task_success_rate` / `tool_accuracy` / `faithfulness` |
| 抓什么 | 指标**口径**被改动 | 模型或流水线**变差** |

几处刻意的设计：

- **闸 A 用等值而不是容差，而且涨了也拦。** 输入是冻结的，数字没有理由变。
  变高同样可疑——多半是判定被放宽了。要改口径就显式跑
  `--write-gate-baseline`，等于在提交历史里公开宣布"我改了定义"。
- **门禁的输入必须含失败样本。** 冻结子集里特意放了检索彻底 miss 的 `cap_035`、
  工具序列判错的 `cap_033/034`、两道诱饵题，以及任务失败的 `cap_007`（裁判判 1 分）。
  全是满分的子集，指标算错了也照样绿。有几条测试盯着这点。
- **裁判分也是冻结输入。** 任务成功率的开放题那一半由裁判分判定，
  让门禁去读会随手一跑就变的 `reports/judge_scores.jsonl`，测的就不再是"口径变没变"。
  冻结的那份还带答案指纹：轨迹被改而分没跟着改，该条分数直接作废而不是将错就错。
- **不可比就说不可比。** 三种情形下闸 B 标 `incomparable` 而不是硬算一个差值：
  黄金集规模不同；某个指标**自己的分母**变了（任务成功率从"只算 26 道闭合题"
  扩到"36 道全集"）；**裁判判据换了版本**（v1 结论级 → v2 要点级那次，
  同一批轨迹、同一个模型，两题掉档让成功率跌 0.056）。
  后两种题数都没变，只比总题数会把纯口径变动当成一次普通波动放过去。
  代价是这三条也能给真实的变差打掩护——对策不在闸 B 里：口径变动必然要动
  fixture 基线（闸 A 逐位比对）、`RUBRIC_VERSION` 或 config 阈值，全都会出现在 diff 里。
- **fixture 不复制 corpus 原文**：门禁只用 `doc_id` 判 recall，把 chunk 全文
  塞进 git 既无必要，也等于把语料再落一份。

---

## 评测设计要点

### 黄金集防污染

36 条题目全部**自建**或基于 `corpus/` 里的私有文档改写，**不复用 MMLU / C-Eval 等公开
benchmark 原题**。理由很直接：公开题目大概率已经进了模型的训练数据，拿它评测量到的是
背诵而不是能力。

这和信贷风控项目里用 **GroupKFold 防数据泄漏**是同一个直觉——都是在阻断
"被评估者提前看过答案"的通路，只是一个发生在训练/验证切分上，一个发生在
预训练语料与评测集之间。

配套纪律：

- 每题标注 `expected_tool`（有序列表）与 `expected_doc_ids`（**证据组**：组间并列、
  组内替代），让"工具选对没""检索准没"可判定。
- 闭合题用 `answer_keys` 规则判定，能不用 LLM 判就不用——省钱且更可复现；
  开放题才交裁判层。
- 标注只依据通读语料得出的书面判据（`data/annotation_criteria.md`），
  **不许看着检索结果反推**。确需修订就对全部题目用同一判据重扫一遍，
  并在报告里写明修订前后的差值与归因（见 `reports/annotation_rescan_2026-09-05.md`）。

### LLM-as-Judge 与人机一致率

- **裁判与被测刻意不同源**：被测 `deepseek-chat`，裁判 `deepseek-reasoner`，
  避免"自己判自己"的自评偏好。有一条测试钉住"裁判 ≠ 被测"，
  另一条钉住"打分时用的确实是配置里那个裁判模型"。
- **人工先盲标**：标注表生成函数显式丢弃任何传入的裁判分数
  （`judge.py` 里那句 `del judge_scores`）。人先看到机器分再标，
  kappa 度量的就不是人机独立看法是否一致，而是确认偏误。
  反向也锁死：裁判打分路径不读 `human_labels.jsonl`。
- **判据同源**：人看的标注表和裁判的提示词，三档判据是同一段字（都从 `LABEL_SCALE` 拼），
  各写一份必然漂移。
- **kappa 报两种口径**：unweighted 与 quadratic weighted（0/1/2 是有序标度），
  都带 n 与 bootstrap 置信区间。**退化情形（某一方标注无变异）打印「未定义」，
  绝不落成 0.0**——0.0 读起来像"完全不一致"，那是编造。
- **已知的信息不对等**：裁判按设计盲看，只拿问题 + 参考答案 + Agent 答案；
  人工标注时手里有检索片段全文、还回查过语料。这条限制连同逐题分歧归因
  一起写在报告的一致率一节里（源文件 `data/agreement_interpretation.md`），
  不藏着。

### 工程纪律

- **密钥安全**：API key 只从环境变量 `DEEPSEEK_API_KEY` 读，`.env` 已被 gitignore，
  并且有一条常驻测试 (`tests/test_secrets.py`) 扫描全仓库，一旦有人把 key 粘进代码就直接红。
- **不外发数据**：除对 DeepSeek 的模型调用外没有任何外部请求；
  `langsmith`（langchain-core 的传递依赖）的追踪上报在 `src/config.py` 里被显式关闭。
- **CI 不调真实 API**：GitHub Actions 上不放 key，LLM 调用全部打桩或读冻结轨迹，
  验证的是流水线和指标计算逻辑本身；真实全量评测在本机手动跑。
- **可复现优先**：默认温度固定为 0；多轮一致性另跑一档 `temperature>0` 的鲁棒性专测，
  两档并列报数、互不替代（temp=0 下一致比例天然接近 1.0，信息量有限）。
- **分母诚实**：指标无定义时返回 `None` 而不是当 0（纯算术题没有应检索文档）；
  每个指标都带 `n/total`，报告正文必须写出分母。
  RAGAS 单个 job 失败会留 NaN 而 `mean()` 默认跳过它——所以报告强制逐指标上报
  "实际打分行数"，行数不同的差值直接标成**不可比**。
- **失败如实入账**：Agent 选错工具、跑满步数没收口、裁判调用超时，全部记录并单列，
  不在 Agent 侧偷偷纠正——这些失败正是评测集要抓的。
- **复用留痕**：`--reuse-ragas` 可以原样取回上一次的 RAGAS 分数（同一批轨迹下省 40 分钟），
  但报告里会打「本节为复用，不是本次重算」的标记，并指向**当初算出它的那份快照**
  （而不是会被覆盖的 `latest.json`）。

---

## 环境说明

Python **3.12**（CLAUDE.md 原文锁 3.11，本机无 3.11，已确认后调整；其余选型不变）。
依赖锁版本在 `requirements.txt`。
