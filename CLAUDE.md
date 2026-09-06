# CLAUDE.md — RAG Agent 评测流水线（rag-agent-eval-harness · DeepSeek API 版）

> 本文件是这个仓库的最高约束。你（Claude Code）在本仓库内的所有产出，都必须服从这里定义的范围、技术栈、验收口径与工作纪律。凡与本文件冲突的实现，一律以本文件为准；需要偏离时，先停下来问，不要自作主张。

---

## 0. 一句话目标

搭一个**被测系统（一个 RAG + 工具调用的 Agent）** 和一套**给它打分的评测流水线**。评测流水线是主角，Agent 是被测对象。模型能力由 **DeepSeek API** 提供，其余（语料、检索、指标、测试）全部在本机运行。最终必须产出一条可复现、指标可回溯的简历 bullet（见第 12 节）。

命中的岗位方向：大模型测试开发 / AI-Agent 评测 / 大模型数据策略 / Agent 应用开发（安全评测方向留到 v2）。

---

## 1. 硬约束 / 非目标（最重要，先读这条）

1. **禁止编造任何指标。** 简历和 README 里的每一个数字，都必须能由仓库里的一条命令重新跑出来。做不到复现的数字，不许写。第一红线。
2. **v1 只做主线**（见第 9 节里程碑）。安全红队、FastAPI/Streamlit demo、MLflow/LangSmith、rerank 全属 v2，v1 里**不要**实现，最多留 `# TODO(v2)`。
3. **先写验收口径，再写实现。** 每个模块动手前，先把"什么算对"落成 pytest 断言或明确的指标定义（第 6、7 节），再写代码去撞它。不许反过来。
4. **评测集不许被污染。** 黄金集题目不许直接抄公开 benchmark；纪律见第 8 节。这是项目的核心工程卖点。
5. **小步提交，每步可跑。** 每个里程碑结束时仓库必须"能跑、测试绿"。
6. **不引入范围外的重型依赖。** 需要新库先在第 3 节确认，不在清单里的先问。
7. **密钥安全，只出必要的模型调用。** 除了对 DeepSeek 的模型调用，不引入任何把语料 / 结果发往其他外部服务的逻辑。API key 只从环境变量读，**绝不写进代码、绝不提交进 git**。

---

## 2. 技术栈（已锁定，不要替换）

| 层 | 选型 | 说明 |
|---|---|---|
| Agent 编排 | **LangGraph** + LangChain | 用 LangGraph 显式状态图实现 ReAct 循环；LangChain 提供工具、splitter、检索器 |
| LLM（被测 + 裁判） | **DeepSeek API**，模型 `deepseek-chat`，端点 `https://api.deepseek.com/v1`（OpenAI 兼容） | key 从环境变量 `DEEPSEEK_API_KEY` 读。模型名以 DeepSeek 官方文档当前可用值为准；想更省钱可换更便宜的 flash 档。裁判可选用不同模型，见下注 |
| Embedding | **BGE**（`bge-large-zh-v1.5`，sentence-transformers 本地加载） | 中文检索标配，本地免费，不走 API |
| 向量库 | **FAISS**（本地） | 最简、免费；预留接口便于日后换 Chroma 做选型对照 |
| 检索 | LangChain text splitter + 相似度检索 | rerank 属 v2 |
| RAG 评测 | **RAGAS** | faithfulness / answer_relevancy / context_precision / context_recall（RAGAS 的裁判模型同样指向 DeepSeek） |
| 一致率 | scikit-learn `cohen_kappa_score` | 自动裁判 ↔ 人工标注一致率 |
| 门禁 | **pytest** + **GitHub Actions** | 见第 7 节 |
| 数据处理 | pandas / numpy | 复用你已有习惯 |
| 校验 | pydantic v2 | 用例与输出的 schema 校验 |

> **裁判模型注（加分设计）**：v1 裁判可先用 `deepseek-chat`。若想更严谨，裁判换用不同模型（如 `deepseek-reasoner`），避免"自己判自己"的自评偏好——面试里能讲成一个有意识的评测设计。

Python **3.12**（原定 3.11；本机只装了 3.12.3，且 langgraph / langchain / openai / faiss-cpu / sentence-transformers / ragas 均有 3.12 轮子，经确认后调整。依赖装在项目内 `.venv`，不污染全局环境）。依赖用 `requirements.txt` 锁版本。

---

## 3. 目录结构（按此创建，不要另起一套）

```
rag-agent-eval-harness/
├── CLAUDE.md
├── README.md                  # 面向招聘方，最后写；指标必须可复现
├── requirements.txt
├── .gitignore                 # 必须包含 .env
├── .env.example               # 只放 KEY 名（DEEPSEEK_API_KEY=），不放值
├── .github/workflows/ci.yml
├── corpus/                    # 知识库原文（你自己塞 ML/风控概念文档）
├── src/
│   ├── agent/
│   │   ├── graph.py           # LangGraph 状态图 / ReAct 循环
│   │   ├── tools.py           # retrieve 工具 + 1 个动作工具
│   │   ├── rag.py             # 切分 / 向量化 / 检索
│   │   └── llm.py             # DeepSeek 客户端封装（OpenAI 兼容）
│   ├── eval/
│   │   ├── datasets.py        # 加载黄金集，pydantic schema
│   │   ├── metrics.py         # 任务成功率 / 工具调用准确率 / 多轮一致性
│   │   ├── ragas_runner.py    # RAGAS 指标
│   │   ├── judge.py           # LLM-as-Judge + kappa
│   │   └── report.py          # 汇总所有指标 → JSON/CSV/markdown 报告
│   └── config.py              # 模型名、base_url、阈值、路径集中管理
├── data/
│   ├── golden_capability.jsonl
│   └── human_labels.jsonl
├── reports/                   # 每次评测的指标快照（进 git，作回归基线）
└── tests/
```

`src/config.py` 集中放：`BASE_URL="https://api.deepseek.com/v1"`、`MODEL_NAME="deepseek-chat"`、`JUDGE_MODEL_NAME="deepseek-chat"`、各指标阈值。key 不放这里，从 `os.environ["DEEPSEEK_API_KEY"]` 读（配合 python-dotenv 从 `.env` 载入）。

---

## 4. 前置环境准备（M1 之前，只做一次）

1. 在 DeepSeek 开放平台注册、充值、创建 API key（这步由人来做，不是 Claude Code 做）。
2. 项目根目录建 `.env`（照抄 `.env.example`），填入 `DEEPSEEK_API_KEY=sk-...`。
3. **确认 `.gitignore` 里有 `.env`**，避免把 key 提交上去。用 python-dotenv 在程序启动时加载。
4. `src/agent/llm.py` 用 OpenAI SDK：`base_url` 指向 DeepSeek 端点，`api_key=os.environ["DEEPSEEK_API_KEY"]`。
5. `corpus/` 里先放两三篇你自己的 ML/风控概念文档起步，否则 M2 的 RAG 没东西可检索。

---

## 5. 被测系统（Agent）规格

- 一个 **ReAct Agent**，用 LangGraph 实现状态图（规划 → 选工具 → 调工具 → 观察 → 回答/继续）。
- 挂**两个工具**（必须两个，单工具无法评测"工具调用准确率"）：
  1. `retrieve(query)`：走 RAG，检索 `corpus/`。
  2. 一个**动作工具**，如 `calc(expr)` 或 `lookup(key)`，制造"该用哪个工具"的判定点。
- Agent 记录每步**中间轨迹**（调了哪个工具、检索到哪些 chunk、最终答案），评测流水线据此打分。轨迹用 pydantic 固化 schema。
- 模型出错（选错工具、漏参数、幻觉）时**如实记录、如实计入指标**，不要在 Agent 侧偷偷纠正——这些失败正是评测集要抓的。

---

## 6. 评测集与指标（验收口径核心）

每个指标必须在代码里有**明确定义 + 单元测试**。定义如下：

| 指标 | 定义 | 归属 |
|---|---|---|
| **检索召回率 recall@k** | 标注的"应检索文档"是否在 top-k 结果中，按题平均 | RAG 检索 |
| **答案忠实度 faithfulness** | 答案论断是否被检索内容支撑（RAGAS） | RAG 生成 |
| **答案相关性 answer_relevancy** | 答案是否切题（RAGAS） | RAG 生成 |
| **任务成功率** | 最终答案是否命中标准答案（闭合题规则判定，开放题走裁判） | Agent 端到端 |
| **工具调用准确率** | 实际工具序列是否匹配标注的"应调工具" | Agent 工具使用 |
| **多轮一致性** | 同题跑 K 次（默认 5），判定结果的方差 / 一致比例 | 稳定性 |
| **自动↔人工一致率 (kappa)** | 裁判打分 vs 人工标注，Cohen's kappa | 评测可信度（**最值钱**） |

**黄金集样本 schema（pydantic）：**
```json
{
  "id": "cap_001",
  "question": "...",
  "reference_answer": "...",
  "answer_keys": ["0.25", "显著漂移"],
  "expected_tool": ["retrieve", "calc"],
  "expected_doc_ids": ["c730fa5c_0027"],
  "answer_type": "closed | open",
  "failure_tag": null
}
```

`expected_doc_ids` 的两种写法（可混用）：

```json
"expected_doc_ids": ["c730fa5c_0027"]
"expected_doc_ids": [["c730fa5c_0008", "c730fa5c_0019"], ["2da65ad3_0026"]]
```

前者是单块证据；后者是**两个证据组**，组内任一命中即算该组已召回。
裸字符串会被归一成单元素组，所以单块标注不必写成嵌套。

> **schema 修订记录（2026-09-05，M3）**
>
> 1. `expected_tool` 由**单个字符串**改为**有序列表**。原因：像"PSI 显著阈值减去
>    轻微下限差多少"这类题天然需要先 `retrieve` 再 `calc`，单字符串表达不了顺序，
>    也就没法判定"工具选对没"。判定口径：把实际工具序列的**连续重复折叠**后
>    与本列表**逐项严格比对**（模型连调两次 retrieve 视同一次；多调一个工具算错）。
> 2. 新增 `answer_keys`：闭合题规则判定所需的**必现关键片段**。§8.3 要求"闭合题
>    规则判定，能不用 LLM 判就不用"，但不标出"什么算答对"就无从规则判定。
>    判定口径：`answer_keys` 全部出现在最终答案里才算成功（比对前统一全角转半角、
>    去空格、转小写）。单个 key 内用 `|` 分隔可接受的写法变体，命中其一即可，
>    例如 `"0.8|80%"`、`"分层|stratify"`——这是为了让规则判定不被措辞差异误伤，
>    而不是放宽正确性标准。开放题 `answer_keys` 留空，交 M5 的裁判层。
> 3. `expected_doc_ids` 允许为空（纯算术题、幻觉诱饵题）。**recall@k 只在非空的
>    题上计算**，报告里必须写明分母（如 `recall@4 = 0.75 (n=8/10)`），
>    不许拿全集当分母稀释或抬高。

> **schema 修订记录（2026-09-05，M5 前置）**
>
> 4. `expected_doc_ids` 由**扁平 doc_id 列表**改为**证据组列表**。
>    口径：外层组之间是并列关系（都要覆盖到），组内是替代关系（任一命中即可）；
>    **recall 的分母是组数**，不是文档数。
>
>    原因：cap_007 问"两个项目各自怎么防数据泄露"，信贷侧有三块、航班侧有两块
>    都能独立作答。按扁平口径要求五块全中，Agent 答得完美也只能拿 0.4——
>    那衡量的不是召回质量，是运气。实测同一批轨迹下 recall@k 由 0.932 升到 0.935
>    （cap_007 单题 0.400 → 0.500），差值纯粹来自建模口径，不含模型波动。
>
> 5. 裁判模型由 `deepseek-chat` 改为 **`deepseek-reasoner`**，与被测模型
>    刻意不同源，避免"自己判自己"的自评偏好。注意该值同时决定 RAGAS 的裁判，
>    换模型后 faithfulness 等指标会整体变动，必须重跑才能与配置对上；
>    报告 meta 里记录了每次实际使用的裁判模型。
>
> **指标口径修订记录（2026-09-06，M5 回填）**
>
> 6. **任务成功率的开放题回填。** M3~M5 期间 `task_success_rate` 只用 `answer_keys`
>    判定闭合题，开放题虽有裁判分却没并进去——报出来的 1.000 实际只对 26 道闭合题成立。
>    现按本节表格原本的定义（"闭合题规则判定，开放题走裁判"）补全：
>
>    - **闭合题**：`answer_keys` 全部命中（不变）。
>    - **开放题**：裁判分 **≥ 2** 才算成功。三级标度里 1 分是"方向对、要点有遗漏"，
>      **不计成功**——把半对算成过关，这个指标衡量的就变成"没答错"而不是"答对了"。
>      门槛集中在 `config.OPEN_SUCCESS_THRESHOLD`，改它必然触发闸 A。
>    - 开放题**没有裁判分**时仍记 None（计入 total 不计入 n），不许当失败。
>
>    同一批轨迹下，分母由 n=26/36 变为 **n=36/36**，值由 1.000 变为 **0.972**
>    （35/36，唯一失败 cap_007：裁判判 1 分，信贷侧缺 70/15/15、航班侧漏了时间序切分
>    与 tail_id GroupKFold）。**这 -0.028 全部来自口径，不含任何模型变化**——
>    轨迹是同一份 `reports/traces_latest.jsonl`。
>
> 7. 配套的两道防护（都是这次改口径直接暴露出来的洞）：
>    - 裁判打分行新增 `answer_sha1`（被打分答案的指纹）。裁判分是给**某一批答案**打的，
>      重跑 Agent 后旧分套新轨迹不会报错，只会静默产出一个错的成功率。回填时逐条核对，
>      对不上就作废并在报告里列出。老的打分行没有指纹，标为"未核验"如实呈现。
>    - 闸 B 新增**分母变动即 incomparable**。题数没变、指标名没变，但 n 从 26 变到 36 时，
>      差值完全来自口径；只看 `golden_set_size` 会把它当成一次普通波动放过去。
>    - 冻结裁判分 `tests/fixtures/gate_judge_scores.jsonl` 进入闸 A 的输入
>      （闸 A 基线：task_success_rate 由 1.000/n=5 变为 **0.900/n=10**）。
>
> **判据修订记录（2026-09-06，裁判判据 v1 结论级 -> v2 要点级）**
>
> 8. **裁判判据由结论级收紧为要点级。** v1 的 1/2 档只说"要点有遗漏"/"要点齐全"，
>    没定义什么算一个要点，人机各按各的理解办（裁判比对结论层、人工比对要点层），
>    这是 kappa=0.216 的主因。v2 把「要点」定死成**参考答案里的每一项具体限定**
>    （数字与阈值、比例、公式或指标定义、方法与算法名、明确的禁止或例外，各算一项），
>    写明**缺一项即降到 1 分**，并在提示词里加了"先列清单再逐项核对"的三步流程。
>    判据正文与打分细则人机同源（`judge.LABEL_SCALE` / `judge.RUBRIC_NOTES`，测试钉住）。
>
>    同一批轨迹、同一份人工标注、同一个被测模型，**唯一变量是判据**：
>
>    | 量 | 改前(v1) | 改后(v2) | 差 |
>    |---|---|---|---|
>    | 任务成功率 | 0.972 | **0.917** | -0.056（n=36/36） |
>    | 完全一致率 | 0.600 | **0.800** | +0.200（n=10） |
>    | kappa（unweighted） | 0.216 | **0.623** | +0.407（n=10） |
>    | kappa（quadratic） | 0.103 | 0.324 | +0.222（n=10） |
>
>    改判 2 题：`cap_032`、`cap_035` 由 2 降到 1，两题改后都与人工判定相同。
>    `cap_017` 未改判（残余分歧，且存疑的是人工侧）；`cap_010` 的 2 档差距是
>    信息不对等，判据层面解决不了。**改前那批分留档在
>    `reports/judge_scores_v1_conclusion_level.jsonl`，报告里有改前改后对照表。**
>
>    **必须连带说明的限定**：v2 本质是让裁判向人工的既有尺子靠拢（人工本来就是
>    要点级标注的，见 v1 那轮解读，早于本次修订）。所以 kappa 的上升**有一部分是
>    构造性的**，不是两个独立评分者自发趋同。诚实的说法是"判据澄清后自动裁判
>    能复现人工的判定口径"。要测真正的独立一致性，得让人工在 v2 判据下重新盲标——
>    本轮没做。另外 n=10 下 95% CI 由 [0.091,0.750] 变成 [0.231,1.000]，上界顶到 1.0，
>    **区间的收窄程度配不上点估计的涨幅**。
>
> 9. 闸 B 新增第三条不可比理由：**裁判判据版本变了即 incomparable**。
>    换判据就是换量尺，同一批答案分数照样会动（这次跌 0.056，按容差判会报 dropped，
>    但被测系统一点没变）。判据版本记在 `judge_scores.jsonl` 的 `rubric_version` 字段，
>    经 `task_success_rate.backfill` 进报告。
>    代价是它也能给真实的变差打掩护——对策不在闸 B 里：口径变动必然要动
>    fixture 基线（闸 A 逐位比对）、`RUBRIC_VERSION` 或 config 阈值，全都会出现在 diff 里。
>
> **一条只改一次的纪律（本轮明确约定）**：判据收紧**只做这一轮**，改完认结果，
> 不许看着 kappa 再调第二轮。同 `expected_doc_ids` 那条血的教训——
> 一路"修"到数字好看为止，指标就失去意义。
>
> **一条关于标注的纪律（血的教训）**：`expected_doc_ids` 的完整性问题已经出现三次，
> 每修一次 recall 都往上走（0.919 → 0.932 → 0.935）。这个模式很危险——
> 若一路"修"到数字好看为止，指标就失去意义。因此规定：**标注只能依据通读语料
> 得出的书面判据来定，不许看着检索结果反推**；确需修订时，必须对**全部**题目
> 用同一判据重扫一遍，并在报告里写明修订前后的差值与归因。

---

## 7. 工程门禁（pytest + CI）

- `tests/` 既有**功能测试**（工具能调通、检索有返回、schema 生效），也有**指标回归门禁**。
- 门禁逻辑：`reports/` 存上一次的指标基线；CI 跑固定小子集，关键指标（任务成功率、工具调用准确率、faithfulness）相对基线**跌破容差**（阈值在 `config.py`，如跌超 0.05）就 **fail**。
- 这是你风控项目里 PSI 触发告警的同一思路，搬到 Agent 评测。README 里点明这个类比。
- **CI 里不调真实 DeepSeek API**：GitHub Actions 上不放 key、也为省钱和结果确定性，CI 里把 LLM 调用**打桩（mock）或读缓存响应**，只验证流水线和指标计算逻辑本身。真实全量评测在你本机手动跑。

---

## 8. 黄金集构造纪律（核心卖点，别省）

1. 题目**自建**或基于 `corpus/` 私有文档改写，**不许直接搬 MMLU/C-Eval 等公开 benchmark 原题**，避免评测被训练数据污染。
2. 每条标注 `expected_tool` 和 `expected_doc_ids`，让"工具选对没""检索准没"可判定。
3. 开放题才交裁判层；闭合题规则判定，能不用 LLM 判就不用（省钱、更可复现）。
4. README 写明这套防污染做法，并点出它和风控项目里 GroupKFold 防数据泄漏是**同一直觉**。

---

## 9. 里程碑（严格按序，每步结束测试绿）

- **M1 骨架**：目录、依赖、config、`.env` 载入、DeepSeek 客户端封装、一个 hello-world LangGraph 图能调通 `deepseek-chat`。
- **M2 被测系统**：RAG（切分→FAISS→检索）+ 两个工具 + ReAct 图，能对一问产出答案和完整轨迹。
- **M3 能力评测集 + 自定义指标**：造 30–50 条黄金集，实现 recall@k / 任务成功率 / 工具调用准确率 / 多轮一致性，出第一份 `reports/` 报告。
- **M4 RAGAS**：接入 faithfulness / relevancy / context recall。
- **M5 裁判层 + kappa**：LLM-as-Judge 给开放题打分，人工标注一小批，算 kappa，写进报告。
- **M6 CI 门禁**：pytest 全绿 + GitHub Actions 回归门禁生效 + README 定稿（指标可复现）。

M1–M6 完成即 v1 收工。**不要跳步，M6 之前不碰 v2。**

---

## 10. 你（Claude Code）的工作纪律

- 每个里程碑开始前，一两句说清交付物和验收标准，再动手。
- 先写测试/指标定义，再写实现。
- 不确定选型或范围就**停下来问**，不擅自扩范围、加依赖。
- 绝不编造指标；绝不把 API key 写进代码或提交；确认 `.env` 已被 gitignore。
- commit 粒度小、信息清楚（如 `feat(agent): add retrieve tool`）。
- 每步结束报告：做了什么、测试是否绿、下一步建议。
- **省钱意识**：开发调试时先用小子集（3–5 题）跑通逻辑，确认无误再跑全量，避免反复全量重跑烧 token。

---

## 11. v2 待办（现在只登记，不实现）

- 安全评测：越狱 / 提示词注入 / 越权用例集 + 攻击成功率 ASR。
- FastAPI 打分服务 + Streamlit 可视化 + Docker，做成 HR 能点开的 demo。
- MLflow 记录每轮指标；LangSmith / Langfuse 做调用链 trace 观测。
- 检索加 rerank；向量库加 Chroma 做选型对照。

---

## 12. 目标简历 bullet（建整个项目就是为了让这句话每个数字都为真）

> 构建 RAG Agent 端到端评测流水线（LangGraph + FAISS + RAGAS）：对 ___ 条自建黄金集自动评估任务成功率、工具调用准确率、检索召回率 recall@k、答案忠实度与多轮一致性；引入 LLM-as-Judge 并与人工标注对照，报出自动↔人工一致率 kappa=___；评测标准以 pytest 固化并接入 GitHub Actions 做提交级回归门禁，指标漂移即 fail。黄金集采用防污染构造（不复用公开 benchmark），沿用信贷风控项目中防数据泄漏的同一方法论。

空格处的数字，等项目跑出真实值再填。填不出来说明那块还没做完——不许提前填。
