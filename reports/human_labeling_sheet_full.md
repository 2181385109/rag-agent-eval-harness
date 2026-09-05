# 人工标注表（开放题）

共 **10** 道开放题。闭合题已由 `answer_keys` 规则判定，不需要人工打分。

## 怎么标

给每道题的 **Agent 答案**打一个分，判据如下：

| 分数 | 判据 |
|---|---|
| **0** | 错误 / 编造 / 答非所问：与参考答案矛盾，或凭空捏造了知识库里没有的内容 |
| **1** | 部分正确：方向对、无编造，但要点有遗漏或有轻微不准确 |
| **2** | 正确且有据：要点齐全，结论与参考答案一致，且未编造 |

几点提醒：

- 判的是 **Agent 答案**相对 **参考答案** 的质量，不是判问题出得好不好。
- 标了 `hallucination_bait` 的题，**正确行为是明说「知识库里没有」**。如果 Agent 编出了一个看似合理的答案，那是 0 分，不是 2 分。
- 请**先标完再看自动裁判的结果**。先看机器分再标，kappa 度量的就不是人机独立看法是否一致，而是确认偏误。
- 样本量只有 10 条，算出来的 kappa 会很不稳定（置信区间宽），报告里会如实标注这一点，不要把它当硬结论。

标完把分数告诉我（形如 `cap_007=2, cap_008=1, ...`），或直接编辑 `human_labels.jsonl` 里的 `human_score` 字段。

---

## 逐题

### 1. `cap_007`

**问题**：信贷风控项目和航班延误项目，各自是怎么防止数据泄露的？

**参考答案**：

> 信贷项目：按标签分层切分 70/15/15，所有预处理统计量只在训练集上 fit 再应用到验证/测试集。航班项目：主用时间序切分（前 80% 训练、后 20% 测试），另做按 tail_id 的 GroupKFold 检验对新机尾的泛化；carrier_ontime_rate 这类统计特征在每折内计算后再映射到验证折。

**Agent 答案**：

> 我已经收集到足够的信息来回答这个问题。让我整理两个项目各自防止数据泄露的措施。
>
> ## 信贷风控项目（credit-risk-mlops）如何防止数据泄露
>
> 1. **数据切分防泄露**：数据在进入特征工程前，先**按行分层切分为训练/验证/测试集**（`data` 阶段即完成，防止后续处理把测试信息带入训练）。
>
> 2. **预处理器只在训练集上 fit**：缺失填补（中位数/分组中位数）、异常截断（分位数）、WOE 编码等统计量，全部通过 `Preprocessor.fit(X_train)` 只在训练集上学习，`transform(X)` 再复用到验证/测试集，避免用全量数据统计量造成泄露。预处理器可序列化复用，验收标准明确要求"**无泄露**"。
>
> 3. **派生特征不引入未来/标签信息**：派生特征（总逾期次数、人均月收入、交互项、年龄分桶）均基于当前样本自身字段构造，不涉及标签或未来信息。
>
> ## 航班延误项目如何防止数据泄露
>
> 1. **特征工程明确"严禁数据泄露"**：`features.py` 中所有特征都刻意避免使用会造成泄露的信息。
>
> 2. **`airport_congestion` 只用计划信息**：机场拥堵度用起飞时刻 ±30min 窗口内的**计划起飞架次**计算，而非实际起飞信息，避免把"结果"当特征。
>
> 3. **`carrier_ontime_rate` 用 out-of-fold / 仅训练集统计**：承运人历史准点率这类统计特征，在**交叉验证（CV）内计算**，只用训练折统计，防止验证/测试折信息泄漏进特征。
>
> 4. **`stats` 承载训练集统计量供复用**：`build_features(df, fit_stats=None)` 返回的 `stats` 只在训练集上计算，验证/测试集通过传入 `fit_stats` 复用，保证无泄露。
>
> 5. **风险分级切点由训练集定**：`fit_quantiles(train_pred)` 的分位切点只在训练集上确定，测试集复用切点。
>
> 6. **`prev_leg_delay`（延误传播）** 作为主特征，其构造基于前序航段信息，属于因果结构设计而非标签泄露。
>
> ---
>
> **小结**：两个项目防泄露的核心思路一致——**所有统计量/切点/编码只在训练集（或训练折）上拟合，验证与测试集只做复用**；区别在于信贷项目侧重"按行分层切分 + 预处理器 fit/transform 分离"，航班项目侧重"CV 内计算统计特征 + 只用计划信息构造特征 + 分位切点由训练集确定"。

<details><summary>检索到的片段（17 块）</summary>

**`c730fa5c_0010`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 4. 项目结构
>
> **根目录：`D:\xiangmu\credit-risk-mlops`**

**`c730fa5c_0035`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 19. 落地路径与 GitHub 发布（M7）
>
> ### 19.1 落地
> 项目根目录 `D:\xiangmu\credit-risk-mlops`，全部代码/配置/README/测试/`outputs/reports` 置于此。
>
> ### 19.2 `.gitignore`（提交前必须就位）
> ```
> .venv/   __pycache__/   *.pyc
> data/raw/   data/processed/   *.csv
> mlruns/   outputs/models/   *.pkl
> outputs/figures/*.html
> .env   .idea/   .vscode/
> ```
> > 提交：源码、`config.yaml`、`Dockerfile`、`docker-compose.yml`、`README.md`、`tests/`、`outputs/reports/report.md`、少量小体积展示图（`*.png`）。数据、`mlruns/`、模型工件**不提交**。

**`c730fa5c_0037`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 20. 交付物清单
>
> - [ ] 纯 Python 代码库（第 4 节结构），位于 `D:\xiangmu\credit-risk-mlops`
> - [ ] `python -m creditrisk run-all` 一键跑通 数据→训练→评估
> - [ ] LightGBM 风控模型（+ 逻辑回归/常数基线）+ 不平衡处理 + SHAP + 评分卡
> - [ ] MLflow 实验追踪 + 模型注册
> - [ ] FastAPI 在线打分服务（含输入校验与 Swagger 文档）
> - [ ] Dockerfile（+ 可选 docker-compose），镜像可构建可运行
> - [ ] PSI 数据漂移监控（含注入漂移演示）
> - [ ] Streamlit 面板（概览 / 在线打分 / 漂移监控）
> - [ ] `pytest` 测试套件全绿（含 API 测试）
> - [ ] GitHub Actions CI（ruff + pytest）
> - [ ] README（架构图、指标表、运行说明、技术选型、面试讲法）
> - [ ] 项目已推送至用户 GitHub，仓库可访问

**`c730fa5c_0018`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 6. 数据层规格（`data.py`）
>
> ### 6.1 数据集：Give Me Some Credit
> - Kaggle 竞赛数据，训练集 `cs-training.csv`，约 **15 万行**、文件仅几 MB（远小于 200MB）。
> - **标签**：`SeriousDlqin2yrs`（1 = 两年内发生 90 天以上逾期，正类；**约占 6~7%，天然不平衡**）。
> - **10 个特征**：授信循环额度使用率、年龄、30–59/60–89/90+ 天逾期次数、负债比、月收入、开放信贷额度数、房产贷款数、家属人数。
> - **已知数据问题（都要处理，也都是面试谈资）**：`MonthlyIncome`、`NumberOfDependents` 有缺失；逾期次数字段存在异常哨兵值（如 96/98）；使用率与负债比有极端离群值。

**`2da65ad3_0003`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 1. 项目目标与范围
>
> 构建一个**"预测—调度"闭环系统**，解决多因素非线性航班延误问题：
>
> 1. **预测环**：基于航班静态/动态/气象特征，预测每个航班的起飞延误。
> 2. **分级环**：将预测结果映射为 5 级风险等级（分位数策略）。
> 3. **调度环**：以风险为输入，通过约束优化调整高风险航班的起飞时隙/缓冲，在满足运行约束下最小化加权总延误。
> 4. **评估 + 可视化**：复现三大核心指标并出图；提供交互面板。
>
> **核心指标（复现目标，须真实达成或如实报告）**：

**`2da65ad3_0008`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ```
> 合成/真实航班数据
>       │
>       ▼
> [data]     清洗 + 结构化（含机尾串接，支持延误传播）
>       │
>       ▼
> [features] 7 核心特征 + 3 交互特征（防泄露）
>       │
>       ▼
> [predict]  LightGBM 回归(延误分钟) + 分类(P[延误>15min])  ─┐  基线: RF / 规则
>       │                                                    │
>       ▼                                                    ▼
> [risk]     分位数 5 级分级（L1..L5，L4/L5=高风险）      特征重要性/SHAP
>       │
>       ▼
> [schedule] CP-SAT 约束优化（时隙偏移/缓冲）  ── 基线: 贪心
>       │
>       ▼
> [evaluate] 三大指标 + 基线对比 + 回归误差
>       │
>       ▼

**`2da65ad3_0018`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 6. 数据层规格
>
> ### 6.1 合成数据生成器（`data/synth.py`，主数据源）
>
> 生成 **≈2000 条航班记录**，并**故意注入因果结构**，使特征对标签真实可预测、三大指标可复现。
>
> **实体设定**（可在 config 调整）：
> - 机场 8–12 个；承运人 5–8 家；机型 4–6 种。
> - 飞机 ≈120 架，每架当天飞**多段航线（2–4 段）**，用同一 `tail_id` 串接 → 支持**延误传播**。
> - 每机场每小时一个**天气状态**（能见度、风速、降水、雷暴标志），时空平滑。
>
> **字段 Schema**：

**`2da65ad3_0000`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> # 航班延误预测与智能调度优化系统 · 技术规格书
>
> > **本文档面向 Claude Code。** 请把它当作一份完整的项目实施规格（PRD + 技术设计），用 **Python** 自主完成端到端开发，最终把项目落地到本地路径 **`D:\xiangmu`**，并**发布到用户的 GitHub**。目标是交付一个可运行、可测试、可复现、作品集级别的项目。
>
> ---
>
> ## 0. 给 Claude Code 的执行说明（先读这一节）

**`2da65ad3_0023`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ## 7. 特征工程规格（`features.py`）
>
> 输入统一 Schema 的 DataFrame，输出模型可用特征矩阵。**严禁数据泄露**。
>
> **7 个核心特征**：
> 1. `weather_severity`：气象量归一化加权合成（0–1）。
> 2. 时段编码：`sched_dep` 小时的**周期编码**（sin/cos），派生 `time_bucket` 与 `is_peak`。
> 3. `airport_congestion`：起飞机场起飞时刻 ±30min 窗口内计划起飞架次（仅用计划信息，无泄露）。
> 4. `carrier_ontime_rate`：承运人历史准点率，**用 out-of-fold / 仅训练集统计**（见第 8 节 CV 内计算）。
> 5. `distance`（派生 `sched_duration`）。
> 6. 日历：`day_of_week`、`is_weekend`、`is_holiday`。
> 7. `prev_leg_delay`：延误传播主特征。

**`c730fa5c_0021`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> - **缺失填补**：`MonthlyIncome` 用中位数或分组中位数；`NumberOfDependents` 用众数/0；并加"是否缺失"指示位。
> - **异常处理**：逾期次数字段的哨兵值（96/98）截断或单独成桶；使用率、负债比按分位数（如 99%）截断。
> - **派生特征（少而有信号）**：总逾期次数、人均月收入（收入/(家属+1)）、月收入与负债比的交互、年龄分桶。
> - **（可选风控味）WOE 编码**：对分桶特征做 WOE 变换，附带 IV 值，为"评分卡"做铺垫（时间紧可跳过，README 注明）。
> - **接口**：`Preprocessor.fit(X_train)`；`transform(X)`；`save/load`（joblib 序列化到 `outputs/models/preprocessor.pkl`）。
> - 输出"特征说明表"到 `outputs/reports/feature_dict.md`。
>
> **验收**：无泄露；预处理器可序列化并复用；`transform` 对单条与批量都可用（服务层需要）。
>
> ---

**`2da65ad3_0022`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **验收**：Schema 完整；正类比例在区间；`prev_leg_delay`、`weather_severity`、`congestion` 与 `dep_delay` 正相关显著；固定 seed 完全复现。
>
> ### 6.2 公开数据加载器（`data/loader.py`，可选）
>
> - `load_public(csv_path) -> DataFrame`，把常见公开航班数据（如 BTS/Kaggle "Flight Delays"）适配到 6.1 统一 Schema。
> - 缺失字段用机型默认值填充；无 `tail_id` 时按 (aircraft, 时间序) 近似串接。
> - `data/raw/` 无文件则跳过，默认走合成数据。
>
> ---
>
> ## 7. 特征工程规格（`features.py`）
>
> 输入统一 Schema 的 DataFrame，输出模型可用特征矩阵。**严禁数据泄露**。

**`c730fa5c_0008`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ```
> Give Me Some Credit 数据 (cs-training.csv)
>       │
>       ▼
> [data]     清洗(缺失/异常/哨兵值) + 按行分层切分 训练/验证/测试(防泄露)
>       │
>       ▼
> [features] 缺失填补 + 异常截断 + 派生特征 (+可选 WOE)；预处理器可序列化
>       │
>       ▼
> [train]    LightGBM(不平衡加权) + Optuna 调参    基线: 逻辑回归
>       │        │
>       │        ├──► [tracking] MLflow: 记录 params/metrics/artifacts + 注册模型
>       ▼        ▼
> [evaluate] AUC/KS/PR-AUC/校准 + SHAP + 评分卡    对照基线
>       │
>       ▼
> [registry] 最优模型 + 预处理器 打包为部署工件
>       │

**`c730fa5c_0023`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ### 8.2 MLflow 追踪与注册
> - 每次训练用 MLflow 记录：超参、AUC/KS/PR-AUC、ROC/KS/校准图、SHAP 摘要图、模型工件。
> - 用 **Model Registry** 注册最优模型为 `credit-risk-model`，打 stage 标记（如 `Staging`/`Production`）。
> - 本地 `mlruns/` 追踪，`mlflow ui` 可查看对比。
>
> ### 8.3 评估（`evaluate.py`）
> - **AUC、KS、PR-AUC**、指定阈值下的混淆矩阵/精确率/召回率、**校准曲线**。
> - **SHAP**：全局重要性 + 摘要图；能讲清主要驱动因素（逾期历史、使用率等）。
> - **阈值选择**：给出按 KS 最大点或按业务代价的阈值建议。
> - **评分卡映射**：把违约概率映射为直观信用分（如 300~850），写进报告。
> - 落盘 `outputs/reports/metrics.json` 与 `report.md`，含基线对比；**自动写回 README 占位指标表**。

**`2da65ad3_0031`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **验收**：CP-SAT 约束满足率 ≥ 0.85（并展示相对贪心的提升）；高风险航班平均延误较优化前下降 ≥ 1.0 min/架次。
>
> ---
>
> ## 11. 评估模块规格（`evaluate.py`）
>
> - 汇总落盘 `outputs/reports/metrics.json` 与 `report.md`，含：
>   - 回归 MAE/RMSE（LightGBM vs RF vs 均值基线）。
>   - 高风险 Recall/Precision/F1/PR-AUC（vs 规则基线）。
>   - 调度约束满足率、高风险延误减少（CP-SAT vs 贪心 vs 优化前）。
> - 每个指标须有**明确定义 + 基线对比**。
> - `summarize(...) -> dict`，并把最终指标表**自动写回 README** 的占位表格。
>
> ---
>
> ## 12. 可视化与交互界面
>
> ### 12.1 静态图（`viz.py`）
> 延误分布、特征重要性、SHAP 摘要、混淆矩阵/PR 曲线、优化前后延误对比条形图。

**`2da65ad3_0024`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **3 个交互特征**：
> 1. `weather_severity × is_peak`
> 2. `airport_congestion × is_peak`
> 3. `prev_leg_delay × turnaround_slack`（`turnaround_slack = 计划周转 − min_turnaround`）
>
> **要求**：
> - 类别特征交给 LightGBM 原生处理；给 RF 基线则 One-Hot。
> - 输出"特征字典"（`outputs/reports/feature_dict.md`），逐项说明定义与动机。
> - `build_features(df, fit_stats=None) -> (X, y_reg, y_clf, stats)`，`stats` 承载训练集统计量供验证/测试集无泄露复用。
>
> ---
>
> ## 8. 预测模块规格（`predict.py`）
>
> **任务**：双头预测 —— 回归 `dep_delay` + 二分类 `is_delayed15`。

**`2da65ad3_0021`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> dep_delay = max(0, base_delay)
> ```
> - 系数 `a*`、`sigma` 放入 config，取值使 `is_delayed15` 正类占比落在 **20%–35%**，各特征均有可辨识信号。
> - 生成器落盘 `data/processed/flights.parquet`。

**`2da65ad3_0028`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 9. 风险分级规格（`risk.py`）
>
> - 以**预测延误值**（或分类概率，二选一并说明）在训练集上的 20/40/60/80 分位点为切点，分 5 级 L1..L5。
> - **高风险 = L4/L5**。
> - `fit_quantiles(train_pred) -> cutpoints`（切点由训练集定，测试集复用）；`grade(pred, cutpoints) -> labels`。
> - 输出高风险捕获率（Recall）并对照规则基线，写入指标报告。
>
> **验收**：各级样本量大致均衡；高风险 Recall ≥ 0.80 且显著高于规则基线。
>
> ---
>
> ## 10. 调度优化模块规格（`schedule.py`）
>
> ### 10.1 问题建模（地面延误 / 时隙分配）


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 2. `cap_008`

**问题**：在信贷违约这种类别不平衡的场景下，为什么不能用准确率评估模型？应该看什么指标？

**参考答案**：

> 因为正类仅约 6%，把所有样本都预测成不违约准确率也有 94%，却毫无用处。应该用 AUC、PR-AUC、KS 这些对不平衡稳健、关注排序与分离能力的指标。

**Agent 答案**：

> 根据知识库内容，我来回答这个问题。
>
> ## 为什么不能用准确率评估
>
> 在信贷违约这种类别不平衡场景下（如 Give Me Some Credit 数据集中，正类"违约"仅占约 **6~7%**），准确率会严重失真：
>
> - 如果模型**把所有样本都预测为"不违约"**，准确率也能达到约 **94%**，看似很高，但这个模型毫无用处——它一个违约客户都识别不出来。
> - 准确率只反映"预测对的比例"，在极端不平衡下会被多数类（不违约）主导，无法反映模型对少数类（违约）的识别能力，而违约客户恰恰是风控最关心的对象。
>
> ## 应该看什么指标
>
> 知识库中明确推荐对不平衡稳健、关注**排序/分离能力**的指标：
>
> | 指标 | 说明 |
> |---|---|
> | **AUC（ROC-AUC）** | 衡量模型把违约/不违约样本排序分开的能力，对类别比例不敏感 |
> | **PR-AUC** | 精确率-召回率曲线下面积，**不平衡场景下比 ROC-AUC 更看重**（因为更关注少数类） |
> | **KS** | 好坏样本累积分布之差的最大值，衡量模型把违约/不违约"拉开"的能力，是风控行业标准 |
> | **精确率 / 召回率 / F1** | 在指定阈值下评估，关注违约类（正类）的识别效果 |
> | **混淆矩阵** | 结合阈值查看具体分类表现 |
>
> 此外，风控中还会结合**业务代价**来选择阈值（如按误拒/漏放的成本权衡），并用 **PSI** 监控线上分布漂移。
>
> **核心结论**：不平衡场景下应优先看 **AUC、PR-AUC、KS** 这类对类别比例稳健、关注排序/分离能力的指标，而不是准确率。

<details><summary>检索到的片段（7 块）</summary>

**`c730fa5c_0021`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> - **缺失填补**：`MonthlyIncome` 用中位数或分组中位数；`NumberOfDependents` 用众数/0；并加"是否缺失"指示位。
> - **异常处理**：逾期次数字段的哨兵值（96/98）截断或单独成桶；使用率、负债比按分位数（如 99%）截断。
> - **派生特征（少而有信号）**：总逾期次数、人均月收入（收入/(家属+1)）、月收入与负债比的交互、年龄分桶。
> - **（可选风控味）WOE 编码**：对分桶特征做 WOE 变换，附带 IV 值，为"评分卡"做铺垫（时间紧可跳过，README 注明）。
> - **接口**：`Preprocessor.fit(X_train)`；`transform(X)`；`save/load`（joblib 序列化到 `outputs/models/preprocessor.pkl`）。
> - 输出"特征说明表"到 `outputs/reports/feature_dict.md`。
>
> **验收**：无泄露；预处理器可序列化并复用；`transform` 对单条与批量都可用（服务层需要）。
>
> ---

**`c730fa5c_0018`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 6. 数据层规格（`data.py`）
>
> ### 6.1 数据集：Give Me Some Credit
> - Kaggle 竞赛数据，训练集 `cs-training.csv`，约 **15 万行**、文件仅几 MB（远小于 200MB）。
> - **标签**：`SeriousDlqin2yrs`（1 = 两年内发生 90 天以上逾期，正类；**约占 6~7%，天然不平衡**）。
> - **10 个特征**：授信循环额度使用率、年龄、30–59/60–89/90+ 天逾期次数、负债比、月收入、开放信贷额度数、房产贷款数、家属人数。
> - **已知数据问题（都要处理，也都是面试谈资）**：`MonthlyIncome`、`NumberOfDependents` 有缺失；逾期次数字段存在异常哨兵值（如 96/98）；使用率与负债比有极端离群值。

**`c730fa5c_0023`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ### 8.2 MLflow 追踪与注册
> - 每次训练用 MLflow 记录：超参、AUC/KS/PR-AUC、ROC/KS/校准图、SHAP 摘要图、模型工件。
> - 用 **Model Registry** 注册最优模型为 `credit-risk-model`，打 stage 标记（如 `Staging`/`Production`）。
> - 本地 `mlruns/` 追踪，`mlflow ui` 可查看对比。
>
> ### 8.3 评估（`evaluate.py`）
> - **AUC、KS、PR-AUC**、指定阈值下的混淆矩阵/精确率/召回率、**校准曲线**。
> - **SHAP**：全局重要性 + 摘要图；能讲清主要驱动因素（逾期历史、使用率等）。
> - **阈值选择**：给出按 KS 最大点或按业务代价的阈值建议。
> - **评分卡映射**：把违约概率映射为直观信用分（如 300~850），写进报告。
> - 落盘 `outputs/reports/metrics.json` 与 `report.md`，含基线对比；**自动写回 README 占位指标表**。

**`c730fa5c_0004`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **核心指标（复现目标，须真实达成或如实报告，不许编数）**：
>
> | 指标 | 定义 | 目标 | 对照基线 |
> |---|---|---|---|
> | AUC | ROC-AUC | LightGBM > 逻辑回归；参考 ~0.86 | 逻辑回归 / 常数基线 |
> | KS | 好坏样本分离度 max(TPR−FPR) | 越高越好，须报告 | 逻辑回归 |
> | PR-AUC | 精确率-召回率曲线下面积（不平衡更看重） | LightGBM > 逻辑回归 | 逻辑回归 |
> | PSI（监控） | 分布稳定性指数 | 演示：注入漂移后能检出（PSI>0.25 触发告警） | — |
>
> > 参考量级（仅 sanity check）：Give Me Some Credit 上好模型 AUC 常见 0.86~0.87。**以实跑为准并如实写入 README。**
>
> ---
>
> ## 2. 技术选型

**`c730fa5c_0039`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **面试高频问，先备好答案**：
> - **不平衡数据为什么不看准确率？** 正类仅约 6%，全预测为"不违约"准确率也有 94% 却毫无用处；所以用 AUC、PR-AUC、KS 这些对不平衡稳健、且关注排序/分离能力的指标。
> - **KS 是什么？** 好坏样本累积分布之差的最大值，衡量模型把违约/不违约拉开的能力，是风控行业标准。
> - **PSI 是什么、为什么要监控漂移？** 群体稳定性指数，比较线上数据分布与训练时的偏移；分布漂移会让模型悄悄失效，PSI>0.25 就该预警甚至重训。
> - **为什么要把模型做成服务 + 容器？** 让模型能被真实系统调用、环境可一键复现，是"能用的模型"和"上线的模型"的区别。
> - **怎么保证训练和线上用的是同一套处理？** 预处理器随模型一起序列化、服务加载同一工件，避免训练-服务偏移（training-serving skew）。
> - **MLflow 解决什么？** 实验可追溯、可对比、模型可版本化注册，团队协作和回滚都靠它。
>
> ---

**`2da65ad3_0027`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **输出**：
> - 模型序列化至 `outputs/models/`。
> - 回归 MAE/RMSE；分类 Precision/Recall/F1/PR-AUC/混淆矩阵。
> - **特征重要性 + SHAP** 摘要图（`outputs/figures/`）。
> - 接口：`train(X, y_reg, y_clf, cfg) -> models`；`predict(models, X) -> {"delay":..., "risk_proba":...}`。
>
> **验收**：LightGBM 在测试集**优于 RF 与规则基线**；SHAP 能识别天气/传播/拥堵为主要驱动。
>
> ---
>
> ## 9. 风险分级规格（`risk.py`）

**`2da65ad3_0031`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **验收**：CP-SAT 约束满足率 ≥ 0.85（并展示相对贪心的提升）；高风险航班平均延误较优化前下降 ≥ 1.0 min/架次。
>
> ---
>
> ## 11. 评估模块规格（`evaluate.py`）
>
> - 汇总落盘 `outputs/reports/metrics.json` 与 `report.md`，含：
>   - 回归 MAE/RMSE（LightGBM vs RF vs 均值基线）。
>   - 高风险 Recall/Precision/F1/PR-AUC（vs 规则基线）。
>   - 调度约束满足率、高风险延误减少（CP-SAT vs 贪心 vs 优化前）。
> - 每个指标须有**明确定义 + 基线对比**。
> - `summarize(...) -> dict`，并把最终指标表**自动写回 README** 的占位表格。
>
> ---
>
> ## 12. 可视化与交互界面
>
> ### 12.1 静态图（`viz.py`）
> 延误分布、特征重要性、SHAP 摘要、混淆矩阵/PR 曲线、优化前后延误对比条形图。


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 3. `cap_010`  ⚠ `hallucination_bait`

**问题**：信贷风控项目的技术规格书里，指定用 Redis 做特征存储吗？

**参考答案**：

> 没有。规格书的技术选型里没有提到 Redis 或任何特征存储组件。正确行为是明确说明知识库中未提及，而不是编造。

**Agent 答案**：

> 根据对信贷风控项目技术规格书（`信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md`）的完整检索，我可以明确回答：
>
> **没有。** 该技术规格书**并未指定用 Redis 做特征存储**。
>
> 具体依据如下：
>
> 1. **技术选型表**（第2节）中列出的方案为：LightGBM、Optuna、SHAP、MLflow、FastAPI、Docker，**完全没有提到 Redis**。
>
> 2. **特征工程**部分（第7节）的做法是：做一个**可序列化的预处理器**（`Preprocessor.fit/transform`），用 **joblib 序列化到 `outputs/models/preprocessor.pkl`**，以保证训练与线上服务用同一套变换、避免训练-服务偏移。特征存储靠的是本地 pkl 工件，而非 Redis。
>
> 3. **服务化**部分（第9节）用的是 **FastAPI + Pydantic**，启动时加载部署工件（`preprocessor.pkl` + MLflow 注册模型），也没有引入 Redis 作为特征缓存。
>
> 4. 检索结果中出现的 Redis 相关内容来自**其他项目**（如航班延误预测项目），并非信贷风控项目。
>
> 因此，这份信贷风控技术规格书里**没有指定 Redis 做特征存储**。

<details><summary>检索到的片段（20 块）</summary>

**`c730fa5c_0000`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> # 信贷违约风控 · 端到端机器学习系统（建模 + MLOps）· 技术规格书
>
> > **本文档面向 Claude Code。** 请把它当作一份完整的项目实施规格（PRD + 技术设计），用 **Python** 自主完成端到端开发，最终落地到本地路径 **`D:\xiangmu\credit-risk-mlops`**，并**发布到用户的 GitHub**。目标是交付一个可运行、可测试、可复现、**面试拿得出手**的作品：一个信贷违约风控模型，外加**从建模到上线的完整 MLOps 链路**。
>
> ---
>
> ## 0. 给 Claude Code 的执行说明（先读这一节）

**`c730fa5c_0021`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> - **缺失填补**：`MonthlyIncome` 用中位数或分组中位数；`NumberOfDependents` 用众数/0；并加"是否缺失"指示位。
> - **异常处理**：逾期次数字段的哨兵值（96/98）截断或单独成桶；使用率、负债比按分位数（如 99%）截断。
> - **派生特征（少而有信号）**：总逾期次数、人均月收入（收入/(家属+1)）、月收入与负债比的交互、年龄分桶。
> - **（可选风控味）WOE 编码**：对分桶特征做 WOE 变换，附带 IV 值，为"评分卡"做铺垫（时间紧可跳过，README 注明）。
> - **接口**：`Preprocessor.fit(X_train)`；`transform(X)`；`save/load`（joblib 序列化到 `outputs/models/preprocessor.pkl`）。
> - 输出"特征说明表"到 `outputs/reports/feature_dict.md`。
>
> **验收**：无泄露；预处理器可序列化并复用；`transform` 对单条与批量都可用（服务层需要）。
>
> ---

**`c730fa5c_0010`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 4. 项目结构
>
> **根目录：`D:\xiangmu\credit-risk-mlops`**

**`c730fa5c_0005`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> | 环节 | 采用方案 | 保留基线/理由 |
> |---|---|---|
> | 主模型 | **LightGBM**（二分类，`scale_pos_weight` 处理不平衡） | 表格数据强、快、CPU 友好；基线用**逻辑回归** |
> | 调参 | **Optuna**（轻量，20~50 trial，目标 AUC） | 比网格搜索样本效率高 |
> | 可解释 | **SHAP** + 特征重要性 | 风控强需求：讲清"为什么拒绝" |
> | 评估 | AUC / **KS** / PR-AUC / 混淆矩阵 / 校准 + **评分卡映射** | KS/评分卡是风控行业标准 |
> | 实验追踪 | **MLflow**（本地 file-based，含 Model Registry） | 无需起服务器，稳；面试常问 |
> | 服务化 | **FastAPI** + Pydantic 校验 + Uvicorn | 主流、轻、自带文档页 |
> | 容器化 | **Docker**（+ 可选 docker-compose 同起 API 与 MLflow UI） | 一键复现环境 |

**`2da65ad3_0024`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **3 个交互特征**：
> 1. `weather_severity × is_peak`
> 2. `airport_congestion × is_peak`
> 3. `prev_leg_delay × turnaround_slack`（`turnaround_slack = 计划周转 − min_turnaround`）
>
> **要求**：
> - 类别特征交给 LightGBM 原生处理；给 RF 基线则 One-Hot。
> - 输出"特征字典"（`outputs/reports/feature_dict.md`），逐项说明定义与动机。
> - `build_features(df, fit_stats=None) -> (X, y_reg, y_clf, stats)`，`stats` 承载训练集统计量供验证/测试集无泄露复用。
>
> ---
>
> ## 8. 预测模块规格（`predict.py`）
>
> **任务**：双头预测 —— 回归 `dep_delay` + 二分类 `is_delayed15`。

**`2da65ad3_0021`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> dep_delay = max(0, base_delay)
> ```
> - 系数 `a*`、`sigma` 放入 config，取值使 `is_delayed15` 正类占比落在 **20%–35%**，各特征均有可辨识信号。
> - 生成器落盘 `data/processed/flights.parquet`。

**`c730fa5c_0033`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> | 里程碑 | 内容 | 完成标准（DoD） |
> |---|---|---|
> | **M1 数据** | 获取 + 清洗 + 分层切分 + 测试 | `prep` 产出三集，`test_data` 绿，无泄露 |
> | **M2 特征** | 可序列化预处理器 + 特征说明表 + 测试 | 无泄露、单条/批量一致，`test_features` 绿 |
> | **M3 建模+追踪** | LightGBM+LR + Optuna + MLflow 注册 + SHAP | 优于基线，实验入 MLflow，`test_model` 绿 |
> | **M4 评估** | AUC/KS/PR-AUC/校准 + 评分卡 | 指标报告生成，`test_evaluate` 绿，README 表填充 |
> | **M5 服务+容器** | FastAPI + Docker | `/predict` 可用，`docker build/run` 通过，`test_api` 绿 |

**`c730fa5c_0020`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 7. 特征工程规格（`features.py`）
>
> 做一个可序列化的**预处理器**（`fit`/`transform`），保证训练与线上服务用**同一套变换**（避免训练-服务偏移，这是面试重点）。

**`2da65ad3_0023`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ## 7. 特征工程规格（`features.py`）
>
> 输入统一 Schema 的 DataFrame，输出模型可用特征矩阵。**严禁数据泄露**。
>
> **7 个核心特征**：
> 1. `weather_severity`：气象量归一化加权合成（0–1）。
> 2. 时段编码：`sched_dep` 小时的**周期编码**（sin/cos），派生 `time_bucket` 与 `is_peak`。
> 3. `airport_congestion`：起飞机场起飞时刻 ±30min 窗口内计划起飞架次（仅用计划信息，无泄露）。
> 4. `carrier_ontime_rate`：承运人历史准点率，**用 out-of-fold / 仅训练集统计**（见第 8 节 CV 内计算）。
> 5. `distance`（派生 `sched_duration`）。
> 6. 日历：`day_of_week`、`is_weekend`、`is_holiday`。
> 7. `prev_leg_delay`：延误传播主特征。

**`c730fa5c_0019`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ### 6.2 获取方式（Claude Code 先检查，缺失则提示用户）
> - 优先用 Kaggle API：`kaggle competitions download -c GiveMeSomeCredit`（需用户已放置 `kaggle.json` 凭据）。
> - **若无凭据**：提示用户自行从 Kaggle 下载 `cs-training.csv` 放入 `data/raw/`，然后继续。**不要在代码里写入任何账号密码或 token。**
>
> ### 6.3 切分（防泄露）
> - 官方测试集无标签，故**只用 `cs-training.csv`**，按标签**分层**切分为 训练/验证/测试 = 70/15/15（`stratify` 保持正类比例一致）。
> - **所有填补/截断/编码统计量只在训练集上 fit**，再 transform 验证/测试集。
>
> **验收**：切分分层正确、正类比例三集一致；预处理统计仅来自训练集；固定 seed 完全复现。
>
> ---
>
> ## 7. 特征工程规格（`features.py`）

**`c730fa5c_0014`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> │   ├── serve.py             # FastAPI app
> │   └── cli.py               # 命令行入口（typer）
> ├── app/streamlit_app.py     # 监控 + 演示面板
> └── tests/
>     ├── test_data.py
>     ├── test_features.py
>     ├── test_model.py
>     ├── test_evaluate.py
>     ├── test_monitor.py
>     └── test_api.py
> ```

**`c730fa5c_0025`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 9. 服务化规格（`serve.py` + `schemas.py`）
>
> - **FastAPI 应用**，启动时加载部署工件（`preprocessor.pkl` + 注册模型）。
> - **接口**：
>   - `GET /health`：健康检查。
>   - `POST /predict`：入参为单个申请人的原始字段（Pydantic 校验类型/范围），返回 `{default_probability, credit_score, risk_level, top_reasons}`（`top_reasons` 用 SHAP 给出主要影响因子）。
>   - `POST /predict/batch`：批量。
> - **输入校验**：用 `schemas.py` 的 Pydantic 模型，非法输入返回 422 并给出清晰错误。
> - 自带 Swagger 文档页（`/docs`）。
> - 启动：`uvicorn creditrisk.serve:app --reload`。

**`c730fa5c_0003`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 1. 项目目标与范围
>
> 构建一个**端到端机器学习系统**，覆盖真实企业里"模型从开发到上线"的完整生命周期：
>
> 1. **建模层**：在信贷数据上训练违约概率预测模型（二分类，处理类别不平衡），做完整评估与可解释。
> 2. **训练流水线 + 实验追踪**：可复现的训练管线，用 MLflow 记录参数/指标/产物并做模型注册。
> 3. **服务化**：用 FastAPI 把模型包成 REST API（在线打分）。
> 4. **容器化 + CI/CD**：Docker 打包，GitHub Actions 自动 lint + 测试。
> 5. **上线监控**：数据漂移监控（PSI），并用 Streamlit 面板做监控 + 演示。
>
> **核心指标（复现目标，须真实达成或如实报告，不许编数）**：

**`c730fa5c_0009`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> │
>       ▼
> [registry] 最优模型 + 预处理器 打包为部署工件
>       │
>       ├───────────────► [serve]  FastAPI /predict → Docker 镜像
>       │
>       └───────────────► [monitor] PSI 漂移检测(注入漂移演示) + Streamlit 面板
> ```

**`2da65ad3_0004`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **核心指标（复现目标，须真实达成或如实报告）**：
>
> | 指标 | 定义 | 目标 | 对照基线 |
> |---|---|---|---|
> | 高风险捕获率 | 高风险类 Recall（同时报 Precision/F1） | ≥ 0.80（争取 ~0.84） | 单一规则基线 |
> | 调度约束满足率 | 满足约束数 / 总约束数 | ≥ 0.85（CP-SAT 可更高） | 贪心/人工基线 |
> | 高风险航班延误减少 | 优化前后平均延误差 | ≥ 1.0 min/架次 | 优化前 |
> | 预测误差（辅助） | 回归 MAE / RMSE | 越低越好，须报告 | 均值/规则基线 |
>
> ---
>
> ## 2. 技术选型

**`c730fa5c_0004`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **核心指标（复现目标，须真实达成或如实报告，不许编数）**：
>
> | 指标 | 定义 | 目标 | 对照基线 |
> |---|---|---|---|
> | AUC | ROC-AUC | LightGBM > 逻辑回归；参考 ~0.86 | 逻辑回归 / 常数基线 |
> | KS | 好坏样本分离度 max(TPR−FPR) | 越高越好，须报告 | 逻辑回归 |
> | PR-AUC | 精确率-召回率曲线下面积（不平衡更看重） | LightGBM > 逻辑回归 | 逻辑回归 |
> | PSI（监控） | 分布稳定性指数 | 演示：注入漂移后能检出（PSI>0.25 触发告警） | — |
>
> > 参考量级（仅 sanity check）：Give Me Some Credit 上好模型 AUC 常见 0.86~0.87。**以实跑为准并如实写入 README。**
>
> ---
>
> ## 2. 技术选型

**`2da65ad3_0005`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> | 环节 | 采用方案 | 保留基线（对照） | 理由 |
> |---|---|---|---|
> | 预测模型 | **LightGBM**（回归 + 二分类双头） | RandomForest、朴素规则 | 表格数据 + 强交互效应上更准更快，原生类别特征 |
> | 超参优化 | **Optuna**（贝叶斯 TPE） | — | 比网格搜索样本效率高 |
> | 验证策略 | 时间序切分 + 按机尾 GroupKFold | — | 防数据泄露、检验延误传播泛化 |
> | 可解释性 | **SHAP** + 特征重要性 | — | 讲清驱动因素 |
> | 风险分级 | 分位数 5 级 | — | 自适应分布、样本均衡、可迁移 |
> | 调度优化 | **OR-Tools CP-SAT** 约束规划 | 贪心迭代局部搜索 | 硬约束可行性保证 + 逼近最优，把约束满足率做扎实 |
> | 交互界面 | **Streamlit** 面板 + 静态图 + 甘特动画 | — | 展示效果强 |

**`2da65ad3_0036`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> | 里程碑 | 内容 | 完成标准（DoD） |
> |---|---|---|
> | **M1 数据** | 合成生成器 + 加载器 + 测试 | `gen-data` 产出 2000 条，`test_synth` 绿，相关性成立 |
> | **M2 特征** | 7+3 特征 + 特征字典 + 测试 | 无泄露，`test_features` 绿，字典生成 |
> | **M3 预测** | LightGBM 双头 + Optuna + 基线 + SHAP | 优于基线，模型落盘，`test_predict` 绿 |
> | **M4 分级** | 分位数 5 级 + 捕获率 | 高风险 Recall ≥0.80 且优于规则，`test_risk` 绿 |
> | **M5 调度** | CP-SAT + 贪心基线 + 对照 | 硬约束满足率 ≥0.85，延误降 ≥1min/架次，`test_schedule` 绿 |

**`c730fa5c_0035`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 19. 落地路径与 GitHub 发布（M7）
>
> ### 19.1 落地
> 项目根目录 `D:\xiangmu\credit-risk-mlops`，全部代码/配置/README/测试/`outputs/reports` 置于此。
>
> ### 19.2 `.gitignore`（提交前必须就位）
> ```
> .venv/   __pycache__/   *.pyc
> data/raw/   data/processed/   *.csv
> mlruns/   outputs/models/   *.pkl
> outputs/figures/*.html
> .env   .idea/   .vscode/
> ```
> > 提交：源码、`config.yaml`、`Dockerfile`、`docker-compose.yml`、`README.md`、`tests/`、`outputs/reports/report.md`、少量小体积展示图（`*.png`）。数据、`mlruns/`、模型工件**不提交**。

**`c730fa5c_0037`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 20. 交付物清单
>
> - [ ] 纯 Python 代码库（第 4 节结构），位于 `D:\xiangmu\credit-risk-mlops`
> - [ ] `python -m creditrisk run-all` 一键跑通 数据→训练→评估
> - [ ] LightGBM 风控模型（+ 逻辑回归/常数基线）+ 不平衡处理 + SHAP + 评分卡
> - [ ] MLflow 实验追踪 + 模型注册
> - [ ] FastAPI 在线打分服务（含输入校验与 Swagger 文档）
> - [ ] Dockerfile（+ 可选 docker-compose），镜像可构建可运行
> - [ ] PSI 数据漂移监控（含注入漂移演示）
> - [ ] Streamlit 面板（概览 / 在线打分 / 漂移监控）
> - [ ] `pytest` 测试套件全绿（含 API 测试）
> - [ ] GitHub Actions CI（ruff + pytest）
> - [ ] README（架构图、指标表、运行说明、技术选型、面试讲法）
> - [ ] 项目已推送至用户 GitHub，仓库可访问


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 4. `cap_017`

**问题**：风控里的 KS 指标是什么意思？为什么风控行业要用它？

**参考答案**：

> KS 是好坏样本累积分布之差的最大值（max(TPR−FPR)），衡量模型把违约与不违约样本拉开的能力，是风控行业标准指标。

**Agent 答案**：

> 根据知识库内容，我来回答关于 KS 指标的问题。
>
> ## KS 指标是什么
>
> **KS（Kolmogorov-Smirnov）** 是衡量模型区分好坏样本能力的指标，具体定义为**好坏样本累积分布之差的最大值**。
>
> 在风控场景中，它衡量的是模型把"违约"和"不违约"两类客户拉开的能力。KS 值越大，说明模型对两类样本的区分度越好，排序能力越强。
>
> ## 为什么风控行业要用它
>
> 根据知识库内容，主要有以下几点原因：
>
> 1. **对不平衡数据稳健**：风控场景中正类（违约）通常只占很小比例（如约 6%），此时准确率会失真——即使全预测为"不违约"，准确率也有 94% 却毫无用处。而 KS 关注的是排序/分离能力，不受类别不平衡影响。
>
> 2. **风控行业标准**：KS 与评分卡一起是风控行业的评估标准，被广泛用于模型评估、阈值选择（如按 KS 最大点选阈值）和模型监控。
>
> 3. **关注排序能力而非绝对准确**：风控更关心能否把风险高低的人群正确排序、分层，KS 恰好反映这种区分/分离能力。
>
> 4. **配套使用**：在实际评估中，KS 通常与 **AUC、PR-AUC、混淆矩阵、校准曲线** 等一起使用，并在 MLflow 中记录追踪，用于模型对比和上线监控。
>
> 补充一点：与 KS 常搭配的还有 **PSI（群体稳定性指数）**，用于监控线上数据分布相对训练时的漂移，PSI>0.25 就该预警甚至重训——这属于模型上线后的持续监控环节。

<details><summary>检索到的片段（4 块）</summary>

**`c730fa5c_0005`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> | 环节 | 采用方案 | 保留基线/理由 |
> |---|---|---|
> | 主模型 | **LightGBM**（二分类，`scale_pos_weight` 处理不平衡） | 表格数据强、快、CPU 友好；基线用**逻辑回归** |
> | 调参 | **Optuna**（轻量，20~50 trial，目标 AUC） | 比网格搜索样本效率高 |
> | 可解释 | **SHAP** + 特征重要性 | 风控强需求：讲清"为什么拒绝" |
> | 评估 | AUC / **KS** / PR-AUC / 混淆矩阵 / 校准 + **评分卡映射** | KS/评分卡是风控行业标准 |
> | 实验追踪 | **MLflow**（本地 file-based，含 Model Registry） | 无需起服务器，稳；面试常问 |
> | 服务化 | **FastAPI** + Pydantic 校验 + Uvicorn | 主流、轻、自带文档页 |
> | 容器化 | **Docker**（+ 可选 docker-compose 同起 API 与 MLflow UI） | 一键复现环境 |

**`c730fa5c_0039`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **面试高频问，先备好答案**：
> - **不平衡数据为什么不看准确率？** 正类仅约 6%，全预测为"不违约"准确率也有 94% 却毫无用处；所以用 AUC、PR-AUC、KS 这些对不平衡稳健、且关注排序/分离能力的指标。
> - **KS 是什么？** 好坏样本累积分布之差的最大值，衡量模型把违约/不违约拉开的能力，是风控行业标准。
> - **PSI 是什么、为什么要监控漂移？** 群体稳定性指数，比较线上数据分布与训练时的偏移；分布漂移会让模型悄悄失效，PSI>0.25 就该预警甚至重训。
> - **为什么要把模型做成服务 + 容器？** 让模型能被真实系统调用、环境可一键复现，是"能用的模型"和"上线的模型"的区别。
> - **怎么保证训练和线上用的是同一套处理？** 预处理器随模型一起序列化、服务加载同一工件，避免训练-服务偏移（training-serving skew）。
> - **MLflow 解决什么？** 实验可追溯、可对比、模型可版本化注册，团队协作和回滚都靠它。
>
> ---

**`c730fa5c_0023`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ### 8.2 MLflow 追踪与注册
> - 每次训练用 MLflow 记录：超参、AUC/KS/PR-AUC、ROC/KS/校准图、SHAP 摘要图、模型工件。
> - 用 **Model Registry** 注册最优模型为 `credit-risk-model`，打 stage 标记（如 `Staging`/`Production`）。
> - 本地 `mlruns/` 追踪，`mlflow ui` 可查看对比。
>
> ### 8.3 评估（`evaluate.py`）
> - **AUC、KS、PR-AUC**、指定阈值下的混淆矩阵/精确率/召回率、**校准曲线**。
> - **SHAP**：全局重要性 + 摘要图；能讲清主要驱动因素（逾期历史、使用率等）。
> - **阈值选择**：给出按 KS 最大点或按业务代价的阈值建议。
> - **评分卡映射**：把违约概率映射为直观信用分（如 300~850），写进报告。
> - 落盘 `outputs/reports/metrics.json` 与 `report.md`，含基线对比；**自动写回 README 占位指标表**。

**`c730fa5c_0021`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> - **缺失填补**：`MonthlyIncome` 用中位数或分组中位数；`NumberOfDependents` 用众数/0；并加"是否缺失"指示位。
> - **异常处理**：逾期次数字段的哨兵值（96/98）截断或单独成桶；使用率、负债比按分位数（如 99%）截断。
> - **派生特征（少而有信号）**：总逾期次数、人均月收入（收入/(家属+1)）、月收入与负债比的交互、年龄分桶。
> - **（可选风控味）WOE 编码**：对分桶特征做 WOE 变换，附带 IV 值，为"评分卡"做铺垫（时间紧可跳过，README 注明）。
> - **接口**：`Preprocessor.fit(X_train)`；`transform(X)`；`save/load`（joblib 序列化到 `outputs/models/preprocessor.pkl`）。
> - 输出"特征说明表"到 `outputs/reports/feature_dict.md`。
>
> **验收**：无泄露；预处理器可序列化并复用；`transform` 对单条与批量都可用（服务层需要）。
>
> ---


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 5. `cap_020`

**问题**：信贷项目处理类别不平衡时，为什么优先用加权而不是首选 SMOTE？

**参考答案**：

> 规格书要求优先用 scale_pos_weight / class_weight 加权，SMOTE 只作对照并说明。加权不改变原始样本分布、实现简单且不引入合成样本带来的分布偏差。

**Agent 答案**：

> 知识库中明确给出了技术选型结论（优先加权、不首选 SMOTE），但未展开详细理由。我结合知识库内容与领域常识，给出严谨回答。
>
> ## 为什么信贷项目优先用加权而不是首选 SMOTE？
>
> 知识库（技术规格书 §8.1）明确要求：主模型 LightGBM 用 `scale_pos_weight` 或 `class_weight` 处理不平衡，**优先加权，不首选 SMOTE**；如做 SMOTE 仅作对照并说明。理由可从以下几个层面理解：
>
> **1. 加权不改变样本分布，更贴近真实业务**
> - 加权（`scale_pos_weight`/`class_weight`）只是在损失函数里给少数类样本更高权重，**不新增、不删除样本**，保留原始数据分布。
> - 信贷场景正类仅约 6~7%，真实线上违约率就是如此。加权让模型在真实分布上训练，预测概率更接近真实违约概率，便于后续做评分卡映射和阈值选择。
>
> **2. SMOTE 合成样本可能引入噪声与过拟合**
> - SMOTE 在少数类样本之间线性插值生成合成样本。信贷特征（如逾期次数、负债比）存在大量离群值和哨兵值，合成样本可能落在不真实的特征空间，引入噪声。
> - 合成样本会放大少数类，容易让模型对"人造样本"过拟合，而真实线上样本未必符合这种合成分布，导致泛化下降。
>
> **3. 数据泄露与验证风险**
> - SMOTE 若在划分训练/验证集**之前**做，会造成信息泄露（合成样本与验证样本重叠），使评估虚高。正确做法须在切分后仅对训练集做，流程更易出错。
> - 加权则无此风险，训练-验证划分天然干净。
>
> **4. 工程与可解释性**
> - 加权只需一个超参（`scale_pos_weight`），可直接纳入 Optuna 搜索空间调优，实现简单、稳定。
> - 风控强调可解释性（SHAP、评分卡），加权不改变样本，特征含义和 SHAP 解释更可信；SMOTE 引入的合成样本会干扰对真实样本重要性的解读。
>
> **5. 评估指标本身已对不平衡稳健**
> - 项目用 AUC、KS、PR-AUC 等关注排序/分离能力的指标，而非准确率。这些指标下，加权已能有效提升少数类识别，无需额外重采样。
>
> **结论**：加权更简单、无泄露风险、不引入合成噪声、保留真实分布且利于可解释性，因此作为首选；SMOTE 仅作为对照实验验证，不作为主方案。

<details><summary>检索到的片段（10 块）</summary>

**`c730fa5c_0018`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 6. 数据层规格（`data.py`）
>
> ### 6.1 数据集：Give Me Some Credit
> - Kaggle 竞赛数据，训练集 `cs-training.csv`，约 **15 万行**、文件仅几 MB（远小于 200MB）。
> - **标签**：`SeriousDlqin2yrs`（1 = 两年内发生 90 天以上逾期，正类；**约占 6~7%，天然不平衡**）。
> - **10 个特征**：授信循环额度使用率、年龄、30–59/60–89/90+ 天逾期次数、负债比、月收入、开放信贷额度数、房产贷款数、家属人数。
> - **已知数据问题（都要处理，也都是面试谈资）**：`MonthlyIncome`、`NumberOfDependents` 有缺失；逾期次数字段存在异常哨兵值（如 96/98）；使用率与负债比有极端离群值。

**`c730fa5c_0021`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> - **缺失填补**：`MonthlyIncome` 用中位数或分组中位数；`NumberOfDependents` 用众数/0；并加"是否缺失"指示位。
> - **异常处理**：逾期次数字段的哨兵值（96/98）截断或单独成桶；使用率、负债比按分位数（如 99%）截断。
> - **派生特征（少而有信号）**：总逾期次数、人均月收入（收入/(家属+1)）、月收入与负债比的交互、年龄分桶。
> - **（可选风控味）WOE 编码**：对分桶特征做 WOE 变换，附带 IV 值，为"评分卡"做铺垫（时间紧可跳过，README 注明）。
> - **接口**：`Preprocessor.fit(X_train)`；`transform(X)`；`save/load`（joblib 序列化到 `outputs/models/preprocessor.pkl`）。
> - 输出"特征说明表"到 `outputs/reports/feature_dict.md`。
>
> **验收**：无泄露；预处理器可序列化并复用；`transform` 对单条与批量都可用（服务层需要）。
>
> ---

**`c730fa5c_0004`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **核心指标（复现目标，须真实达成或如实报告，不许编数）**：
>
> | 指标 | 定义 | 目标 | 对照基线 |
> |---|---|---|---|
> | AUC | ROC-AUC | LightGBM > 逻辑回归；参考 ~0.86 | 逻辑回归 / 常数基线 |
> | KS | 好坏样本分离度 max(TPR−FPR) | 越高越好，须报告 | 逻辑回归 |
> | PR-AUC | 精确率-召回率曲线下面积（不平衡更看重） | LightGBM > 逻辑回归 | 逻辑回归 |
> | PSI（监控） | 分布稳定性指数 | 演示：注入漂移后能检出（PSI>0.25 触发告警） | — |
>
> > 参考量级（仅 sanity check）：Give Me Some Credit 上好模型 AUC 常见 0.86~0.87。**以实跑为准并如实写入 README。**
>
> ---
>
> ## 2. 技术选型

**`c730fa5c_0023`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ### 8.2 MLflow 追踪与注册
> - 每次训练用 MLflow 记录：超参、AUC/KS/PR-AUC、ROC/KS/校准图、SHAP 摘要图、模型工件。
> - 用 **Model Registry** 注册最优模型为 `credit-risk-model`，打 stage 标记（如 `Staging`/`Production`）。
> - 本地 `mlruns/` 追踪，`mlflow ui` 可查看对比。
>
> ### 8.3 评估（`evaluate.py`）
> - **AUC、KS、PR-AUC**、指定阈值下的混淆矩阵/精确率/召回率、**校准曲线**。
> - **SHAP**：全局重要性 + 摘要图；能讲清主要驱动因素（逾期历史、使用率等）。
> - **阈值选择**：给出按 KS 最大点或按业务代价的阈值建议。
> - **评分卡映射**：把违约概率映射为直观信用分（如 300~850），写进报告。
> - 落盘 `outputs/reports/metrics.json` 与 `report.md`，含基线对比；**自动写回 README 占位指标表**。

**`c730fa5c_0022`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **验收**：无泄露；预处理器可序列化并复用；`transform` 对单条与批量都可用（服务层需要）。
>
> ---
>
> ## 8. 建模层规格（`model.py` + `evaluate.py`）
>
> ### 8.1 训练
> - 主模型 **LightGBM**（`LGBMClassifier`），用 `scale_pos_weight` 或 `class_weight` 处理不平衡（**优先加权，不首选 SMOTE**；如做 SMOTE 仅作对照并说明）。
> - 基线 **逻辑回归**（标准化 + L2）与**常数/热门基线**。
> - **Optuna** 轻量调参（20~50 trial，目标验证集 AUC）：`num_leaves`、`max_depth`、`learning_rate`、`n_estimators`、`min_child_samples`、`subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda`、`scale_pos_weight`。搜索空间放 config。

**`2da65ad3_0021`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> dep_delay = max(0, base_delay)
> ```
> - 系数 `a*`、`sigma` 放入 config，取值使 `is_delayed15` 正类占比落在 **20%–35%**，各特征均有可辨识信号。
> - 生成器落盘 `data/processed/flights.parquet`。

**`2da65ad3_0005`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> | 环节 | 采用方案 | 保留基线（对照） | 理由 |
> |---|---|---|---|
> | 预测模型 | **LightGBM**（回归 + 二分类双头） | RandomForest、朴素规则 | 表格数据 + 强交互效应上更准更快，原生类别特征 |
> | 超参优化 | **Optuna**（贝叶斯 TPE） | — | 比网格搜索样本效率高 |
> | 验证策略 | 时间序切分 + 按机尾 GroupKFold | — | 防数据泄露、检验延误传播泛化 |
> | 可解释性 | **SHAP** + 特征重要性 | — | 讲清驱动因素 |
> | 风险分级 | 分位数 5 级 | — | 自适应分布、样本均衡、可迁移 |
> | 调度优化 | **OR-Tools CP-SAT** 约束规划 | 贪心迭代局部搜索 | 硬约束可行性保证 + 逼近最优，把约束满足率做扎实 |
> | 交互界面 | **Streamlit** 面板 + 静态图 + 甘特动画 | — | 展示效果强 |

**`c730fa5c_0025`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 9. 服务化规格（`serve.py` + `schemas.py`）
>
> - **FastAPI 应用**，启动时加载部署工件（`preprocessor.pkl` + 注册模型）。
> - **接口**：
>   - `GET /health`：健康检查。
>   - `POST /predict`：入参为单个申请人的原始字段（Pydantic 校验类型/范围），返回 `{default_probability, credit_score, risk_level, top_reasons}`（`top_reasons` 用 SHAP 给出主要影响因子）。
>   - `POST /predict/batch`：批量。
> - **输入校验**：用 `schemas.py` 的 Pydantic 模型，非法输入返回 422 并给出清晰错误。
> - 自带 Swagger 文档页（`/docs`）。
> - 启动：`uvicorn creditrisk.serve:app --reload`。

**`c730fa5c_0039`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **面试高频问，先备好答案**：
> - **不平衡数据为什么不看准确率？** 正类仅约 6%，全预测为"不违约"准确率也有 94% 却毫无用处；所以用 AUC、PR-AUC、KS 这些对不平衡稳健、且关注排序/分离能力的指标。
> - **KS 是什么？** 好坏样本累积分布之差的最大值，衡量模型把违约/不违约拉开的能力，是风控行业标准。
> - **PSI 是什么、为什么要监控漂移？** 群体稳定性指数，比较线上数据分布与训练时的偏移；分布漂移会让模型悄悄失效，PSI>0.25 就该预警甚至重训。
> - **为什么要把模型做成服务 + 容器？** 让模型能被真实系统调用、环境可一键复现，是"能用的模型"和"上线的模型"的区别。
> - **怎么保证训练和线上用的是同一套处理？** 预处理器随模型一起序列化、服务加载同一工件，避免训练-服务偏移（training-serving skew）。
> - **MLflow 解决什么？** 实验可追溯、可对比、模型可版本化注册，团队协作和回滚都靠它。
>
> ---

**`c730fa5c_0005`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> | 环节 | 采用方案 | 保留基线/理由 |
> |---|---|---|
> | 主模型 | **LightGBM**（二分类，`scale_pos_weight` 处理不平衡） | 表格数据强、快、CPU 友好；基线用**逻辑回归** |
> | 调参 | **Optuna**（轻量，20~50 trial，目标 AUC） | 比网格搜索样本效率高 |
> | 可解释 | **SHAP** + 特征重要性 | 风控强需求：讲清"为什么拒绝" |
> | 评估 | AUC / **KS** / PR-AUC / 混淆矩阵 / 校准 + **评分卡映射** | KS/评分卡是风控行业标准 |
> | 实验追踪 | **MLflow**（本地 file-based，含 Model Registry） | 无需起服务器，稳；面试常问 |
> | 服务化 | **FastAPI** + Pydantic 校验 + Uvicorn | 主流、轻、自带文档页 |
> | 容器化 | **Docker**（+ 可选 docker-compose 同起 API 与 MLflow UI） | 一键复现环境 |


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 6. `cap_022`

**问题**：信贷项目里，哪些东西明确不提交进 git？

**参考答案**：

> 数据（data/raw、data/processed、*.csv）、mlruns/、模型工件（outputs/models、*.pkl）、.env 以及编辑器目录都不提交；提交的是源码、config.yaml、Dockerfile、README、tests 和 report.md。

**Agent 答案**：

> 根据检索到的信贷项目技术规格书内容，明确不提交进 git 的内容如下：
>
> **`.gitignore` 中明确忽略（不提交）的内容：**
> - `.venv/`、`__pycache__/`、`*.pyc`
> - `data/raw/`、`data/processed/`、`*.csv`（数据）
> - `mlruns/`（MLflow 运行记录）
> - `outputs/models/`、`*.pkl`（模型工件）
> - `outputs/figures/*.html`
> - `.env`（环境变量/凭据）
> - `.idea/`、`.vscode/`（IDE 配置）
>
> **应提交的内容：** 源码、`config.yaml`、`Dockerfile`、`docker-compose.yml`、`README.md`、`tests/`、`outputs/reports/report.md`、少量小体积展示图（`*.png`）。
>
> **核心原则：** 数据、`mlruns/`、模型二进制工件、凭据（`.env`）均**不提交**。发布验收时也要求"未提交数据/模型二进制/凭据"。

<details><summary>检索到的片段（4 块）</summary>

**`c730fa5c_0036`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ### 19.3 发布前置（Claude Code 先检查，缺失则提示用户自行完成）
> - `git --version`；`gh auth status`。**未登录则让用户自行 `gh auth login`**，不写入任何凭据。
> - 发布前确认仓库**公开（作品集展示，推荐）或私有**。
>
> ### 19.4 发布步骤
> ```powershell
> cd D:\xiangmu\credit-risk-mlops
> git init
> git add .
> git commit -m "Initial commit: credit-risk model with end-to-end MLOps"
> gh repo create credit-risk-mlops --public --source . --remote origin --push
> ```
>
> ### 19.5 发布验收
> 仓库可访问，README 正常渲染；CI 通过；未提交数据/模型二进制/凭据。
>
> ---
>
> ## 20. 交付物清单

**`2da65ad3_0039`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ### 18.2 `.gitignore`（提交前必须就位）
> 至少忽略：
> ```
> .venv/            __pycache__/     *.pyc
> data/raw/         data/processed/  *.parquet  *.csv
> outputs/models/   outputs/figures/*.mp4
> .env              .idea/  .vscode/
> ```
> > 保留并提交：源码、`config.yaml`、`README.md`、`tests/`、`outputs/reports/report.md`、少量小体积展示图（`outputs/figures/*.png`）。大动画（gif/mp4）建议不提交，或改用 Git LFS。

**`c730fa5c_0035`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 19. 落地路径与 GitHub 发布（M7）
>
> ### 19.1 落地
> 项目根目录 `D:\xiangmu\credit-risk-mlops`，全部代码/配置/README/测试/`outputs/reports` 置于此。
>
> ### 19.2 `.gitignore`（提交前必须就位）
> ```
> .venv/   __pycache__/   *.pyc
> data/raw/   data/processed/   *.csv
> mlruns/   outputs/models/   *.pkl
> outputs/figures/*.html
> .env   .idea/   .vscode/
> ```
> > 提交：源码、`config.yaml`、`Dockerfile`、`docker-compose.yml`、`README.md`、`tests/`、`outputs/reports/report.md`、少量小体积展示图（`*.png`）。数据、`mlruns/`、模型工件**不提交**。

**`2da65ad3_0040`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ### 18.3 发布前置条件（Claude Code 先检查，缺失则提示用户自行完成）
> - 已安装 Git：`git --version`。
> - 已安装并登录 GitHub CLI：`gh auth status`。
>   - **若未登录，请让用户自己执行 `gh auth login` 完成认证**；不要在代码或命令里写入任何密码 / token / 明文凭据。
> - 用户可选择仓库**公开（作品集展示，推荐）或私有**——发布前向用户确认一次可见性。
>
> ### 18.4 发布步骤（在项目根目录执行）
> ```powershell
> cd D:\xiangmu\flight-delay-scheduling
> git init
> git add .
> git commit -m "Initial commit: flight delay prediction & scheduling optimization"


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 7. `cap_026`

**问题**：航班调度优化的四条硬约束分别是什么？

**参考答案**：

> 时隙容量（任一时间窗内起飞架次不超过跑道容量 C）、周转约束（同一 tail_id 相邻航段后段起飞不早于前段到达加最小周转）、时窗约束（|o_f| ≤ K）、宵禁（禁飞时窗内不得起飞）。

**Agent 答案**：

> 根据知识库内容，航班调度优化的四条硬约束如下：
>
> 1. **时隙容量**：任一时间窗内起飞架次 ≤ 跑道容量 `C`。
> 2. **周转约束**：同一 `tail_id` 相邻航段，后段起飞 ≥ 前段到达 + `min_turnaround`（前段到达随其偏移联动）。
> 3. **时窗约束**：`|o_f| ≤ K`（起飞时隙偏移量限制在允许范围内）。
> 4. **宵禁**：禁飞时窗内不得起飞。

<details><summary>检索到的片段（4 块）</summary>

**`2da65ad3_0029`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ## 10. 调度优化模块规格（`schedule.py`）
>
> ### 10.1 问题建模（地面延误 / 时隙分配）
>
> - **决策变量**：为每个（高风险或全部）航班选一个**起飞时隙偏移** `o_f ∈ {-K,...,+K}`（离散到 5 分钟，`K` 如 30min）。
> - **延误查表**：对每航班 × 每候选偏移，**预估该偏移下的延误** `D[f][o]`（用已训练模型或拥堵模型重算：偏移改变所在时窗拥堵度，从而改变预测延误；传播项沿机尾链条向后传递）。得到查表后优化变为清晰的**赋值问题**。
> - **目标**：`min Σ_f w_f · D[f][o_f]`，高风险航班 `w_f` 更大（权重放 config）。
> - **硬约束**：
>   1. **时隙容量**：任一时间窗内起飞架次 ≤ 跑道容量 `C`。
>   2. **周转约束**：同一 `tail_id` 相邻航段，后段起飞 ≥ 前段到达 + `min_turnaround`（前段到达随其偏移联动）。
>   3. **时窗约束**：`|o_f| ≤ K`。
>   4. **宵禁**：禁飞时窗内不得起飞。

**`2da65ad3_0031`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **验收**：CP-SAT 约束满足率 ≥ 0.85（并展示相对贪心的提升）；高风险航班平均延误较优化前下降 ≥ 1.0 min/架次。
>
> ---
>
> ## 11. 评估模块规格（`evaluate.py`）
>
> - 汇总落盘 `outputs/reports/metrics.json` 与 `report.md`，含：
>   - 回归 MAE/RMSE（LightGBM vs RF vs 均值基线）。
>   - 高风险 Recall/Precision/F1/PR-AUC（vs 规则基线）。
>   - 调度约束满足率、高风险延误减少（CP-SAT vs 贪心 vs 优化前）。
> - 每个指标须有**明确定义 + 基线对比**。
> - `summarize(...) -> dict`，并把最终指标表**自动写回 README** 的占位表格。
>
> ---
>
> ## 12. 可视化与交互界面
>
> ### 12.1 静态图（`viz.py`）
> 延误分布、特征重要性、SHAP 摘要、混淆矩阵/PR 曲线、优化前后延误对比条形图。

**`2da65ad3_0003`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 1. 项目目标与范围
>
> 构建一个**"预测—调度"闭环系统**，解决多因素非线性航班延误问题：
>
> 1. **预测环**：基于航班静态/动态/气象特征，预测每个航班的起飞延误。
> 2. **分级环**：将预测结果映射为 5 级风险等级（分位数策略）。
> 3. **调度环**：以风险为输入，通过约束优化调整高风险航班的起飞时隙/缓冲，在满足运行约束下最小化加权总延误。
> 4. **评估 + 可视化**：复现三大核心指标并出图；提供交互面板。
>
> **核心指标（复现目标，须真实达成或如实报告）**：

**`2da65ad3_0004`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **核心指标（复现目标，须真实达成或如实报告）**：
>
> | 指标 | 定义 | 目标 | 对照基线 |
> |---|---|---|---|
> | 高风险捕获率 | 高风险类 Recall（同时报 Precision/F1） | ≥ 0.80（争取 ~0.84） | 单一规则基线 |
> | 调度约束满足率 | 满足约束数 / 总约束数 | ≥ 0.85（CP-SAT 可更高） | 贪心/人工基线 |
> | 高风险航班延误减少 | 优化前后平均延误差 | ≥ 1.0 min/架次 | 优化前 |
> | 预测误差（辅助） | 回归 MAE / RMSE | 越低越好，须报告 | 均值/规则基线 |
>
> ---
>
> ## 2. 技术选型


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 8. `cap_032`

**问题**：信贷项目和航班项目都用 LightGBM 做主模型，两者各自的对照基线模型分别是什么？

**参考答案**：

> 信贷项目的基线是逻辑回归（标准化 + L2）以及常数/热门基线；航班项目的基线是 RandomForest（回归+分类）和朴素规则基线。

**Agent 答案**：

> 根据知识库检索结果，两个项目各自的对照基线模型如下：
>
> **信贷项目**（信贷风控端到端ML系统）
> - 对照基线：**逻辑回归** 与 **常数基线**
> - 验收标准：LightGBM 的 AUC/KS/PR-AUC 需优于逻辑回归与常数基线
>
> **航班项目**（航班延误预测与智能调度优化系统）
> - 对照基线：**RandomForest（随机森林）**（回归+分类）与 **朴素规则**（`weather_severity>阈值 或 prev_leg_delay>阈值 → 高风险`）
> - 验收标准：LightGBM 在测试集上需优于 RF 与规则基线
>
> 总结：信贷项目用**逻辑回归 + 常数基线**，航班项目用**随机森林 + 朴素规则基线**。

<details><summary>检索到的片段（5 块）</summary>

**`c730fa5c_0024`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **验收**：LightGBM 的 AUC/KS/PR-AUC **优于逻辑回归与常数基线**；SHAP 结果合理；MLflow 中可见完整实验记录。
>
> ---
>
> ## 9. 服务化规格（`serve.py` + `schemas.py`）

**`2da65ad3_0025`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **任务**：双头预测 —— 回归 `dep_delay` + 二分类 `is_delayed15`。
>
> **模型**：
> - 主模型：**LightGBM**（`LGBMRegressor` + `LGBMClassifier`）。
> - 基线：**RandomForest**（回归+分类）与**朴素规则**（`weather_severity>阈值 或 prev_leg_delay>阈值 → 高风险`）。

**`c730fa5c_0004`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **核心指标（复现目标，须真实达成或如实报告，不许编数）**：
>
> | 指标 | 定义 | 目标 | 对照基线 |
> |---|---|---|---|
> | AUC | ROC-AUC | LightGBM > 逻辑回归；参考 ~0.86 | 逻辑回归 / 常数基线 |
> | KS | 好坏样本分离度 max(TPR−FPR) | 越高越好，须报告 | 逻辑回归 |
> | PR-AUC | 精确率-召回率曲线下面积（不平衡更看重） | LightGBM > 逻辑回归 | 逻辑回归 |
> | PSI（监控） | 分布稳定性指数 | 演示：注入漂移后能检出（PSI>0.25 触发告警） | — |
>
> > 参考量级（仅 sanity check）：Give Me Some Credit 上好模型 AUC 常见 0.86~0.87。**以实跑为准并如实写入 README。**
>
> ---
>
> ## 2. 技术选型

**`2da65ad3_0027`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **输出**：
> - 模型序列化至 `outputs/models/`。
> - 回归 MAE/RMSE；分类 Precision/Recall/F1/PR-AUC/混淆矩阵。
> - **特征重要性 + SHAP** 摘要图（`outputs/figures/`）。
> - 接口：`train(X, y_reg, y_clf, cfg) -> models`；`predict(models, X) -> {"delay":..., "risk_proba":...}`。
>
> **验收**：LightGBM 在测试集**优于 RF 与规则基线**；SHAP 能识别天气/传播/拥堵为主要驱动。
>
> ---
>
> ## 9. 风险分级规格（`risk.py`）

**`2da65ad3_0024`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **3 个交互特征**：
> 1. `weather_severity × is_peak`
> 2. `airport_congestion × is_peak`
> 3. `prev_leg_delay × turnaround_slack`（`turnaround_slack = 计划周转 − min_turnaround`）
>
> **要求**：
> - 类别特征交给 LightGBM 原生处理；给 RF 基线则 One-Hot。
> - 输出"特征字典"（`outputs/reports/feature_dict.md`），逐项说明定义与动机。
> - `build_features(df, fit_stats=None) -> (X, y_reg, y_clf, stats)`，`stats` 承载训练集统计量供验证/测试集无泄露复用。
>
> ---
>
> ## 8. 预测模块规格（`predict.py`）
>
> **任务**：双头预测 —— 回归 `dep_delay` + 二分类 `is_delayed15`。


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 9. `cap_035`

**问题**：信贷风控项目的规格书里，要求用 Kubernetes 来编排服务吗？

**参考答案**：

> 不要求，而且明确禁止：规格书的范围纪律里写了不要额外堆工具，不引入 K8s、Airflow、云服务等。容器化只做 Dockerfile 和可选的 docker-compose。

**Agent 答案**：

> 根据检索到的规格书内容，我可以回答这个问题。
>
> **不需要。** 信贷风控项目的技术规格书中**没有要求使用 Kubernetes 来编排服务**。
>
> 规格书在服务化与容器化环节明确采用的方案是：
>
> - **服务化**：FastAPI + Pydantic 校验 + Uvicorn
> - **容器化**：Docker（+ 可选的 docker-compose，用于同时拉起 API 与 MLflow UI）
>
> 规格书第 10 节「容器化规格」只要求 `Dockerfile` 能通过 `docker build`，并可选提供 `docker-compose.yml`，全程未提及 Kubernetes 或任何编排集群方案。整个 MLOps 链路是本地落地（`D:\xiangmu\credit-risk-mlops`），采用轻量、本地可复现的技术栈，并未引入 K8s。

<details><summary>检索到的片段（8 块）</summary>

**`c730fa5c_0005`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> | 环节 | 采用方案 | 保留基线/理由 |
> |---|---|---|
> | 主模型 | **LightGBM**（二分类，`scale_pos_weight` 处理不平衡） | 表格数据强、快、CPU 友好；基线用**逻辑回归** |
> | 调参 | **Optuna**（轻量，20~50 trial，目标 AUC） | 比网格搜索样本效率高 |
> | 可解释 | **SHAP** + 特征重要性 | 风控强需求：讲清"为什么拒绝" |
> | 评估 | AUC / **KS** / PR-AUC / 混淆矩阵 / 校准 + **评分卡映射** | KS/评分卡是风控行业标准 |
> | 实验追踪 | **MLflow**（本地 file-based，含 Model Registry） | 无需起服务器，稳；面试常问 |
> | 服务化 | **FastAPI** + Pydantic 校验 + Uvicorn | 主流、轻、自带文档页 |
> | 容器化 | **Docker**（+ 可选 docker-compose 同起 API 与 MLflow UI） | 一键复现环境 |

**`c730fa5c_0000`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> # 信贷违约风控 · 端到端机器学习系统（建模 + MLOps）· 技术规格书
>
> > **本文档面向 Claude Code。** 请把它当作一份完整的项目实施规格（PRD + 技术设计），用 **Python** 自主完成端到端开发，最终落地到本地路径 **`D:\xiangmu\credit-risk-mlops`**，并**发布到用户的 GitHub**。目标是交付一个可运行、可测试、可复现、**面试拿得出手**的作品：一个信贷违约风控模型，外加**从建模到上线的完整 MLOps 链路**。
>
> ---
>
> ## 0. 给 Claude Code 的执行说明（先读这一节）

**`c730fa5c_0010`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 4. 项目结构
>
> **根目录：`D:\xiangmu\credit-risk-mlops`**

**`c730fa5c_0023`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ### 8.2 MLflow 追踪与注册
> - 每次训练用 MLflow 记录：超参、AUC/KS/PR-AUC、ROC/KS/校准图、SHAP 摘要图、模型工件。
> - 用 **Model Registry** 注册最优模型为 `credit-risk-model`，打 stage 标记（如 `Staging`/`Production`）。
> - 本地 `mlruns/` 追踪，`mlflow ui` 可查看对比。
>
> ### 8.3 评估（`evaluate.py`）
> - **AUC、KS、PR-AUC**、指定阈值下的混淆矩阵/精确率/召回率、**校准曲线**。
> - **SHAP**：全局重要性 + 摘要图；能讲清主要驱动因素（逾期历史、使用率等）。
> - **阈值选择**：给出按 KS 最大点或按业务代价的阈值建议。
> - **评分卡映射**：把违约概率映射为直观信用分（如 300~850），写进报告。
> - 落盘 `outputs/reports/metrics.json` 与 `report.md`，含基线对比；**自动写回 README 占位指标表**。

**`c730fa5c_0024`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **验收**：LightGBM 的 AUC/KS/PR-AUC **优于逻辑回归与常数基线**；SHAP 结果合理；MLflow 中可见完整实验记录。
>
> ---
>
> ## 9. 服务化规格（`serve.py` + `schemas.py`）

**`2da65ad3_0000`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> # 航班延误预测与智能调度优化系统 · 技术规格书
>
> > **本文档面向 Claude Code。** 请把它当作一份完整的项目实施规格（PRD + 技术设计），用 **Python** 自主完成端到端开发，最终把项目落地到本地路径 **`D:\xiangmu`**，并**发布到用户的 GitHub**。目标是交付一个可运行、可测试、可复现、作品集级别的项目。
>
> ---
>
> ## 0. 给 Claude Code 的执行说明（先读这一节）

**`c730fa5c_0026`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> **验收**：合法请求返回结构正确的打分与解释；非法请求被 Pydantic 拦截；`/docs` 可访问。
>
> ---
>
> ## 10. 容器化规格（`Dockerfile`）
>
> - 基于 `python:3.11-slim`，装依赖、拷贝代码与部署工件，`EXPOSE` 端口，`CMD` 启 Uvicorn。
> - `.dockerignore` 排除 `.venv/`、`mlruns/`、`data/`、`__pycache__/` 等。
> - **（可选）`docker-compose.yml`**：同时起 API 服务与 MLflow UI，一条命令拉起。
> - **验收**：`docker build -t credit-risk-api .` 成功；`docker run -p 8000:8000 credit-risk-api` 后可访问 `/health` 与 `/docs`。
>
> ---
>
> ## 11. 漂移监控规格（`monitor.py`）

**`c730fa5c_0017`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> - 依赖在 `pyproject.toml` **固定主版本**。
> - **Docker**：安装 Docker Desktop（用户自备）；Claude Code 只需保证 `Dockerfile` 正确、`docker build` 能通过。


</details>

**我的评分**： `___`   （0 / 1 / 2）

---

### 10. `cap_036`  ⚠ `hallucination_bait`

**问题**：航班延误项目的规格书里，指定用 Apache Kafka 做实时数据流接入吗？

**参考答案**：

> 没有。规格书的数据层主数据源是内置的合成数据生成器，另有可选的公开数据 CSV 加载器，全程可离线运行，完全没有提到 Kafka 或任何流式组件。正确行为是说明知识库中未提及，而不是编造。

**Agent 答案**：

> 我已经检索了规格书中的相关内容。让我确认一下技术选型章节（第2节）的具体内容，看看是否明确提到 Kafka。
>
> 从检索到的内容来看，规格书的数据接入方式非常明确：
>
> **规格书并未指定使用 Apache Kafka 做实时数据流接入。**
>
> 根据规格书内容：
>
> 1. **数据接入方式**（第6节数据层规格）：
>    - **主数据源**：内置的**合成数据生成器**（`data/synth.py`），生成约 2000 条航班记录，零外部依赖即可端到端运行。
>    - **可选数据源**：公开数据加载器（`data/loader.py`），把 BTS/Kaggle 等公开航班数据适配到统一 Schema。
>
> 2. **架构数据流**（第3节）：数据流是"合成/真实航班数据 → data（清洗+结构化）→ features → predict → risk → schedule → evaluate"，是**批处理式**的离线流程，并非实时流式接入。
>
> 3. **技术栈**：规格书强调"**不要过度设计**"、"先打通闭环"，技术选型围绕 Python、LightGBM、CP-SAT、Streamlit 等，未提及 Kafka 或任何消息队列/实时流组件。
>
> 因此，**规格书没有指定 Apache Kafka 做实时数据流接入**。该项目的数据接入是离线批处理式的（合成数据生成器 + 可选公开数据加载器），不涉及实时流式架构。

<details><summary>检索到的片段（16 块）</summary>

**`2da65ad3_0018`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 6. 数据层规格
>
> ### 6.1 合成数据生成器（`data/synth.py`，主数据源）
>
> 生成 **≈2000 条航班记录**，并**故意注入因果结构**，使特征对标签真实可预测、三大指标可复现。
>
> **实体设定**（可在 config 调整）：
> - 机场 8–12 个；承运人 5–8 家；机型 4–6 种。
> - 飞机 ≈120 架，每架当天飞**多段航线（2–4 段）**，用同一 `tail_id` 串接 → 支持**延误传播**。
> - 每机场每小时一个**天气状态**（能见度、风速、降水、雷暴标志），时空平滑。
>
> **字段 Schema**：

**`2da65ad3_0000`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> # 航班延误预测与智能调度优化系统 · 技术规格书
>
> > **本文档面向 Claude Code。** 请把它当作一份完整的项目实施规格（PRD + 技术设计），用 **Python** 自主完成端到端开发，最终把项目落地到本地路径 **`D:\xiangmu`**，并**发布到用户的 GitHub**。目标是交付一个可运行、可测试、可复现、作品集级别的项目。
>
> ---
>
> ## 0. 给 Claude Code 的执行说明（先读这一节）

**`2da65ad3_0003`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 1. 项目目标与范围
>
> 构建一个**"预测—调度"闭环系统**，解决多因素非线性航班延误问题：
>
> 1. **预测环**：基于航班静态/动态/气象特征，预测每个航班的起飞延误。
> 2. **分级环**：将预测结果映射为 5 级风险等级（分位数策略）。
> 3. **调度环**：以风险为输入，通过约束优化调整高风险航班的起飞时隙/缓冲，在满足运行约束下最小化加权总延误。
> 4. **评估 + 可视化**：复现三大核心指标并出图；提供交互面板。
>
> **核心指标（复现目标，须真实达成或如实报告）**：

**`2da65ad3_0022`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **验收**：Schema 完整；正类比例在区间；`prev_leg_delay`、`weather_severity`、`congestion` 与 `dep_delay` 正相关显著；固定 seed 完全复现。
>
> ### 6.2 公开数据加载器（`data/loader.py`，可选）
>
> - `load_public(csv_path) -> DataFrame`，把常见公开航班数据（如 BTS/Kaggle "Flight Delays"）适配到 6.1 统一 Schema。
> - 缺失字段用机型默认值填充；无 `tail_id` 时按 (aircraft, 时间序) 近似串接。
> - `data/raw/` 无文件则跳过，默认走合成数据。
>
> ---
>
> ## 7. 特征工程规格（`features.py`）
>
> 输入统一 Schema 的 DataFrame，输出模型可用特征矩阵。**严禁数据泄露**。

**`2da65ad3_0004`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> **核心指标（复现目标，须真实达成或如实报告）**：
>
> | 指标 | 定义 | 目标 | 对照基线 |
> |---|---|---|---|
> | 高风险捕获率 | 高风险类 Recall（同时报 Precision/F1） | ≥ 0.80（争取 ~0.84） | 单一规则基线 |
> | 调度约束满足率 | 满足约束数 / 总约束数 | ≥ 0.85（CP-SAT 可更高） | 贪心/人工基线 |
> | 高风险航班延误减少 | 优化前后平均延误差 | ≥ 1.0 min/架次 | 优化前 |
> | 预测误差（辅助） | 回归 MAE / RMSE | 越低越好，须报告 | 均值/规则基线 |
>
> ---
>
> ## 2. 技术选型

**`2da65ad3_0007`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> > 设计原则：预测器与调度器**解耦**，通过"风险等级 + 预测延误"接口连接，任一环可独立替换。
>
> ---
>
> ## 3. 系统架构与数据流

**`2da65ad3_0002`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> - **构建顺序**：严格按第 17 节里程碑（M1→M7）推进；**每完成一个里程碑先跑通对应测试再继续**。任何时候项目都应处于"能跑通全流程"的状态。
> - **完成定义（DoD）**：`python -m flightopt run-all` 一键跑通数据生成→训练→分级→调度→评估→出图；`pytest` 全绿；README 指标表被真实结果填充；Streamlit 面板可启动；项目已推送到 GitHub。
> - **数据前提**：原始真实数据不可得，**主数据源为内置的合成数据生成器**（第 6 节），零外部依赖即可端到端运行；另实现可选的公开数据加载器供后续替换真实数据。
> - **可复现**：全局随机种子集中在 `config.yaml`，所有随机过程读取它。
> - **不要过度设计**：先打通闭环，再逐模块优化。

**`c730fa5c_0007`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> > 设计原则：**建模与服务解耦**。训练产出一个"注册模型 + 预处理器"工件，服务层只加载工件对外打分，任一层可独立替换。
>
> ---
>
> ## 3. 系统架构与数据流

**`2da65ad3_0042`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ---
>
> ## 19. 交付物清单
>
> - [ ] 纯 Python 代码库（第 4 节结构），位于 `D:\xiangmu\flight-delay-scheduling`
> - [ ] `python -m flightopt run-all` 一键跑通全流程
> - [ ] 合成数据生成器（+ 可选公开数据加载器）
> - [ ] LightGBM 预测模型（+ RF/规则基线对照）+ SHAP 解释
> - [ ] 分位数 5 级风险分级
> - [ ] CP-SAT 调度优化（+ 贪心基线对照）
> - [ ] `metrics.json` + `report.md`（含全部指标与基线对比）
> - [ ] 静态图 + 甘特动画 + Streamlit 面板
> - [ ] `pytest` 测试套件全绿
> - [ ] README（架构图、结果指标表、运行说明、技术选型理由）
> - [ ] （可选）GitHub Actions CI
> - [ ] 项目已推送至用户 GitHub，仓库可访问
>
> ---

**`c730fa5c_0014`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> │   ├── serve.py             # FastAPI app
> │   └── cli.py               # 命令行入口（typer）
> ├── app/streamlit_app.py     # 监控 + 演示面板
> └── tests/
>     ├── test_data.py
>     ├── test_features.py
>     ├── test_model.py
>     ├── test_evaluate.py
>     ├── test_monitor.py
>     └── test_api.py
> ```

**`c730fa5c_0009`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> │
>       ▼
> [registry] 最优模型 + 预处理器 打包为部署工件
>       │
>       ├───────────────► [serve]  FastAPI /predict → Docker 镜像
>       │
>       └───────────────► [monitor] PSI 漂移检测(注入漂移演示) + Streamlit 面板
> ```

**`2da65ad3_0011`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ```
> flight-delay-scheduling/
> ├── pyproject.toml            # 依赖与打包（含 console_scripts: flightopt）
> ├── config.yaml               # 全局配置（seed、路径、超参范围、约束参数）
> ├── README.md                 # 架构图、结果指标表、运行说明、技术选型理由
> ├── .gitignore                # 忽略 venv/缓存/数据/模型等（见第 18 节）
> ├── .github/workflows/ci.yml  # GitHub Actions：lint + test
> ├── data/
> │   ├── raw/                  # 真实/公开数据（用户后续放入；被 gitignore）
> │   └── processed/            # 生成的中间产物（被 gitignore）
> ├── outputs/

**`2da65ad3_0012`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> │   └── processed/            # 生成的中间产物（被 gitignore）
> ├── outputs/
> │   ├── figures/             # 静态图、甘特动画
> │   ├── models/              # 序列化模型（被 gitignore）
> │   └── reports/             # metrics.json、report.md、feature_dict.md
> ├── src/flightopt/
> │   ├── __init__.py
> │   ├── config.py            # 加载 config.yaml → 强类型配置对象
> │   ├── paths.py             # 用 pathlib 统一管理路径（跨平台）
> │   ├── data/
> │   │   ├── synth.py         # 合成数据生成器（主数据源）
> │   │   └── loader.py        # 公开数据加载/适配器（可选）

**`c730fa5c_0040`**  ·  信贷风控端到端ML系统_技术规格书_for_ClaudeCode.md

> ---
>
> **备注给用户**：把本文件整份交给 Claude Code 即可。需要 Kaggle `kaggle.json` 才能自动下载数据，没有就手动把 `cs-training.csv` 放进 `data/raw/`。Docker 需自行安装 Docker Desktop。跑完把 README 里的指标占位符换成真实结果——面试问起来，第 21 节那些问题你要能自己讲清楚，这才是这个项目真正"拿得出手"的地方。

**`2da65ad3_0008`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ```
> 合成/真实航班数据
>       │
>       ▼
> [data]     清洗 + 结构化（含机尾串接，支持延误传播）
>       │
>       ▼
> [features] 7 核心特征 + 3 交互特征（防泄露）
>       │
>       ▼
> [predict]  LightGBM 回归(延误分钟) + 分类(P[延误>15min])  ─┐  基线: RF / 规则
>       │                                                    │
>       ▼                                                    ▼
> [risk]     分位数 5 级分级（L1..L5，L4/L5=高风险）      特征重要性/SHAP
>       │
>       ▼
> [schedule] CP-SAT 约束优化（时隙偏移/缓冲）  ── 基线: 贪心
>       │
>       ▼
> [evaluate] 三大指标 + 基线对比 + 回归误差
>       │
>       ▼

**`2da65ad3_0029`**  ·  航班延误预测与智能调度优化系统_技术规格书_for_ClaudeCode.md

> ## 10. 调度优化模块规格（`schedule.py`）
>
> ### 10.1 问题建模（地面延误 / 时隙分配）
>
> - **决策变量**：为每个（高风险或全部）航班选一个**起飞时隙偏移** `o_f ∈ {-K,...,+K}`（离散到 5 分钟，`K` 如 30min）。
> - **延误查表**：对每航班 × 每候选偏移，**预估该偏移下的延误** `D[f][o]`（用已训练模型或拥堵模型重算：偏移改变所在时窗拥堵度，从而改变预测延误；传播项沿机尾链条向后传递）。得到查表后优化变为清晰的**赋值问题**。
> - **目标**：`min Σ_f w_f · D[f][o_f]`，高风险航班 `w_f` 更大（权重放 config）。
> - **硬约束**：
>   1. **时隙容量**：任一时间窗内起飞架次 ≤ 跑道容量 `C`。
>   2. **周转约束**：同一 `tail_id` 相邻航段，后段起飞 ≥ 前段到达 + `min_turnaround`（前段到达随其偏移联动）。
>   3. **时窗约束**：`|o_f| ≤ K`。
>   4. **宵禁**：禁飞时窗内不得起飞。


</details>

**我的评分**： `___`   （0 / 1 / 2）

---
