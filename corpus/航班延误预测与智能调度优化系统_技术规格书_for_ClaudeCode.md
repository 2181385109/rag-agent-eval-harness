# 航班延误预测与智能调度优化系统 · 技术规格书

> **本文档面向 Claude Code。** 请把它当作一份完整的项目实施规格（PRD + 技术设计），用 **Python** 自主完成端到端开发，最终把项目落地到本地路径 **`D:\xiangmu`**，并**发布到用户的 GitHub**。目标是交付一个可运行、可测试、可复现、作品集级别的项目。

---

## 0. 给 Claude Code 的执行说明（先读这一节）

- **语言**：全程使用 **Python**（3.11）。所有文件路径在代码中一律用 `pathlib.Path` 处理，保证在 **Windows** 上正常工作。
- **落地路径**：项目最终根目录为 **`D:\xiangmu\flight-delay-scheduling`**。若 `D:\xiangmu` 不存在则创建；所有代码、配置、产物都放在该目录下。
- **发布 GitHub**：全部开发与自测完成后，按第 18 节把项目推送到用户 GitHub。
- **自主度**：可自主做合理工程决策，不必逐字征询确认；文档未指定处选成熟简洁的默认方案，并在 README 记录取舍。
- **构建顺序**：严格按第 17 节里程碑（M1→M7）推进；**每完成一个里程碑先跑通对应测试再继续**。任何时候项目都应处于"能跑通全流程"的状态。
- **完成定义（DoD）**：`python -m flightopt run-all` 一键跑通数据生成→训练→分级→调度→评估→出图；`pytest` 全绿；README 指标表被真实结果填充；Streamlit 面板可启动；项目已推送到 GitHub。
- **数据前提**：原始真实数据不可得，**主数据源为内置的合成数据生成器**（第 6 节），零外部依赖即可端到端运行；另实现可选的公开数据加载器供后续替换真实数据。
- **可复现**：全局随机种子集中在 `config.yaml`，所有随机过程读取它。
- **不要过度设计**：先打通闭环，再逐模块优化。

---

## 1. 项目目标与范围

构建一个**"预测—调度"闭环系统**，解决多因素非线性航班延误问题：

1. **预测环**：基于航班静态/动态/气象特征，预测每个航班的起飞延误。
2. **分级环**：将预测结果映射为 5 级风险等级（分位数策略）。
3. **调度环**：以风险为输入，通过约束优化调整高风险航班的起飞时隙/缓冲，在满足运行约束下最小化加权总延误。
4. **评估 + 可视化**：复现三大核心指标并出图；提供交互面板。

**核心指标（复现目标，须真实达成或如实报告）**：

| 指标 | 定义 | 目标 | 对照基线 |
|---|---|---|---|
| 高风险捕获率 | 高风险类 Recall（同时报 Precision/F1） | ≥ 0.80（争取 ~0.84） | 单一规则基线 |
| 调度约束满足率 | 满足约束数 / 总约束数 | ≥ 0.85（CP-SAT 可更高） | 贪心/人工基线 |
| 高风险航班延误减少 | 优化前后平均延误差 | ≥ 1.0 min/架次 | 优化前 |
| 预测误差（辅助） | 回归 MAE / RMSE | 越低越好，须报告 | 均值/规则基线 |

---

## 2. 技术选型

| 环节 | 采用方案 | 保留基线（对照） | 理由 |
|---|---|---|---|
| 预测模型 | **LightGBM**（回归 + 二分类双头） | RandomForest、朴素规则 | 表格数据 + 强交互效应上更准更快，原生类别特征 |
| 超参优化 | **Optuna**（贝叶斯 TPE） | — | 比网格搜索样本效率高 |
| 验证策略 | 时间序切分 + 按机尾 GroupKFold | — | 防数据泄露、检验延误传播泛化 |
| 可解释性 | **SHAP** + 特征重要性 | — | 讲清驱动因素 |
| 风险分级 | 分位数 5 级 | — | 自适应分布、样本均衡、可迁移 |
| 调度优化 | **OR-Tools CP-SAT** 约束规划 | 贪心迭代局部搜索 | 硬约束可行性保证 + 逼近最优，把约束满足率做扎实 |
| 交互界面 | **Streamlit** 面板 + 静态图 + 甘特动画 | — | 展示效果强 |
| 工程化 | pytest + 配置化 + CLI + GitHub Actions CI | — | 可测试、可复现 |

> 设计原则：预测器与调度器**解耦**，通过"风险等级 + 预测延误"接口连接，任一环可独立替换。

---

## 3. 系统架构与数据流

```
合成/真实航班数据
      │
      ▼
[data]     清洗 + 结构化（含机尾串接，支持延误传播）
      │
      ▼
[features] 7 核心特征 + 3 交互特征（防泄露）
      │
      ▼
[predict]  LightGBM 回归(延误分钟) + 分类(P[延误>15min])  ─┐  基线: RF / 规则
      │                                                    │
      ▼                                                    ▼
[risk]     分位数 5 级分级（L1..L5，L4/L5=高风险）      特征重要性/SHAP
      │
      ▼
[schedule] CP-SAT 约束优化（时隙偏移/缓冲）  ── 基线: 贪心
      │
      ▼
[evaluate] 三大指标 + 基线对比 + 回归误差
      │
      ▼
[viz]      静态图 + 甘特动画 + Streamlit 面板
```

---

## 4. 项目结构

**根目录：`D:\xiangmu\flight-delay-scheduling`**

```
flight-delay-scheduling/
├── pyproject.toml            # 依赖与打包（含 console_scripts: flightopt）
├── config.yaml               # 全局配置（seed、路径、超参范围、约束参数）
├── README.md                 # 架构图、结果指标表、运行说明、技术选型理由
├── .gitignore                # 忽略 venv/缓存/数据/模型等（见第 18 节）
├── .github/workflows/ci.yml  # GitHub Actions：lint + test
├── data/
│   ├── raw/                  # 真实/公开数据（用户后续放入；被 gitignore）
│   └── processed/            # 生成的中间产物（被 gitignore）
├── outputs/
│   ├── figures/             # 静态图、甘特动画
│   ├── models/              # 序列化模型（被 gitignore）
│   └── reports/             # metrics.json、report.md、feature_dict.md
├── src/flightopt/
│   ├── __init__.py
│   ├── config.py            # 加载 config.yaml → 强类型配置对象
│   ├── paths.py             # 用 pathlib 统一管理路径（跨平台）
│   ├── data/
│   │   ├── synth.py         # 合成数据生成器（主数据源）
│   │   └── loader.py        # 公开数据加载/适配器（可选）
│   ├── features.py          # 特征工程
│   ├── predict.py           # 训练/预测（LightGBM + 基线）
│   ├── risk.py              # 分位数风险分级
│   ├── schedule.py          # CP-SAT 调度 + 贪心基线
│   ├── evaluate.py          # 指标计算与对比
│   ├── viz.py               # 绘图与甘特动画
│   └── cli.py               # 命令行入口（typer）
├── app/streamlit_app.py     # 交互面板
└── tests/
    ├── test_synth.py
    ├── test_features.py
    ├── test_predict.py
    ├── test_risk.py
    ├── test_schedule.py
    └── test_evaluate.py
```

---

## 5. 环境与依赖（Windows）

- **Python**：3.11
- **虚拟环境**（在项目根目录）：
  ```powershell
  cd D:\xiangmu\flight-delay-scheduling
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1        # PowerShell（cmd 用 .venv\Scripts\activate.bat）
  pip install -e .                     # 安装本项目及依赖
  ```
- **核心库**：`pandas`、`numpy`、`scikit-learn`、`lightgbm`、`optuna`、`shap`、`ortools`、`matplotlib`、`plotly`、`streamlit`、`pyyaml`、`typer`
- **测试/质量**：`pytest`、`ruff`
- 依赖在 `pyproject.toml` 中**固定主版本**。
- **路径处理**：代码内一律用 `pathlib.Path`，禁止硬编码 `/` 或 `\` 分隔符；根路径可由 `config.yaml` 覆盖，默认取项目根目录。
- 全程可离线运行（合成数据 + 上述纯 Python/科学计算库）。

---

## 6. 数据层规格

### 6.1 合成数据生成器（`data/synth.py`，主数据源）

生成 **≈2000 条航班记录**，并**故意注入因果结构**，使特征对标签真实可预测、三大指标可复现。

**实体设定**（可在 config 调整）：
- 机场 8–12 个；承运人 5–8 家；机型 4–6 种。
- 飞机 ≈120 架，每架当天飞**多段航线（2–4 段）**，用同一 `tail_id` 串接 → 支持**延误传播**。
- 每机场每小时一个**天气状态**（能见度、风速、降水、雷暴标志），时空平滑。

**字段 Schema**：

| 字段 | 类型 | 说明 |
|---|---|---|
| flight_id | str | 唯一标识 |
| tail_id | str | 机尾号（串接同机连续航段） |
| leg_index | int | 该机当天第几段 |
| airline / origin / dest / aircraft_type | category | — |
| sched_dep / sched_arr | datetime | 计划起降 |
| distance | float | 航段距离(km) |
| min_turnaround | int | 该机型最小周转(min) |
| prev_leg_delay | float | 前序航段延误(min)，首段=0 |
| vis / wind / precip / thunder | float/int | 起飞机场起飞时刻气象原始量 |
| dep_delay | float | **回归标签**：起飞延误(min) |
| is_delayed15 | int | **分类标签**：dep_delay>15 |

**标签生成模型**（注入因果，含噪声）：
```
weather_severity = 归一化加权(低能见度, 大风, 强降水, 雷暴)          # 0..1
congestion       = 该机场起飞时段窗口内计划起飞架次(标准化)          # 0..1
peak             = 时段是否处于早/晚高峰                            # {0,1}
base_delay = a0
           + a1*weather_severity
           + a2*congestion
           + a3*peak
           + a4*weather_severity*peak         # 非线性交互
           + a5*prev_leg_delay                # 传播项
           + N(0, sigma)                       # 噪声
dep_delay = max(0, base_delay)
```
- 系数 `a*`、`sigma` 放入 config，取值使 `is_delayed15` 正类占比落在 **20%–35%**，各特征均有可辨识信号。
- 生成器落盘 `data/processed/flights.parquet`。

**验收**：Schema 完整；正类比例在区间；`prev_leg_delay`、`weather_severity`、`congestion` 与 `dep_delay` 正相关显著；固定 seed 完全复现。

### 6.2 公开数据加载器（`data/loader.py`，可选）

- `load_public(csv_path) -> DataFrame`，把常见公开航班数据（如 BTS/Kaggle "Flight Delays"）适配到 6.1 统一 Schema。
- 缺失字段用机型默认值填充；无 `tail_id` 时按 (aircraft, 时间序) 近似串接。
- `data/raw/` 无文件则跳过，默认走合成数据。

---

## 7. 特征工程规格（`features.py`）

输入统一 Schema 的 DataFrame，输出模型可用特征矩阵。**严禁数据泄露**。

**7 个核心特征**：
1. `weather_severity`：气象量归一化加权合成（0–1）。
2. 时段编码：`sched_dep` 小时的**周期编码**（sin/cos），派生 `time_bucket` 与 `is_peak`。
3. `airport_congestion`：起飞机场起飞时刻 ±30min 窗口内计划起飞架次（仅用计划信息，无泄露）。
4. `carrier_ontime_rate`：承运人历史准点率，**用 out-of-fold / 仅训练集统计**（见第 8 节 CV 内计算）。
5. `distance`（派生 `sched_duration`）。
6. 日历：`day_of_week`、`is_weekend`、`is_holiday`。
7. `prev_leg_delay`：延误传播主特征。

**3 个交互特征**：
1. `weather_severity × is_peak`
2. `airport_congestion × is_peak`
3. `prev_leg_delay × turnaround_slack`（`turnaround_slack = 计划周转 − min_turnaround`）

**要求**：
- 类别特征交给 LightGBM 原生处理；给 RF 基线则 One-Hot。
- 输出"特征字典"（`outputs/reports/feature_dict.md`），逐项说明定义与动机。
- `build_features(df, fit_stats=None) -> (X, y_reg, y_clf, stats)`，`stats` 承载训练集统计量供验证/测试集无泄露复用。

---

## 8. 预测模块规格（`predict.py`）

**任务**：双头预测 —— 回归 `dep_delay` + 二分类 `is_delayed15`。

**模型**：
- 主模型：**LightGBM**（`LGBMRegressor` + `LGBMClassifier`）。
- 基线：**RandomForest**（回归+分类）与**朴素规则**（`weather_severity>阈值 或 prev_leg_delay>阈值 → 高风险`）。

**验证与调参**：
- **切分**：主用时间序切分（前 80% 训练，后 20% 测试）；另做**按 `tail_id` 的 GroupKFold** 检验对新机尾的泛化。
- **调参**：**Optuna**（TPE，50–100 trial），回归目标 MAE，分类目标 PR-AUC 或 F1。搜索空间（放入 config）：`num_leaves`、`max_depth`、`learning_rate`、`n_estimators`、`min_child_samples`、`subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda`。
- **防泄露**：`carrier_ontime_rate` 等统计特征每折内计算后映射到验证折。

**输出**：
- 模型序列化至 `outputs/models/`。
- 回归 MAE/RMSE；分类 Precision/Recall/F1/PR-AUC/混淆矩阵。
- **特征重要性 + SHAP** 摘要图（`outputs/figures/`）。
- 接口：`train(X, y_reg, y_clf, cfg) -> models`；`predict(models, X) -> {"delay":..., "risk_proba":...}`。

**验收**：LightGBM 在测试集**优于 RF 与规则基线**；SHAP 能识别天气/传播/拥堵为主要驱动。

---

## 9. 风险分级规格（`risk.py`）

- 以**预测延误值**（或分类概率，二选一并说明）在训练集上的 20/40/60/80 分位点为切点，分 5 级 L1..L5。
- **高风险 = L4/L5**。
- `fit_quantiles(train_pred) -> cutpoints`（切点由训练集定，测试集复用）；`grade(pred, cutpoints) -> labels`。
- 输出高风险捕获率（Recall）并对照规则基线，写入指标报告。

**验收**：各级样本量大致均衡；高风险 Recall ≥ 0.80 且显著高于规则基线。

---

## 10. 调度优化模块规格（`schedule.py`）

### 10.1 问题建模（地面延误 / 时隙分配）

- **决策变量**：为每个（高风险或全部）航班选一个**起飞时隙偏移** `o_f ∈ {-K,...,+K}`（离散到 5 分钟，`K` 如 30min）。
- **延误查表**：对每航班 × 每候选偏移，**预估该偏移下的延误** `D[f][o]`（用已训练模型或拥堵模型重算：偏移改变所在时窗拥堵度，从而改变预测延误；传播项沿机尾链条向后传递）。得到查表后优化变为清晰的**赋值问题**。
- **目标**：`min Σ_f w_f · D[f][o_f]`，高风险航班 `w_f` 更大（权重放 config）。
- **硬约束**：
  1. **时隙容量**：任一时间窗内起飞架次 ≤ 跑道容量 `C`。
  2. **周转约束**：同一 `tail_id` 相邻航段，后段起飞 ≥ 前段到达 + `min_turnaround`（前段到达随其偏移联动）。
  3. **时窗约束**：`|o_f| ≤ K`。
  4. **宵禁**：禁飞时窗内不得起飞。

### 10.2 求解器

- **主：OR-Tools CP-SAT**。整数变量表示 `o_f`（或 one-hot 选择），线性化目标（查表值为系数），加入上述约束，设求解时限。CP-SAT 对硬约束给可行性保证 → **约束满足率接近 100%**（可放松的约束设为带惩罚软约束，则如实统计满足率）。
- **基线：贪心迭代局部搜索**（还原原始路线）：按风险从高到低，为每航班在合法邻域内选使加权总延误下降最多的偏移，多轮扫描至收敛。
- **对照实验**：在报告中给出 CP-SAT vs 贪心 在（总延误、约束满足率、求解时间）上的对比。

### 10.3 输出

- 优化后时刻表（含每航班 `o_f`）。
- 逐迭代/求解过程 `(总延误, 约束满足率)` 轨迹，供甘特动画使用。
- 接口：`optimize(flights, risk, cfg, solver="cpsat"|"greedy") -> (schedule, trace, metrics)`。

**验收**：CP-SAT 约束满足率 ≥ 0.85（并展示相对贪心的提升）；高风险航班平均延误较优化前下降 ≥ 1.0 min/架次。

---

## 11. 评估模块规格（`evaluate.py`）

- 汇总落盘 `outputs/reports/metrics.json` 与 `report.md`，含：
  - 回归 MAE/RMSE（LightGBM vs RF vs 均值基线）。
  - 高风险 Recall/Precision/F1/PR-AUC（vs 规则基线）。
  - 调度约束满足率、高风险延误减少（CP-SAT vs 贪心 vs 优化前）。
- 每个指标须有**明确定义 + 基线对比**。
- `summarize(...) -> dict`，并把最终指标表**自动写回 README** 的占位表格。

---

## 12. 可视化与交互界面

### 12.1 静态图（`viz.py`）
延误分布、特征重要性、SHAP 摘要、混淆矩阵/PR 曲线、优化前后延误对比条形图。

### 12.2 甘特动画
`matplotlib.animation.FuncAnimation`（导出 gif/mp4）或 Plotly（导出交互 html）：横轴时间、纵轴机尾/航班，色块平移表示 `o_f`；叠加"总延误"与"约束满足率"收敛曲线。

### 12.3 Streamlit 面板（`app/streamlit_app.py`）
- 侧栏：生成/加载数据、设定天气强度与跑道容量、选择求解器。
- 主区：预测结果表、风险分布、指标卡片、甘特图、SHAP 图、"运行全流程"按钮。
- 启动：`streamlit run app/streamlit_app.py`。

---

## 13. 测试要求（`tests/`，pytest）

- `test_synth`：Schema 完整、正类比例在区间、seed 可复现、注入相关性成立。
- `test_features`：无缺失、无泄露、交互特征数值正确。
- `test_predict`：管线可训练；LightGBM 优于均值/规则基线；输出维度正确。
- `test_risk`：分级单调、切点由训练集决定、高风险占比合理。
- `test_schedule`：CP-SAT 解满足全部硬约束；优化后加权总延误 ≤ 优化前；贪心可运行。
- `test_evaluate`：指标计算正确（构造样例校验）。

---

## 14. 运行方式（CLI，跨平台）

CLI（`src/flightopt/cli.py`，typer；`pyproject.toml` 注册 `flightopt` 命令）子命令：
```powershell
python -m flightopt gen-data      # 生成合成数据
python -m flightopt train         # 训练 + 调参 + 存模型
python -m flightopt grade         # 风险分级
python -m flightopt schedule --solver cpsat   # 调度优化（cpsat|greedy）
python -m flightopt evaluate      # 出指标
python -m flightopt viz           # 出图与动画
python -m flightopt run-all       # 一键全流程
```
> 不使用 Makefile（Windows 默认无 make）。所有操作统一走上面的 CLI。

---

## 15. 配置化

`config.yaml` 统一管理：`seed`、路径、数据生成系数、特征参数、Optuna 搜索空间、调度约束参数（`K`、`C`、`min_turnaround`、宵禁时段、权重）、指标目标。全代码从 config 读取，**禁止硬编码魔数**。

---

## 16. （可选）GitHub Actions CI

`.github/workflows/ci.yml`：push 时在 Ubuntu + Python 3.11 上跑 `ruff check` 与 `pytest`。因数据为合成、无外部依赖，CI 可完整跑通预测/分级/调度的快速冒烟测试。

---

## 17. 里程碑与验收标准

| 里程碑 | 内容 | 完成标准（DoD） |
|---|---|---|
| **M1 数据** | 合成生成器 + 加载器 + 测试 | `gen-data` 产出 2000 条，`test_synth` 绿，相关性成立 |
| **M2 特征** | 7+3 特征 + 特征字典 + 测试 | 无泄露，`test_features` 绿，字典生成 |
| **M3 预测** | LightGBM 双头 + Optuna + 基线 + SHAP | 优于基线，模型落盘，`test_predict` 绿 |
| **M4 分级** | 分位数 5 级 + 捕获率 | 高风险 Recall ≥0.80 且优于规则，`test_risk` 绿 |
| **M5 调度** | CP-SAT + 贪心基线 + 对照 | 硬约束满足率 ≥0.85，延误降 ≥1min/架次，`test_schedule` 绿 |
| **M6 评估+界面** | 指标报告 + 静态图 + 甘特动画 + Streamlit + README | `run-all` 一键跑通，`pytest` 全绿，README 指标表填充，面板可启动 |
| **M7 落地+发布** | 放到 `D:\xiangmu`，推送 GitHub | 项目位于 `D:\xiangmu\flight-delay-scheduling`，GitHub 仓库可访问、README 正常渲染、CI 通过 |

---

## 18. 落地路径与 GitHub 发布（M7）

### 18.1 落地到本地路径
- 确认/创建目录 `D:\xiangmu`，项目根目录为 `D:\xiangmu\flight-delay-scheduling`。
- 全部代码、配置、`README.md`、测试、`outputs/reports` 等置于此目录下。

### 18.2 `.gitignore`（提交前必须就位）
至少忽略：
```
.venv/            __pycache__/     *.pyc
data/raw/         data/processed/  *.parquet  *.csv
outputs/models/   outputs/figures/*.mp4
.env              .idea/  .vscode/
```
> 保留并提交：源码、`config.yaml`、`README.md`、`tests/`、`outputs/reports/report.md`、少量小体积展示图（`outputs/figures/*.png`）。大动画（gif/mp4）建议不提交，或改用 Git LFS。

### 18.3 发布前置条件（Claude Code 先检查，缺失则提示用户自行完成）
- 已安装 Git：`git --version`。
- 已安装并登录 GitHub CLI：`gh auth status`。
  - **若未登录，请让用户自己执行 `gh auth login` 完成认证**；不要在代码或命令里写入任何密码 / token / 明文凭据。
- 用户可选择仓库**公开（作品集展示，推荐）或私有**——发布前向用户确认一次可见性。

### 18.4 发布步骤（在项目根目录执行）
```powershell
cd D:\xiangmu\flight-delay-scheduling
git init
git add .
git commit -m "Initial commit: flight delay prediction & scheduling optimization"

# 首选（GitHub CLI，自动创建远程仓库并推送；公开可改 --private）
gh repo create flight-delay-scheduling --public --source . --remote origin --push

# 备选（若无 gh）：用户先在 github.com 手动建空仓库，然后
# git branch -M main
# git remote add origin https://github.com/<用户名>/flight-delay-scheduling.git
# git push -u origin main
```

### 18.5 发布验收
- 远程仓库 URL 可访问，`README.md` 在 GitHub 正常渲染（含架构图、指标表、运行说明）。
- 若配置了 CI，Actions 页面显示通过。
- 确认**未提交**任何数据、模型二进制或敏感信息。

---

## 19. 交付物清单

- [ ] 纯 Python 代码库（第 4 节结构），位于 `D:\xiangmu\flight-delay-scheduling`
- [ ] `python -m flightopt run-all` 一键跑通全流程
- [ ] 合成数据生成器（+ 可选公开数据加载器）
- [ ] LightGBM 预测模型（+ RF/规则基线对照）+ SHAP 解释
- [ ] 分位数 5 级风险分级
- [ ] CP-SAT 调度优化（+ 贪心基线对照）
- [ ] `metrics.json` + `report.md`（含全部指标与基线对比）
- [ ] 静态图 + 甘特动画 + Streamlit 面板
- [ ] `pytest` 测试套件全绿
- [ ] README（架构图、结果指标表、运行说明、技术选型理由）
- [ ] （可选）GitHub Actions CI
- [ ] 项目已推送至用户 GitHub，仓库可访问

---

**备注给用户**：把本文件整份交给 Claude Code 即可。发布 GitHub 前请确认已安装 Git 与 GitHub CLI 并完成 `gh auth login`，并想好仓库要公开还是私有。若之后拿到真实公开数据（如 BTS/Kaggle 航班延误 CSV），放入 `data/raw/` 并按 6.2 节适配 Schema，即可无缝替换合成数据复跑全流程。
