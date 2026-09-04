# 信贷违约风控 · 端到端机器学习系统（建模 + MLOps）· 技术规格书

> **本文档面向 Claude Code。** 请把它当作一份完整的项目实施规格（PRD + 技术设计），用 **Python** 自主完成端到端开发，最终落地到本地路径 **`D:\xiangmu\credit-risk-mlops`**，并**发布到用户的 GitHub**。目标是交付一个可运行、可测试、可复现、**面试拿得出手**的作品：一个信贷违约风控模型，外加**从建模到上线的完整 MLOps 链路**。

---

## 0. 给 Claude Code 的执行说明（先读这一节）

- **语言**：全程 **Python 3.11**。所有路径用 `pathlib.Path`，保证在 **Windows** 正常工作。
- **落地路径**：项目根目录 **`D:\xiangmu\credit-risk-mlops`**。`D:\xiangmu` 不存在则创建。
- **硬件**：本项目主模型是**树模型（LightGBM），纯 CPU 即可**，不依赖 GPU，不吃显存，天然稳。
- **核心定位**：项目的星光在 **MLOps 工程链路**（训练流水线 → 实验追踪 → 服务化 → 容器化 → CI/CD → 漂移监控），建模层**刻意保持简洁**，别在调模型上耗时间。
- **范围纪律（重要）**：**只做本文档列出的核心 MLOps 组件，不要额外堆工具**（不引入 K8s、Airflow、云服务等）。宁可少而精、每件都跑通，也不要多而散。
- **发布 GitHub**：全部开发与自测完成后，按第 19 节推送。
- **构建顺序**：严格按第 18 节里程碑（M1→M7）推进；**每完成一个里程碑先跑通对应测试再继续**。项目任何时候都应处于"能跑通全流程"状态。
- **完成定义（DoD）**：`python -m creditrisk run-all` 一键跑通 数据→特征→训练→评估→注册模型；`uvicorn` 起 FastAPI 服务可预测；`docker build` 能构建镜像；`pytest` 全绿；README 指标表被真实结果填充；MLflow UI 与 Streamlit 面板可启动；项目已推送 GitHub。
- **可复现**：全局随机种子集中在 `config.yaml`，所有随机过程读取它。

---

## 1. 项目目标与范围

构建一个**端到端机器学习系统**，覆盖真实企业里"模型从开发到上线"的完整生命周期：

1. **建模层**：在信贷数据上训练违约概率预测模型（二分类，处理类别不平衡），做完整评估与可解释。
2. **训练流水线 + 实验追踪**：可复现的训练管线，用 MLflow 记录参数/指标/产物并做模型注册。
3. **服务化**：用 FastAPI 把模型包成 REST API（在线打分）。
4. **容器化 + CI/CD**：Docker 打包，GitHub Actions 自动 lint + 测试。
5. **上线监控**：数据漂移监控（PSI），并用 Streamlit 面板做监控 + 演示。

**核心指标（复现目标，须真实达成或如实报告，不许编数）**：

| 指标 | 定义 | 目标 | 对照基线 |
|---|---|---|---|
| AUC | ROC-AUC | LightGBM > 逻辑回归；参考 ~0.86 | 逻辑回归 / 常数基线 |
| KS | 好坏样本分离度 max(TPR−FPR) | 越高越好，须报告 | 逻辑回归 |
| PR-AUC | 精确率-召回率曲线下面积（不平衡更看重） | LightGBM > 逻辑回归 | 逻辑回归 |
| PSI（监控） | 分布稳定性指数 | 演示：注入漂移后能检出（PSI>0.25 触发告警） | — |

> 参考量级（仅 sanity check）：Give Me Some Credit 上好模型 AUC 常见 0.86~0.87。**以实跑为准并如实写入 README。**

---

## 2. 技术选型

| 环节 | 采用方案 | 保留基线/理由 |
|---|---|---|
| 主模型 | **LightGBM**（二分类，`scale_pos_weight` 处理不平衡） | 表格数据强、快、CPU 友好；基线用**逻辑回归** |
| 调参 | **Optuna**（轻量，20~50 trial，目标 AUC） | 比网格搜索样本效率高 |
| 可解释 | **SHAP** + 特征重要性 | 风控强需求：讲清"为什么拒绝" |
| 评估 | AUC / **KS** / PR-AUC / 混淆矩阵 / 校准 + **评分卡映射** | KS/评分卡是风控行业标准 |
| 实验追踪 | **MLflow**（本地 file-based，含 Model Registry） | 无需起服务器，稳；面试常问 |
| 服务化 | **FastAPI** + Pydantic 校验 + Uvicorn | 主流、轻、自带文档页 |
| 容器化 | **Docker**（+ 可选 docker-compose 同起 API 与 MLflow UI） | 一键复现环境 |
| CI/CD | **GitHub Actions**（ruff + pytest） | 自动质量门禁 |
| 漂移监控 | **PSI**（手写实现，风控标准；可选 Evidently 出报告） | 域内地道、可解释 |
| 面板/演示 | **Streamlit**（指标 + SHAP + 在线打分 + 漂移页） | 展示效果强 |
| 工程化 | 配置化 + Typer CLI + pytest | 可测试、可复现 |

> 设计原则：**建模与服务解耦**。训练产出一个"注册模型 + 预处理器"工件，服务层只加载工件对外打分，任一层可独立替换。

---

## 3. 系统架构与数据流

```
Give Me Some Credit 数据 (cs-training.csv)
      │
      ▼
[data]     清洗(缺失/异常/哨兵值) + 按行分层切分 训练/验证/测试(防泄露)
      │
      ▼
[features] 缺失填补 + 异常截断 + 派生特征 (+可选 WOE)；预处理器可序列化
      │
      ▼
[train]    LightGBM(不平衡加权) + Optuna 调参    基线: 逻辑回归
      │        │
      │        ├──► [tracking] MLflow: 记录 params/metrics/artifacts + 注册模型
      ▼        ▼
[evaluate] AUC/KS/PR-AUC/校准 + SHAP + 评分卡    对照基线
      │
      ▼
[registry] 最优模型 + 预处理器 打包为部署工件
      │
      ├───────────────► [serve]  FastAPI /predict → Docker 镜像
      │
      └───────────────► [monitor] PSI 漂移检测(注入漂移演示) + Streamlit 面板
```

---

## 4. 项目结构

**根目录：`D:\xiangmu\credit-risk-mlops`**

```
credit-risk-mlops/
├── pyproject.toml            # 依赖与打包（console_scripts: creditrisk）
├── config.yaml               # 全局配置（seed、路径、超参、阈值、PSI 告警线）
├── README.md                 # 架构图、指标表、运行说明、技术选型、面试讲法
├── Dockerfile                # 服务镜像
├── docker-compose.yml        # （可选）API + MLflow UI
├── .dockerignore
├── .gitignore                # 见第 19 节
├── .github/workflows/ci.yml  # GitHub Actions：ruff + pytest（+可选 docker build）
├── data/
│   ├── raw/                  # cs-training.csv（被 gitignore）
│   └── processed/            # 切分后的数据（被 gitignore）
├── mlruns/                   # MLflow 本地追踪（被 gitignore）
├── outputs/
│   ├── figures/             # ROC/KS/校准/SHAP/漂移图
│   ├── models/              # 部署工件：model.pkl + preprocessor.pkl（被 gitignore）
│   └── reports/             # metrics.json、report.md
├── src/creditrisk/
│   ├── __init__.py
│   ├── config.py            # 加载 config.yaml → 强类型配置
│   ├── paths.py             # pathlib 统一路径
│   ├── data.py              # 下载/校验 + 清洗 + 切分
│   ├── features.py          # 预处理器（fit/transform，可序列化）
│   ├── model.py             # 训练(LightGBM+LR) + Optuna + MLflow 记录/注册
│   ├── evaluate.py          # AUC/KS/PR-AUC/校准 + SHAP + 评分卡
│   ├── monitor.py           # PSI 漂移检测 + 注入漂移演示
│   ├── schemas.py           # Pydantic 请求/响应模型
│   ├── serve.py             # FastAPI app
│   └── cli.py               # 命令行入口（typer）
├── app/streamlit_app.py     # 监控 + 演示面板
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_model.py
    ├── test_evaluate.py
    ├── test_monitor.py
    └── test_api.py
```

---

## 5. 环境与依赖（Windows）

- **Python**：3.11
- **虚拟环境**（项目根目录）：
  ```powershell
  cd D:\xiangmu\credit-risk-mlops
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .
  ```
- **核心库**：`pandas`、`numpy`、`scikit-learn`、`lightgbm`、`optuna`、`shap`、`mlflow`、`fastapi`、`uvicorn`、`pydantic`、`streamlit`、`matplotlib`、`pyyaml`、`typer`、`joblib`；可选 `evidently`
- **测试/质量**：`pytest`、`httpx`（测 FastAPI）、`ruff`
- 依赖在 `pyproject.toml` **固定主版本**。
- **Docker**：安装 Docker Desktop（用户自备）；Claude Code 只需保证 `Dockerfile` 正确、`docker build` 能通过。

---

## 6. 数据层规格（`data.py`）

### 6.1 数据集：Give Me Some Credit
- Kaggle 竞赛数据，训练集 `cs-training.csv`，约 **15 万行**、文件仅几 MB（远小于 200MB）。
- **标签**：`SeriousDlqin2yrs`（1 = 两年内发生 90 天以上逾期，正类；**约占 6~7%，天然不平衡**）。
- **10 个特征**：授信循环额度使用率、年龄、30–59/60–89/90+ 天逾期次数、负债比、月收入、开放信贷额度数、房产贷款数、家属人数。
- **已知数据问题（都要处理，也都是面试谈资）**：`MonthlyIncome`、`NumberOfDependents` 有缺失；逾期次数字段存在异常哨兵值（如 96/98）；使用率与负债比有极端离群值。

### 6.2 获取方式（Claude Code 先检查，缺失则提示用户）
- 优先用 Kaggle API：`kaggle competitions download -c GiveMeSomeCredit`（需用户已放置 `kaggle.json` 凭据）。
- **若无凭据**：提示用户自行从 Kaggle 下载 `cs-training.csv` 放入 `data/raw/`，然后继续。**不要在代码里写入任何账号密码或 token。**

### 6.3 切分（防泄露）
- 官方测试集无标签，故**只用 `cs-training.csv`**，按标签**分层**切分为 训练/验证/测试 = 70/15/15（`stratify` 保持正类比例一致）。
- **所有填补/截断/编码统计量只在训练集上 fit**，再 transform 验证/测试集。

**验收**：切分分层正确、正类比例三集一致；预处理统计仅来自训练集；固定 seed 完全复现。

---

## 7. 特征工程规格（`features.py`）

做一个可序列化的**预处理器**（`fit`/`transform`），保证训练与线上服务用**同一套变换**（避免训练-服务偏移，这是面试重点）。

- **缺失填补**：`MonthlyIncome` 用中位数或分组中位数；`NumberOfDependents` 用众数/0；并加"是否缺失"指示位。
- **异常处理**：逾期次数字段的哨兵值（96/98）截断或单独成桶；使用率、负债比按分位数（如 99%）截断。
- **派生特征（少而有信号）**：总逾期次数、人均月收入（收入/(家属+1)）、月收入与负债比的交互、年龄分桶。
- **（可选风控味）WOE 编码**：对分桶特征做 WOE 变换，附带 IV 值，为"评分卡"做铺垫（时间紧可跳过，README 注明）。
- **接口**：`Preprocessor.fit(X_train)`；`transform(X)`；`save/load`（joblib 序列化到 `outputs/models/preprocessor.pkl`）。
- 输出"特征说明表"到 `outputs/reports/feature_dict.md`。

**验收**：无泄露；预处理器可序列化并复用；`transform` 对单条与批量都可用（服务层需要）。

---

## 8. 建模层规格（`model.py` + `evaluate.py`）

### 8.1 训练
- 主模型 **LightGBM**（`LGBMClassifier`），用 `scale_pos_weight` 或 `class_weight` 处理不平衡（**优先加权，不首选 SMOTE**；如做 SMOTE 仅作对照并说明）。
- 基线 **逻辑回归**（标准化 + L2）与**常数/热门基线**。
- **Optuna** 轻量调参（20~50 trial，目标验证集 AUC）：`num_leaves`、`max_depth`、`learning_rate`、`n_estimators`、`min_child_samples`、`subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda`、`scale_pos_weight`。搜索空间放 config。

### 8.2 MLflow 追踪与注册
- 每次训练用 MLflow 记录：超参、AUC/KS/PR-AUC、ROC/KS/校准图、SHAP 摘要图、模型工件。
- 用 **Model Registry** 注册最优模型为 `credit-risk-model`，打 stage 标记（如 `Staging`/`Production`）。
- 本地 `mlruns/` 追踪，`mlflow ui` 可查看对比。

### 8.3 评估（`evaluate.py`）
- **AUC、KS、PR-AUC**、指定阈值下的混淆矩阵/精确率/召回率、**校准曲线**。
- **SHAP**：全局重要性 + 摘要图；能讲清主要驱动因素（逾期历史、使用率等）。
- **阈值选择**：给出按 KS 最大点或按业务代价的阈值建议。
- **评分卡映射**：把违约概率映射为直观信用分（如 300~850），写进报告。
- 落盘 `outputs/reports/metrics.json` 与 `report.md`，含基线对比；**自动写回 README 占位指标表**。

**验收**：LightGBM 的 AUC/KS/PR-AUC **优于逻辑回归与常数基线**；SHAP 结果合理；MLflow 中可见完整实验记录。

---

## 9. 服务化规格（`serve.py` + `schemas.py`）

- **FastAPI 应用**，启动时加载部署工件（`preprocessor.pkl` + 注册模型）。
- **接口**：
  - `GET /health`：健康检查。
  - `POST /predict`：入参为单个申请人的原始字段（Pydantic 校验类型/范围），返回 `{default_probability, credit_score, risk_level, top_reasons}`（`top_reasons` 用 SHAP 给出主要影响因子）。
  - `POST /predict/batch`：批量。
- **输入校验**：用 `schemas.py` 的 Pydantic 模型，非法输入返回 422 并给出清晰错误。
- 自带 Swagger 文档页（`/docs`）。
- 启动：`uvicorn creditrisk.serve:app --reload`。

**验收**：合法请求返回结构正确的打分与解释；非法请求被 Pydantic 拦截；`/docs` 可访问。

---

## 10. 容器化规格（`Dockerfile`）

- 基于 `python:3.11-slim`，装依赖、拷贝代码与部署工件，`EXPOSE` 端口，`CMD` 启 Uvicorn。
- `.dockerignore` 排除 `.venv/`、`mlruns/`、`data/`、`__pycache__/` 等。
- **（可选）`docker-compose.yml`**：同时起 API 服务与 MLflow UI，一条命令拉起。
- **验收**：`docker build -t credit-risk-api .` 成功；`docker run -p 8000:8000 credit-risk-api` 后可访问 `/health` 与 `/docs`。

---

## 11. 漂移监控规格（`monitor.py`）

- **PSI（Population Stability Index）**：对每个特征，比较"参考分布（训练集）"与"当前分布（新到数据）"，分桶后按 `Σ(当前占比−参考占比)·ln(当前占比/参考占比)` 计算。约定阈值：`<0.1` 稳定、`0.1~0.25` 轻微、`>0.25` 显著漂移（阈值放 config）。
- **漂移注入演示**：从测试集造一份"漂移数据"（如整体调高负债比/调低收入、或按时间/年龄切片），跑 PSI，**证明能检出并触发告警**。
- 同时监控**模型性能漂移**：在有标签的新数据上重算 AUC/KS，观察衰减。
- **（可选）Evidently**：生成一份 HTML 数据漂移报告，更直观。
- 接口：`compute_psi(ref, cur, feature) -> float`；`drift_report(ref_df, cur_df, cfg) -> dict`。

**验收**：PSI 计算正确（构造样例校验）；注入漂移后显著特征 PSI 越过告警线并被标记。

---

## 12. 可视化 / 监控面板（`app/streamlit_app.py`）

- **概览页**：模型指标卡片（AUC/KS/PR-AUC）、ROC/KS 曲线、SHAP 全局重要性。
- **在线打分页**：表单输入一个申请人 → 调用打分逻辑 → 显示违约概率、信用分、风险等级、SHAP 归因（"为什么是这个分"）。
- **漂移监控页**：展示各特征 PSI 条形图、告警状态、性能衰减曲线（用注入漂移的数据演示）。
- 启动：`streamlit run app/streamlit_app.py`。

---

## 13. 测试要求（`tests/`，pytest）

- `test_data`：切分分层正确、正类比例一致、seed 可复现、已知缺失/哨兵值被正确处理。
- `test_features`：无泄露、预处理器可序列化且单条/批量一致、派生特征数值正确。
- `test_model`：可训练；LightGBM AUC 优于逻辑回归；MLflow 记录了指标与工件。
- `test_evaluate`：AUC/KS/PR-AUC 用构造样例校验；评分卡映射单调。
- `test_monitor`：PSI 计算正确；注入漂移后越过告警线。
- `test_api`：用 `httpx`/`TestClient` 测 `/health` 与 `/predict`（合法返回结构正确、非法返回 422）。

---

## 14. 运行方式（CLI，跨平台）

CLI（`src/creditrisk/cli.py`，typer；注册 `creditrisk` 命令）：
```powershell
python -m creditrisk get-data       # 下载/校验数据
python -m creditrisk prep           # 清洗 + 切分 + 拟合预处理器
python -m creditrisk train          # 训练 + Optuna + MLflow 记录/注册
python -m creditrisk evaluate       # 指标 + SHAP + 评分卡 + 出图
python -m creditrisk monitor        # PSI 漂移演示
python -m creditrisk run-all        # 一键：get-data→prep→train→evaluate
# 服务：uvicorn creditrisk.serve:app --reload
# 面板：streamlit run app/streamlit_app.py
# 追踪：mlflow ui
```
> 不使用 Makefile。所有操作统一走 CLI 或上面的命令。

---

## 15. 配置化

`config.yaml` 统一管理：`seed`、路径、`rating`/切分比例、Optuna 搜索空间、决策阈值、评分卡参数、PSI 分桶数与告警线、服务端口。全代码从 config 读取，**禁止硬编码魔数**。

---

## 16. GitHub Actions CI/CD（`.github/workflows/ci.yml`）

- push 时在 Ubuntu + Python 3.11 跑 `ruff check` 与 `pytest`。
- 测试用**极小抽样数据或合成迷你数据**跑冒烟，不训练完整模型、不依赖 Kaggle 凭据。
- **（可选）** 追加一步 `docker build` 验证镜像可构建。

---

## 17.（略，见 18）

## 18. 里程碑与验收标准

| 里程碑 | 内容 | 完成标准（DoD） |
|---|---|---|
| **M1 数据** | 获取 + 清洗 + 分层切分 + 测试 | `prep` 产出三集，`test_data` 绿，无泄露 |
| **M2 特征** | 可序列化预处理器 + 特征说明表 + 测试 | 无泄露、单条/批量一致，`test_features` 绿 |
| **M3 建模+追踪** | LightGBM+LR + Optuna + MLflow 注册 + SHAP | 优于基线，实验入 MLflow，`test_model` 绿 |
| **M4 评估** | AUC/KS/PR-AUC/校准 + 评分卡 | 指标报告生成，`test_evaluate` 绿，README 表填充 |
| **M5 服务+容器** | FastAPI + Docker | `/predict` 可用，`docker build/run` 通过，`test_api` 绿 |
| **M6 监控+面板** | PSI 漂移 + Streamlit + README | `monitor` 能检出漂移，面板三页可用，`pytest` 全绿 |
| **M7 落地+发布** | 放到 `D:\xiangmu`，推送 GitHub + CI 通过 | 仓库可访问、README 正常渲染、Actions 通过 |

---

## 19. 落地路径与 GitHub 发布（M7）

### 19.1 落地
项目根目录 `D:\xiangmu\credit-risk-mlops`，全部代码/配置/README/测试/`outputs/reports` 置于此。

### 19.2 `.gitignore`（提交前必须就位）
```
.venv/   __pycache__/   *.pyc
data/raw/   data/processed/   *.csv
mlruns/   outputs/models/   *.pkl
outputs/figures/*.html
.env   .idea/   .vscode/
```
> 提交：源码、`config.yaml`、`Dockerfile`、`docker-compose.yml`、`README.md`、`tests/`、`outputs/reports/report.md`、少量小体积展示图（`*.png`）。数据、`mlruns/`、模型工件**不提交**。

### 19.3 发布前置（Claude Code 先检查，缺失则提示用户自行完成）
- `git --version`；`gh auth status`。**未登录则让用户自行 `gh auth login`**，不写入任何凭据。
- 发布前确认仓库**公开（作品集展示，推荐）或私有**。

### 19.4 发布步骤
```powershell
cd D:\xiangmu\credit-risk-mlops
git init
git add .
git commit -m "Initial commit: credit-risk model with end-to-end MLOps"
gh repo create credit-risk-mlops --public --source . --remote origin --push
```

### 19.5 发布验收
仓库可访问，README 正常渲染；CI 通过；未提交数据/模型二进制/凭据。

---

## 20. 交付物清单

- [ ] 纯 Python 代码库（第 4 节结构），位于 `D:\xiangmu\credit-risk-mlops`
- [ ] `python -m creditrisk run-all` 一键跑通 数据→训练→评估
- [ ] LightGBM 风控模型（+ 逻辑回归/常数基线）+ 不平衡处理 + SHAP + 评分卡
- [ ] MLflow 实验追踪 + 模型注册
- [ ] FastAPI 在线打分服务（含输入校验与 Swagger 文档）
- [ ] Dockerfile（+ 可选 docker-compose），镜像可构建可运行
- [ ] PSI 数据漂移监控（含注入漂移演示）
- [ ] Streamlit 面板（概览 / 在线打分 / 漂移监控）
- [ ] `pytest` 测试套件全绿（含 API 测试）
- [ ] GitHub Actions CI（ruff + pytest）
- [ ] README（架构图、指标表、运行说明、技术选型、面试讲法）
- [ ] 项目已推送至用户 GitHub，仓库可访问

---

## 21. 面试讲法（写进 README，也要背下来）

**简历一行（跑完后填真实数字）**：
> 搭建信贷违约风控的端到端机器学习系统：LightGBM 模型（AUC __ / KS __，处理类别不平衡 + SHAP 可解释 + 评分卡）+ 全套 MLOps（MLflow 实验追踪与模型注册、FastAPI 服务化、Docker 容器化、GitHub Actions CI、PSI 数据漂移监控），已开源。

**面试高频问，先备好答案**：
- **不平衡数据为什么不看准确率？** 正类仅约 6%，全预测为"不违约"准确率也有 94% 却毫无用处；所以用 AUC、PR-AUC、KS 这些对不平衡稳健、且关注排序/分离能力的指标。
- **KS 是什么？** 好坏样本累积分布之差的最大值，衡量模型把违约/不违约拉开的能力，是风控行业标准。
- **PSI 是什么、为什么要监控漂移？** 群体稳定性指数，比较线上数据分布与训练时的偏移；分布漂移会让模型悄悄失效，PSI>0.25 就该预警甚至重训。
- **为什么要把模型做成服务 + 容器？** 让模型能被真实系统调用、环境可一键复现，是"能用的模型"和"上线的模型"的区别。
- **怎么保证训练和线上用的是同一套处理？** 预处理器随模型一起序列化、服务加载同一工件，避免训练-服务偏移（training-serving skew）。
- **MLflow 解决什么？** 实验可追溯、可对比、模型可版本化注册，团队协作和回滚都靠它。

---

**备注给用户**：把本文件整份交给 Claude Code 即可。需要 Kaggle `kaggle.json` 才能自动下载数据，没有就手动把 `cs-training.csv` 放进 `data/raw/`。Docker 需自行安装 Docker Desktop。跑完把 README 里的指标占位符换成真实结果——面试问起来，第 21 节那些问题你要能自己讲清楚，这才是这个项目真正"拿得出手"的地方。
