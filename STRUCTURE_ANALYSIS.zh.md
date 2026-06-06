# a-evolve 目录结构与功能分析

本文档基于对 `a-evolve` 目录的静态阅读整理，重点说明各目录的结构、职责和相互关系。当前目录是一个独立的 Python 项目，包名为 `a-evolve`，核心 Python 模块名为 `agent_evolve`。

## 1. 总体定位

`a-evolve` 是一个用于“自演化 Agent”的基础设施项目。它的核心思想是：**工作区目录就是 Agent 与 Evolver 的接口**。

- Agent 从工作区读取 `prompts/`、`skills/`、`tools/`、`memory/` 等可演化状态。
- Benchmark 负责提供任务并评价 Agent 轨迹。
- EvolutionEngine 读取观测结果并修改工作区文件。
- EvolutionLoop 负责把 Agent、Benchmark、Engine、版本控制和观测日志串成循环。

典型循环为：

```text
Solve -> Observe -> Snapshot -> Evolve -> Snapshot -> Reload -> Next cycle
```

从项目说明和代码看，`a-evolve` 支持多个 benchmark/domain，包括 SWE-bench、MCP-Atlas、Terminal-Bench 2.0、SkillsBench、ARC-AGI-3、OSWorld、CL Bench 等；并内置多种演化算法，例如 `skillforge`、`guided_synth`、`adaptive_skill`、`adaptive_evolve`、`gepa`、`meta_harness`、`unified`。

## 2. 顶层结构概览

顶层主要目录和文件如下：

```text
a-evolve/
├── .git/                         # 嵌套 Git 仓库元数据
├── .github/workflows/            # GitHub Actions 发布流程
├── agent_evolve/                 # 核心 Python 包
├── artifacts/                    # 已完成实验或演化结果产物
├── docs/                         # 算法和 benchmark 使用文档
├── examples/                     # 各 benchmark 的运行脚本与实验入口
├── figs/                         # README/论文展示图和绘图脚本
├── seed_workspaces/              # 内置初始 Agent 工作区
├── tests/                        # 单元测试与回归测试
├── CLAUDE.md                     # 面向 Claude/Codex 的项目说明，偏 ARC 分支开发说明
├── DESIGN.md                     # 架构设计文档
├── Makefile                      # install/test/lint/fmt 命令
├── QUICKSTART.md                 # 快速上手文档
├── README.md                     # 项目主介绍
└── pyproject.toml                # Python 包配置和依赖声明
```

目录规模统计：

| 目录 | 子目录数 | 文件数 | 功能定位 |
|---|---:|---:|---|
| `agent_evolve/` | 35 | 171 | 核心框架、agent、benchmark、算法实现 |
| `examples/` | 14 | 71 | 实验入口和 benchmark 专用脚本 |
| `seed_workspaces/` | 45 | 57 | 各类 Agent 的初始 prompts/tools/skills/memory |
| `artifacts/` | 13 | 17 | 已演化工作区、评估结果、报告 |
| `docs/` | 1 | 7 | 算法和 setup 文档 |
| `tests/` | 1 | 11 | 主要覆盖 GEPA、UnifiedEngine、SkillBench、ARC MAS |
| `figs/` | 0 | 6 | 图片和绘图脚本 |
| `.github/` | 1 | 1 | PyPI 发布 workflow |

## 3. 顶层文件

### `README.md`

项目主入口文档。说明 `A-Evolve` 的定位是“通用 Agent 自演化基础设施”，强调三行 API：

```python
import agent_evolve as ae

evolver = ae.Evolver(agent="./my_agent", benchmark="swe-verified")
results = evolver.run(cycles=10)
```

它还展示了 benchmark 结果、安装方式、内置 seed workspace、内置 benchmark adapter，以及 Bring Your Own Agent 的 `BaseAgent.solve()` 接口。

### `DESIGN.md`

架构设计说明，最重要的观点是：**workspace IS the interface**。该文件解释了三类组件的职责：

- Data + Eval：`BenchmarkAdapter`，负责 `get_tasks()` 和 `evaluate()`。
- Agent Impl：继承 `BaseAgent`，实现 `solve()`。
- Evolve Algo：继承 `EvolutionEngine`，实现 `step()` 并修改工作区。

它还描述了 workspace contract：

```text
manifest.yaml
prompts/system.md
skills/*/SKILL.md
tools/registry.yaml
memory/*.jsonl
evolution/
```

### `QUICKSTART.md`

快速使用指南，覆盖 uv 安装、运行内置 benchmark、演化 Agent、Bring Your Own Agent、编写自定义演化算法等。注意该项目建议用 `uv` 安装和运行 Python 环境。

### `pyproject.toml`

Python 包配置：

- 项目名：`a-evolve`
- 版本：`0.1.0`
- Python：`>=3.11`
- 基础依赖：`matplotlib`、`pyyaml`
- extras：`anthropic`、`openai`、`bedrock`、`swe`、`mcp`、`skillbench`、`osworld`、`gepa`、`all`、`dev`
- wheel 中包含 `agent_evolve` 包和 `seed_workspaces`
- 使用 `hatchling` 构建
- ruff 目标版本为 py311，行宽 100

### `Makefile`

提供四个简单命令：

```makefile
install: pip install -e ".[all,dev]"
test:    pytest tests/ -v
lint:    ruff check agent_evolve/
fmt:     ruff format agent_evolve/
```

虽然 Makefile 中写的是 `pip`/`pytest`，结合仓库说明，实际在本仓库运行时应优先使用 `uv`，例如 `uv run pytest tests/ -v`。

### `.github/workflows/publish-pypi.yml`

GitHub Actions 发布流程。触发条件是 push `v*` tag，流程为安装 build 工具、构建 dist、上传 artifact、通过 `pypa/gh-action-pypi-publish` 发布到 PyPI。

### `CLAUDE.md`

面向 coding agent 的项目说明。当前内容偏 `arc-agi-3-dev` 分支，重点解释 ARC-AGI-3 agent、game loop、frame helper、LS20 测试结果等。它可看作本地协作说明，不是最终用户 API 文档。

## 4. `agent_evolve/` 核心包

`agent_evolve` 是项目主体，分为框架基础层、agent 适配层、benchmark 适配层、演化算法层和工具层。

```text
agent_evolve/
├── __init__.py
├── api.py
├── config.py
├── types.py
├── py.typed
├── protocol/
├── contract/
├── engine/
├── algorithms/
├── agents/
├── benchmarks/
├── llm/
└── utils/
```

### 4.1 包入口：`__init__.py`、`api.py`、`types.py`、`config.py`

#### `__init__.py`

导出公开 API：

- `Evolver`
- `EvolutionEngine`
- `EvolutionHistory`
- `TrialRunner`
- `BaseAgent`
- `BenchmarkAdapter`
- `AgentWorkspace`
- `Manifest`
- `EvolveConfig`
- `Task`、`Trajectory`、`Feedback`、`Observation`、`StepResult` 等数据类型

#### `api.py`

定义顶层 API `Evolver`。它负责把用户传入的 `agent`、`benchmark`、`config`、`engine` 解析成可运行对象，然后创建 `EvolutionLoop`。

关键逻辑：

- `_BENCHMARK_REGISTRY`：字符串 benchmark 名称到 adapter class 的映射。
- `_SEED_REGISTRY`：字符串 seed 名称到 `seed_workspaces/` 子目录的映射。
- `_resolve_agent()`：支持三类 agent 输入：
  - 已实例化的 `BaseAgent`
  - 已存在的 workspace 路径
  - 内置 seed workspace 名称
- `_resolve_workspace_path()`：会把 seed workspace 复制到 `work_dir`，避免直接改动 seed。
- 默认 engine 是 `agent_evolve.algorithms.skillforge.AEvolveEngine`。

需要注意：`api.py` 中部分 registry dotted path 与当前实际目录名存在潜在不一致，例如 registry 中出现 `agent_evolve.benchmarks.swe_verified.SweVerifiedBenchmark`，但当前目录实际有 `benchmarks/swe_verified_mini/`。这可能是历史兼容或尚未同步的入口，使用顶层 `Evolver(benchmark="swe-verified")` 前需要验证。

#### `types.py`

定义跨框架共享 dataclass：

- `Task`：benchmark 任务，包含 `id`、`input`、`metadata`
- `Trajectory`：agent 执行轨迹，包含 `output`、`steps`、`conversation`
- `Feedback`：benchmark 评价结果，包含 `success`、`score`、`detail`、`raw`
- `Observation`：把 task、trajectory、feedback 组合给 evolver
- `SkillMeta`：从 `SKILL.md` frontmatter 解析的技能元数据
- `StepResult`：一次演化步骤结果，包含 `mutated`、`summary`、`metadata`、`stop`
- `CycleRecord`、`EvolutionResult`：演化循环记录和最终摘要

#### `config.py`

定义 `EvolveConfig`：

- 批大小、最大 cycle、holdout 比例
- 可演化层开关：`evolve_prompts`、`evolve_skills`、`evolve_memory`、`evolve_tools`
- `trajectory_only` 模式
- evolver 模型和 token 上限
- EGL/convergence 参数
- `extra` 字典用于 benchmark 或算法专用参数

### 4.2 `protocol/`

```text
protocol/
├── __init__.py
└── base_agent.py
```

`base_agent.py` 定义所有可演化 Agent 的基类 `BaseAgent`。职责是处理文件系统契约：

- 初始化 `AgentWorkspace`
- 从 `prompts/system.md` 读取 system prompt
- 从 `skills/*/SKILL.md` 读取技能列表
- 从 `memory/*.jsonl` 读取记忆
- 可选加载 workspace 根目录的 `harness.py`
- 提供 `reload_from_fs()` 和 `export_to_fs()`
- 提供 `remember()` 和 `get_skill_content()`
- 强制子类实现 `solve(task: Task) -> Trajectory`

也就是说，具体 Agent 不需要自己实现通用 workspace 读取逻辑，只需要实现如何解决单个任务。

### 4.3 `contract/`

```text
contract/
├── __init__.py
├── manifest.py
├── schema.py
└── workspace.py
```

该目录实现 workspace 文件系统契约。

#### `manifest.py`

解析和保存 `manifest.yaml`，核心字段：

- `name`
- `version`
- `contract_version`
- `agent.entrypoint`
- `agent.type`
- `evolvable_layers`
- `reload_strategy`

`agent.entrypoint` 是 workspace 到具体 Agent class 的桥。

#### `workspace.py`

定义 `AgentWorkspace`，为 prompt、skill、tool、memory、harness、evolution metadata 提供 typed read/write 方法。

主要能力：

- `read_prompt()` / `write_prompt()`
- `read_fragment()` / `write_fragment()` / `list_fragments()`
- `list_skills()` / `read_skill()` / `write_skill()` / `delete_skill()`
- `list_drafts()` / `write_draft()` / `clear_drafts()`
- `read_tool_registry()` / `write_tool_registry()` / `read_tool()` / `write_tool()`
- `add_memory()` / `read_memories()` / `read_all_memories()`
- `read_harness()` / `write_harness()`
- `read_evolution_history()` / `read_evolution_metrics()`

`_parse_skill_frontmatter()` 从 `SKILL.md` YAML frontmatter 中解析 `name` 和 `description`。

#### `schema.py`

负责 workspace schema 校验，供 `api.py` 在解析 agent workspace 时调用。

### 4.4 `engine/`

```text
engine/
├── __init__.py
├── base.py
├── history.py
├── loop.py
├── observer.py
├── trial.py
└── versioning.py
```

该目录是演化循环的基础设施。

#### `base.py`

定义 `EvolutionEngine` 抽象类。所有算法实现必须实现：

```python
step(workspace, observations, history, trial) -> StepResult
```

它还提供：

- `manages_own_evaluation`：某些算法如 GEPA 自己调度评价时返回 True
- `on_cycle_end()`：cycle 结束回调

#### `loop.py`

定义 `EvolutionLoop`，把 Agent、Benchmark、Engine 串起来。

每个 cycle 的核心步骤：

1. 从 benchmark 获取 train tasks
2. 调用 `agent.solve(task)`
3. 调用 `benchmark.evaluate(task, trajectory)`
4. 收集 observation 到 `evolution/observations`
5. git snapshot：`pre-evo-N`
6. 调用 `engine.step(...)`
7. git snapshot：`evo-N`
8. 记录 cycle history 和 metrics
9. `agent.reload_from_fs()`
10. 检查 `StepResult.stop` 和收敛条件

如果 engine 的 `manages_own_evaluation=True`，loop 会跳过 solve/evaluate，把空 observations 传给 engine。

#### `trial.py`

定义 `TrialRunner`，供演化算法内部主动试跑任务：

- `run_tasks(tasks)`
- `run_single(task)`
- `get_tasks(split, limit)`
- 暴露只读 `agent` 和 `benchmark` property

#### `observer.py`

负责把 observations 写入演化目录，作为后续算法读取的证据。

#### `history.py`

提供对历史 observations、cycle record、版本信息的查询 facade。算法可用它判断是否改进、回滚到历史版本等。

#### `versioning.py`

负责 workspace 的 git 初始化、提交、tag、回滚等版本控制操作。README/DESIGN 中提到每次 mutation 会被打 tag，例如 `evo-1`、`evo-2`。

### 4.5 `llm/`

```text
llm/
├── __init__.py
├── anthropic.py
├── base.py
├── bedrock.py
└── openai.py
```

该目录抽象 evolver LLM provider。

#### `base.py`

定义：

- `LLMMessage`
- `LLMResponse`
- `LLMProvider`

Provider 必须实现：

- `complete(messages, max_tokens, temperature, **kwargs)`
- `complete_with_tools(messages, tools, max_tokens, **kwargs)`

#### `anthropic.py`、`bedrock.py`、`openai.py`

分别实现 Anthropic、AWS Bedrock、OpenAI provider。演化算法中需要调用 LLM 生成 skill、修改 prompt、做 judge 或 curator 时会使用这些 provider。

### 4.6 `utils/`

```text
utils/
├── __init__.py
├── logging.py
└── metrics.py
```

功能较轻：

- `logging.py`：日志初始化
- `metrics.py`：演化能力、学习曲线面积等指标函数

## 5. `agent_evolve/algorithms/` 演化算法

```text
algorithms/
├── __init__.py
├── adaptive_evolve/
├── adaptive_skill/
├── gepa/
├── guided_synth/
├── mas_adaptive_skill/
├── meta_harness/
├── propose_curate/
├── skillforge/
└── unified/
```

所有算法都围绕 `EvolutionEngine.step()` 接口实现，但策略不同。

### 5.1 `skillforge/`

```text
skillforge/
├── __init__.py
├── egl.py
├── engine.py
├── gating.py
├── prompts.py
└── tools.py
```

默认顶层 `Evolver` 使用的 engine 是 `skillforge.AEvolveEngine`。

功能定位：

- 面向 SkillsBench 的“solve-fail-evolve-retry”模式。
- 通过失败观察生成或更新 `skills/<name>/SKILL.md`。
- 支持技能预算，避免 skill library 无限制增长。
- `prompts.py` 构造演化 prompt、压缩轨迹、trajectory-only 指令等。
- `tools.py` 提供 evolver LLM 可用的 workspace bash 和默认 LLM 创建函数。
- `egl.py` 计算 Evolutionary Generality Loss，用于判断收敛。
- `gating.py` 定义 gating 策略接口。

### 5.2 `guided_synth/`

```text
guided_synth/
├── __init__.py
└── engine.py
```

功能定位：

- 面向 SWE-bench 的 guided synthesis。
- 不是让 evolver 从零发明技能，而是让 solver 在完成任务后提出 skill proposal，再由 LLM curator 判断 ACCEPT/MERGE/SKIP。
- 注重写入轻量 episodic memory 和 curate generalizable skills。
- 支持 verification focus，即只接受验证修复相关技能。

### 5.3 `adaptive_skill/`

```text
adaptive_skill/
├── __init__.py
├── egl.py
├── engine.py
├── gating.py
├── prompts.py
└── tools.py
```

功能定位：

- 轨迹优先、可在没有真实标签的情况下演化 skill library。
- 从 trajectory 中抽取行为信号，例如 tool calls、错误、重复命令、timeout、submit 情况。
- 可用 LLM judge 估计 trajectory 成功概率。
- 根据低分轨迹中的共性失败模式创建或精炼 skills。
- 用 `max_skills` 控制 skill 数量。

### 5.4 `adaptive_evolve/`

```text
adaptive_evolve/
├── README.md
├── __init__.py
├── analyzer.py
├── base_analysis.py
├── code_analysis.py
├── engine.py
└── prompts.py
```

功能定位：

- 更完整的多阶段分析式演化算法，尤其适合 MCP-Atlas 这种有 per-claim feedback 的任务。
- `base_analysis.py`：统计 pass/fail、工具错误、幻觉工具名、策略问题等。
- `code_analysis.py`：分析 code execution 使用情况和错失机会。
- `analyzer.py`：按 claim type、task type、judge feedback、failure pattern 做分层分析。
- `prompts.py`：构造 adaptive evolution prompt，并内置若干 auto-seed skill 模板。
- `engine.py`：串联分析、auto-correction、auto-seed、LLM-driven mutation、sanity check、meta-evolution。

典型会生成 `multi-requirement-handler`、`entity-verification`、`calculate-handler` 等针对性技能。

### 5.5 `gepa/`

```text
gepa/
├── __init__.py
├── engine.py
├── evaluator.py
└── serialization.py
```

功能定位：

- 把 A-Evolve workspace 序列化成 GEPA 的 candidate 格式，借助 GEPA 做 population-based prompt/skill/memory evolution。
- `GEPAEngine.manages_own_evaluation=True`，因此主 loop 会跳过常规 solve/evaluate。
- `serialization.py`：workspace 到 candidate 的序列化/恢复。
- `evaluator.py`：把 `TrialRunner` 包装成 GEPA evaluator，并生成 side info。
- `engine.py`：调用 GEPA 的 `optimize_anything()`，完成后把最佳 candidate 写回 workspace，并返回 `StepResult(stop=True)`。

### 5.6 `meta_harness/`

```text
meta_harness/
├── __init__.py
├── engine.py
└── prompts.py
```

功能定位：

- 面向 harness 本身的演化，不只改 prompt/skill/memory，还可改 workspace 根目录的 `harness.py`。
- 用 proposer prompt 产生候选 harness 更新，再评价、筛选、选择。
- 目录中的 `engine.py` 较大，包含候选生成、执行、评分、Pareto frontier 等逻辑。

### 5.7 `propose_curate/`

```text
propose_curate/
├── __init__.py
├── engine.py
└── prompts.py
```

功能定位：

- “提出-策展”式演化。
- 从失败或 trajectory 中提取提案，再由 curator 决定写入、合并或跳过。
- 和 `guided_synth` 思路接近，但更通用，文件中包含 proposal 解析、摘要格式化、内容截断等辅助逻辑。

### 5.8 `mas_adaptive_skill/`

```text
mas_adaptive_skill/
├── __init__.py
├── engine.py
├── orchestrator.py
└── prompts.py
```

功能定位：

- 面向 multi-agent system 的 adaptive skill 演化。
- `orchestrator.py` 组织一次演化 cycle。
- `prompts.py` 构造 MAS 相关批量数据和 prompt。

### 5.9 `unified/`

```text
unified/
├── __init__.py
├── controller.py
├── engine.py
├── interfaces.py
├── openai_compat.py
├── regimes.py
├── registry.py
├── types.py
├── operators/
├── readers/
└── verifiers/
```

功能定位：

- 这是一个“统一配方执行器”，把旧算法中的 reader/operator/verifier 原子化。
- `UnifiedEngine` 根据当前 regime 和 benchmark capability 选择一个 `Plan`，再按计划执行 readers、operators 和 verifier。
- 它避免直接 import legacy engine，使用 registry 注册机制发现原子组件。

核心子目录：

#### `unified/readers/`

从 observations/workspace/history 中读取证据：

- `pass_fail.py`：读 pass/fail 和 score
- `trajectory.py`：压缩一般 trajectory
- `terminal_trajectory.py`：压缩 Terminal-Bench trajectory
- `claim.py`、`claim_types.py`：提取 per-claim 信号
- `judge.py`：LLM judge 代理评分
- `proposal.py`：读取 solver proposal
- `draft.py`：读取 draft skills
- `patterns.py`：模式检测
- `score_curve.py`：读取分数曲线

#### `unified/operators/`

根据 evidence 修改 workspace：

- `write_episodic_memory.py`
- `fix_hallucinations.py`
- `auto_seed_skills.py`
- `llm_bash_evolve.py`
- `sanity_check.py`
- `prune_skills.py`
- `skill_curator.py`
- `terminal_skill_evolve.py`

#### `unified/verifiers/`

对 mutation 做验证：

- `no_verify.py`：无验证
- `stagnation_rollback.py`：停滞时回滚

#### `controller.py`

规则式调度器。它根据 capability、regime、config 选择 recipe，例如：

- SWE solver proposal -> `ProposalReader` + `SkillCurator`
- terminal profile -> `TerminalTrajectoryReader` + `LLMJudgeReader` + `TerminalSkillEvolve`
- per-claim feedback -> MCP 风格 recipe
- trajectory-only -> judge-backed recipe
- default -> `LLMBashEvolve`

## 6. `agent_evolve/agents/` Agent 实现

```text
agents/
├── __init__.py
├── arc/
├── mcp/
├── mcp_mh/
├── osworld/
├── skillbench/
├── swe/
└── terminal/
```

每个子目录实现一个或多个 `BaseAgent` 子类，负责具体 domain 的 solve 逻辑。

### 6.1 `agents/swe/`

```text
swe/
├── __init__.py
├── agent.py
├── conversation_manager.py
└── env.py
```

功能定位：

- SWE-bench 代码修复 Agent。
- `agent.py` 中的 `SweAgent` 继承 `BaseAgent`。
- `env.py` 管理 SWE-bench Docker container。
- `conversation_manager.py` 提供会话窗口管理，保留首条消息并滑动截断。

典型流程是启动 Docker 环境、加载 workspace tools、构造 system prompt、运行 agent loop、提取 patch、返回 `Trajectory`。

### 6.2 `agents/mcp/`

```text
mcp/
├── __init__.py
├── agent.py
├── code_executor.py
├── conversation_manager.py
├── docker_env.py
├── key_registry.py
├── mcp_client.py
├── server_keys.yaml
├── task_filter.py
└── tools.py
```

功能定位：

- MCP-Atlas Agent。
- `agent.py` 中 `McpAgent` 继承 `BaseAgent`。
- `docker_env.py` 管理 MCP-Atlas container。
- `mcp_client.py` 包装 MCP client。
- `key_registry.py` 管理 API key、secret redaction、错误分类。
- `tools.py` 把 MCP server tools 包装成 agent 可调用工具。
- `code_executor.py` 提供安全代码执行工具。
- `task_filter.py` 根据可用 key 过滤任务。

### 6.3 `agents/mcp_mh/`

```text
mcp_mh/
├── __init__.py
└── agent.py
```

功能定位：

- Meta-Harness 版本的 MCP Agent。
- `McpMHAgent` 继承自 `McpAgent`，配合 workspace 中的 `harness.py` 使用。

### 6.4 `agents/terminal/`

```text
terminal/
├── __init__.py
├── agent.py
├── dataset.py
├── docker_env.py
├── react_solver.py
└── tools.py
```

功能定位：

- Terminal-Bench 2.0 Agent。
- `agent.py` 中 `TerminalAgent` 继承 `BaseAgent`。
- `dataset.py` 下载/加载 challenge。
- `docker_env.py` 管理 TB2 container。
- `react_solver.py` 实现 ReAct 终端求解循环，并提取 trajectory、反思生成 skill。
- `tools.py` 提供 `bash`、`python`、`submit` 等工具包装。

### 6.5 `agents/skillbench/`

```text
skillbench/
├── __init__.py
├── __main__.py
├── agent.py
├── artifacts.py
├── backends.py
├── cli.py
├── dataset.py
├── docker_env.py
├── evolver.py
├── loop.py
├── paths.py
├── repo.py
├── tools.py
├── prompts/
└── official_terminus/
```

功能定位：

- SkillsBench Agent 和运行框架。
- `agent.py` 中 `SkillBenchAgent` 继承 `BaseAgent`。
- `backends.py` 包含 Native/Harbor 等执行后端，文件较大，是 SkillsBench 求解和验证的主要实现。
- `docker_env.py` 构建并管理 task Docker 容器，执行 verifier。
- `repo.py` 解析、校验和 bootstrap SkillsBench 仓库。
- `dataset.py` 解析任务。
- `tools.py` 为容器内命令、文件读写、技能加载提供工具。
- `evolver.py`、`loop.py` 提供 SkillsBench 专用演化循环包装。
- `official_terminus/` 保存官方 Terminus parser、prompt 和技能文档 loader。
- `artifacts.py` 负责导出 SkillBench 运行产物。

### 6.6 `agents/arc/`

```text
arc/
├── __init__.py
├── agent.py
├── basic_agent.py
├── bedrock_agent.py
├── bedrock_prompts.py
├── bedrock_tools.py
├── colors.py
├── frame.py
├── game_loop.py
├── grid_render.py
├── mas_agent.py
├── memories.py
├── orchestrator.py
├── repl.py
├── strands_agent.py
├── swarm.py
└── wiki.py
```

功能定位：

- ARC-AGI-3 游戏型 benchmark 的 Agent 实现。
- `agent.py`：主 `ArcAgent`。
- `basic_agent.py`：最小实现版本。
- `strands_agent.py`：基于 Strands SDK 的 ARC Agent。
- `mas_agent.py`：多 Agent 系统版本。
- `game_loop.py`：游戏循环，处理 reset/step/action。
- `frame.py`：Frame 抽象，提供 diff、render、颜色查找、bounding box 等。
- `grid_render.py`：把 grid 渲染为图片或 base64，支持多模态输入。
- `colors.py`：颜色 palette。
- `bedrock_agent.py`、`bedrock_prompts.py`、`bedrock_tools.py`：Bedrock 模型和工具调用相关实现。
- `orchestrator.py`、`swarm.py`：多子 Agent 协同。
- `memories.py`、`wiki.py`：游戏经验和 wiki 式知识。
- `repl.py`：持久 Python REPL，用于代码辅助分析。

### 6.7 `agents/osworld/`

```text
osworld/
├── __init__.py
└── react_solver.py
```

功能定位：

- OSWorld GUI 任务的 ReAct solver。
- 处理 screenshot、accessibility tree、computer_use action parsing、conversation extraction 等。
- 与 `examples/osworld_examples/` 中的脚本配合使用。

## 7. `agent_evolve/benchmarks/` Benchmark 适配器

```text
benchmarks/
├── __init__.py
├── base.py
├── cl_bench.py
├── skill_bench.py
├── arc_agi3/
├── mcp_atlas/
├── skillbench/
├── swe_verified_mini/
└── tb2/
```

所有 benchmark adapter 都围绕 `BenchmarkAdapter` 接口：

```python
get_tasks(split: str = "train", limit: int = 10) -> list[Task]
evaluate(task: Task, trajectory: Trajectory) -> Feedback
```

### `base.py`

定义 `BenchmarkAdapter` 抽象基类。

### `mcp_atlas/`

```text
mcp_atlas/
├── __init__.py
├── mcp-atlas.md
└── mcp_atlas.py
```

MCP-Atlas benchmark adapter。负责加载 MCP tasks、调用 judge/evaluator、返回 per-task feedback。

### `swe_verified_mini/`

```text
swe_verified_mini/
├── __init__.py
└── benchmark.py
```

SWE-bench Verified mini adapter。包含 SWEbench instance 构建、Docker image 映射、swebench grader 调用、结果字段解析等逻辑。

### `tb2/`

```text
tb2/
├── README.md
├── download_challenges.sh
└── terminal2.py
```

Terminal-Bench 2.0 adapter，配合 challenge 下载脚本使用。

### `skillbench/`

```text
skillbench/
├── __init__.py
└── skill_bench.py
```

SkillsBench benchmark adapter。

### `arc_agi3/`

```text
arc_agi3/
├── __init__.py
└── benchmark.py
```

ARC-AGI-3 benchmark adapter。包含 `ArcAgi3Benchmark` 和 `GameResult`，负责游戏任务加载和 RHAE 等评价。

### `cl_bench.py`

CL Bench adapter 和相关 skill distillation/selection 逻辑。文件较大，包含 Bedrock 调用、JSONL 读写、rubric 构造、skill schema、embedding/ranking、LLM skill selection、skill guidance 生成等。

### `skill_bench.py`

旧路径或兼容入口，和 `benchmarks/skillbench/skill_bench.py` 并存。使用时需要确认实际 import path。

## 8. `seed_workspaces/` 初始工作区

```text
seed_workspaces/
├── arc/
├── arc-mas/
├── mcp/
├── mcp_mh/
├── osworld/
├── skillbench/
├── swe/
└── terminal/
```

这些目录是内置初始 Agent 工作区。`api.py` 会根据 seed 名称复制这些目录到 `work_dir`，之后演化算法只修改副本。

每个 workspace 通常包含：

```text
manifest.yaml
prompts/system.md
skills/*/SKILL.md
tools/registry.yaml
memory/*.jsonl
harness.py             # 仅部分 workspace 有
```

### `seed_workspaces/swe/`

```text
swe/
├── manifest.yaml
├── prompts/system.md
└── tools/
    ├── bash.py
    ├── python_exec.py
    ├── registry.yaml
    ├── sequentialthinking.py
    ├── submit.py
    └── text_editor.py
```

入口 Agent：`agent_evolve.agents.swe.agent.SweAgent`。

定位：SWE-bench 代码修复 Agent 的初始 prompt 和工具集。

### `seed_workspaces/mcp/`

```text
mcp/
├── manifest.yaml
├── memory/memories.jsonl
├── prompts/system.md
└── tools/registry.yaml
```

入口 Agent：`agent_evolve.agents.mcp.agent.McpAgent`。

定位：MCP-Atlas 工具调用 Agent 的初始工作区，可演化 prompts、skills、tools、memory。

### `seed_workspaces/mcp_mh/`

```text
mcp_mh/
├── .gitignore
├── CLAUDE.md
├── harness.py
├── manifest.yaml
├── memory/memories.jsonl
├── prompts/system.md
└── tools/registry.yaml
```

入口 Agent：`agent_evolve.agents.mcp_mh.agent.McpMHAgent`。

定位：用于 Meta-Harness 实验的 MCP 工作区，比普通 MCP 多了可加载的 `harness.py`。

### `seed_workspaces/skillbench/`

```text
skillbench/
├── manifest.yaml
├── memory/memories.jsonl
├── prompts/system.md
└── skills/
    ├── data-formats/SKILL.md
    ├── environment-discovery/SKILL.md
    ├── python-packages/SKILL.md
    └── skill-usage/SKILL.md
```

入口 Agent：`agent_evolve.agents.skillbench.agent.SkillBenchAgent`。

定位：SkillsBench 的初始技能库和 prompt。

### `seed_workspaces/terminal/`

```text
terminal/
├── manifest.yaml
├── memory/memories.jsonl
├── prompts/system.md
├── skills/
│   ├── build-compiled-extensions/SKILL.md
│   ├── debug-and-fix/SKILL.md
│   ├── environment-discovery/SKILL.md
│   ├── scientific-computing/SKILL.md
│   └── self-verification/SKILL.md
└── tools/
    ├── bash.py
    ├── python.py
    ├── registry.yaml
    └── submit.py
```

入口 Agent：`agent_evolve.agents.terminal.agent.TerminalAgent`。

定位：Terminal-Bench 2.0 的终端工具和通用任务技能。

### `seed_workspaces/arc/`

```text
arc/
├── manifest.yaml
└── prompts/system.md
```

入口 Agent：`agent_evolve.agents.arc.agent.ArcAgent`。

定位：单 Agent ARC-AGI-3 初始工作区。

### `seed_workspaces/arc-mas/`

```text
arc-mas/
├── manifest.yaml
├── prompts/
│   ├── explorer.md
│   ├── game_reference.md
│   ├── solver.md
│   ├── system.md
│   └── theorist.md
└── tools/
    ├── explorer.yaml
    ├── orchestrator.yaml
    ├── solver.yaml
    └── theorist.yaml
```

入口 Agent：`agent_evolve.agents.arc.mas_agent.MASArcAgent`。

定位：ARC-AGI-3 多 Agent 系统工作区，分 explorer/solver/theorist/orchestrator prompt 与工具配置。

### `seed_workspaces/osworld/`

```text
osworld/
└── skills/
    ├── application-shortcuts/SKILL.md
    ├── gui-click-over-keyboard/SKILL.md
    ├── gui-navigation/SKILL.md
    ├── screenshot-analysis/SKILL.md
    ├── self-verification/SKILL.md
    └── web-access-strategies/SKILL.md
```

定位：OSWorld GUI 任务的初始技能库。当前 maxdepth 读取中未看到 manifest，说明它可能是示例脚本直接使用的技能集合，而不是完整 `Evolver` seed workspace。

## 9. `examples/` 示例与实验入口

```text
examples/
├── arc_examples/
├── cl_bench_examples/
├── configs/
├── harness-disentangling/
├── mcp_examples/
├── osworld_examples/
├── skillbench_examples/
├── swe_examples/
└── tb_examples/
```

该目录不是核心库，而是运行实验的脚本集合。大部分脚本依赖外部 benchmark 数据、Docker、模型 API key 或云资源。

### `examples/swe_examples/`

```text
swe_examples/
├── README.md
├── evolve_sequential.py
├── evolve_sequential_split_unified.py
├── evolve_sequential_unified.py
├── run_solve_all.sh
├── run_swe_evolve_in-situ_unified.sh
├── run_swe_evolve_split_unified.sh
└── solve_all.py
```

功能：

- `solve_all.py`：无演化 baseline，批量求解 SWE-bench。
- `evolve_sequential.py`：按 batch 演化 workspace。
- `*_unified.py`：使用 `UnifiedEngine` 的新实验入口。
- shell 脚本封装常用参数。

README 中明确提示 Python 命令使用 `uv run`。

### `examples/mcp_examples/`

```text
mcp_examples/
├── adaptive_evolve_all.py
├── adaptive_evolve_baseline.py
├── run_adaptive_evolve_all_split_unified.py
├── run_adaptive_evolve_all_unified.py
├── run_adaptive_evolve_baseline.sh
├── run_adaptive_evolve_in-situ_unified.sh
├── run_adaptive_evolve_split_unified.sh
├── run_final_eval.py
├── run_metaharness.py
└── run_metaharness.sh
```

功能：

- MCP-Atlas baseline、adaptive evolve、final eval、meta-harness 实验。
- unified 版本脚本用于 harness-disentangling 等实验。

### `examples/tb_examples/`

```text
tb_examples/
├── README.md
├── batch_evolve_terminal.py
├── run_baseline.sh
└── run_evolution.sh
```

功能：

- Terminal-Bench 2.0 baseline 和两阶段 evolution/evaluation。
- `batch_evolve_terminal.py` 是核心脚本。
- README 说明 `--trajectory-only`、`--skills-only`、`--protect-skills` 等演化约束。

### `examples/skillbench_examples/`

```text
skillbench_examples/
├── run_skillbench_evolve_in-situ_unified.sh
├── run_skillbench_evolve_in_situ_cycle.sh
├── run_skillbench_evolve_in_situ_cycle_unified.sh
├── run_skillbench_evolve_split_unified.sh
├── run_skillbench_solve_all.sh
├── skillbench_evolve_in_situ_cycle.py
├── skillbench_evolve_in_situ_cycle_unified.py
├── skillbench_evolve_split_unified.py
└── skillbench_solve_one.py
```

功能：

- SkillsBench 单任务求解、in-situ cycle、split/unified 演化脚本。
- 与 `agents/skillbench` 和 `benchmarks/skillbench` 配套。

### `examples/arc_examples/`

```text
arc_examples/
├── eval_2games.py
├── eval_mas.py
├── eval_results.json
├── evolve_arc.py
├── mas_dashboard.html
├── play_ls20.py
├── serve_mas_dashboard.py
├── submit_competition.py
└── visualize_replay.py
```

功能：

- ARC-AGI-3 的本地测试、演化、评估、可视化和比赛提交。
- `play_ls20.py` 用于 LS20 实时显示。
- `eval_mas.py` 和 `mas_dashboard.html` 服务多 Agent 评估与可视化。

### `examples/osworld_examples/`

```text
osworld_examples/
├── README.md
├── evolve_osworld.py
├── evolve_osworld_engine.py
└── run_osworld.sh
```

功能：

- OSWorld GUI benchmark 实验。
- README 说明需要 OSWorld 仓库、AWS EC2、VNC/NoVNC、Bedrock 权限等。
- 适合大规模 GUI task 演化实验。

### `examples/cl_bench_examples/`

```text
cl_bench_examples/
├── evolve_cl_bench.py
├── evolve_cl_bench_engine.py
├── group_by_context.py
└── run_cl_bench.sh
```

功能：

- CL Bench 演化脚本。
- 包含按 context 分组、专用 engine、运行 shell。

### `examples/configs/`

```text
configs/
├── hle.yaml
├── mcp.yaml
├── metaharness_mcp.yaml
├── skillbench.yaml
├── swe.yaml
└── terminal.yaml
```

功能：

- 各 benchmark/实验的 YAML 配置。
- 可由 `EvolveConfig.from_yaml()` 读取，未知字段进入 `extra`。

### `examples/harness-disentangling/`

```text
harness-disentangling/
├── README.md
├── _region_picker.py
├── model_region_availability.json
├── run_exp0_unified_insitu.py
├── run_exp1.py
├── run_exp1_unified_insitu.py
├── assets/
├── hfr_analysis/
└── scripts/
```

功能：

- 论文实验 artifact：`Harness Updating Is Not Harness Benefit`。
- Exp0：固定 solver，变化 evolver，测 harness-updating。
- Exp1：固定 evolver，变化 solver，测 harness-benefit。
- `hfr_analysis/`：Harness-Following Rate 诊断流水线。
- `model_region_availability.json`：模型昵称到 provider/region 的映射。
- `scripts/`：单 seed sweep 和 status helper。

## 10. `docs/` 文档

```text
docs/
├── algorithms/
│   ├── adaptive-evolve.md
│   ├── adaptive-skill.md
│   ├── gepa.md
│   ├── guided-synth.md
│   └── skillforge.md
├── mcp-atlas-demo.md
└── skillbench-setup.md
```

功能：

- `docs/algorithms/`：各演化算法的设计、流程、参数和用法。
- `mcp-atlas-demo.md`：MCP-Atlas 演示说明。
- `skillbench-setup.md`：SkillsBench 安装和数据准备说明。

这些文档和 `agent_evolve/algorithms/` 一一对应，是理解算法意图的主要来源。

## 11. `artifacts/` 实验产物

```text
artifacts/
├── mcp_mh_opus46/
└── tb2_clawcode_opus46/
```

该目录保存已经完成的演化结果、报告和可复用 workspace 片段，不属于核心库代码。

### `artifacts/mcp_mh_opus46/`

```text
mcp_mh_opus46/
├── REPORT.md
├── baseline_eval_5trials.json
├── final_eval_5trials.json
├── harness.py
├── memory/memories.jsonl
├── prompts/system.md
├── results.json
└── tools/registry.yaml
```

功能：

- MCP Meta-Harness + Opus 4.6 的实验结果。
- 包含 baseline/final 评估 JSON、最终 harness、memory、prompt、tool registry 和报告。

### `artifacts/tb2_clawcode_opus46/`

```text
tb2_clawcode_opus46/
├── TB2_README.md
├── prompt.md
└── skills/
    ├── build-compiled-extensions/SKILL.md
    ├── debug-and-fix/SKILL.md
    ├── environment-discovery/SKILL.md
    ├── python-data-analysis/SKILL.md
    ├── scientific-computing/SKILL.md
    ├── self-verification/SKILL.md
    └── systematic-exploration/SKILL.md
```

功能：

- Terminal-Bench 2.0 + ClawCode/Opus 4.6 的演化产物。
- 保存最终 prompt 和技能库，可作为后续 seed 或分析材料。

## 12. `figs/` 图片资源

```text
figs/
├── A-EVOLVE-FRAMEWORK.pdf
├── A-EVOLVE-FRAMEWORK.png
├── a_evolve_benchmarks.pdf
├── a_evolve_benchmarks.png
├── figure.py
└── teaser.png
```

功能：

- README 和论文/展示使用的图。
- `figure.py` 是绘图脚本。
- `teaser.png`、framework 和 benchmarks 图用于项目介绍。

## 13. `tests/` 测试

```text
tests/
├── gepa/
├── test_mas_agent.py
├── test_skillbench_setup.py
└── test_unified_engine.py
```

### `tests/gepa/`

覆盖 GEPA 集成：

- candidate serialization/restore
- evaluator side info 和 trajectory compression
- `GEPAEngine.manages_own_evaluation`
- loop 在 engine `stop=True` 时提前结束
- `TrialRunner` property
- `StepResult.stop`

### `test_unified_engine.py`

覆盖 `UnifiedEngine` 的基础执行逻辑，使用 fake agent 和 fake benchmark。

### `test_skillbench_setup.py`

覆盖 SkillsBench 路径解析、repo bootstrap、路径校验、benchmark 任务加载、seed workspace root 等。

### `test_mas_agent.py`

覆盖 ARC multi-agent agent 行为，包括 skills loading、max actions、thinking effort、subagent artifacts、shared knowledge、reload 等。

## 14. 关键数据流与目录之间的关系

### 14.1 运行入口

常见入口有两类：

1. 高层 API：

```python
import agent_evolve as ae
evolver = ae.Evolver(agent="swe", benchmark="swe-verified")
result = evolver.run(cycles=10)
```

2. benchmark 专用脚本：

```text
examples/swe_examples/evolve_sequential.py
examples/mcp_examples/adaptive_evolve_all.py
examples/tb_examples/batch_evolve_terminal.py
examples/skillbench_examples/*.py
examples/arc_examples/evolve_arc.py
```

### 14.2 Workspace 复制和演化

```text
seed_workspaces/<domain>/
        |
        | copied by Evolver/scripts
        v
work_dir/<domain>/
        |
        | Agent reads prompts/skills/tools/memory
        | Engine writes prompts/skills/tools/memory/harness
        v
evolution/
  observations/
  history.jsonl
  metrics.json
  git tags: pre-evo-N, evo-N
```

### 14.3 Agent/Benchmark/Engine 依赖方向

```text
BenchmarkAdapter.get_tasks()
        |
        v
Task -> BaseAgent.solve() -> Trajectory
        |
        v
BenchmarkAdapter.evaluate() -> Feedback
        |
        v
Observation -> EvolutionEngine.step()
        |
        v
AgentWorkspace writes -> Agent.reload_from_fs()
```

框架设计上，Agent 和 Engine 不直接调用彼此内部方法，它们通过 workspace 文件和 observation 数据交互。

## 15. 代码结构中的注意点

1. `a-evolve` 是嵌套 Git 仓库  
   顶层存在 `.git/`，说明它可能是主仓库 `OR-Claw` 中的子仓库或独立 clone。分析、提交或状态检查时要注意路径边界。

2. `api.py` registry 与当前文件布局可能存在历史差异  
   例如 registry 中存在 `swe_verified`、`terminal2` 等路径，而当前实际目录包含 `swe_verified_mini/` 和 `tb2/`。一些脚本可能绕过顶层 registry 直接 import 正确模块；使用统一 API 前应运行最小 smoke test。

3. `seed_workspaces/osworld` 看起来不是完整 workspace  
   它只有 skills，没有看到 `manifest.yaml`。更像 OSWorld 示例的技能资源目录，而不是 `Evolver(agent="osworld")` 可直接解析的 seed。

4. 算法实现存在 legacy 与 unified 两套路线  
   `adaptive_evolve`、`adaptive_skill`、`guided_synth`、`skillforge` 是较具体的 legacy engine；`unified` 把这些能力拆成 reader/operator/verifier 组合，用 rule-based controller 选择 recipe。

5. 多数实验脚本依赖外部资源  
   SWE/MCP/TB/SkillBench/OSWorld 运行通常需要 Docker、benchmark 数据、模型 API、AWS/Bedrock 或 provider 配置。静态结构分析不等同于所有实验可以直接跑通。

6. 本项目建议使用 uv  
   仓库说明要求后续运行 Python 代码或安装依赖使用 uv 环境，例如：

```bash
uv run pytest tests/ -v
uv pip install -e ".[all,dev]"
```

## 16. 快速索引

| 目标 | 主要文件/目录 |
|---|---|
| 了解项目总体 | `README.md`、`DESIGN.md`、`QUICKSTART.md` |
| 使用顶层 API | `agent_evolve/api.py`、`agent_evolve/__init__.py` |
| 新增 Agent | `agent_evolve/protocol/base_agent.py`、`seed_workspaces/*/manifest.yaml` |
| 新增 Benchmark | `agent_evolve/benchmarks/base.py` |
| 新增演化算法 | `agent_evolve/engine/base.py`、`agent_evolve/algorithms/*/engine.py` |
| 修改 workspace contract | `agent_evolve/contract/workspace.py`、`manifest.py`、`schema.py` |
| SWE 实验 | `agents/swe/`、`benchmarks/swe_verified_mini/`、`examples/swe_examples/`、`seed_workspaces/swe/` |
| MCP 实验 | `agents/mcp/`、`benchmarks/mcp_atlas/`、`examples/mcp_examples/`、`seed_workspaces/mcp/` |
| TB2 实验 | `agents/terminal/`、`benchmarks/tb2/`、`examples/tb_examples/`、`seed_workspaces/terminal/` |
| SkillsBench 实验 | `agents/skillbench/`、`benchmarks/skillbench/`、`examples/skillbench_examples/`、`seed_workspaces/skillbench/` |
| ARC-AGI-3 实验 | `agents/arc/`、`benchmarks/arc_agi3/`、`examples/arc_examples/`、`seed_workspaces/arc*` |
| UnifiedEngine | `agent_evolve/algorithms/unified/`、`tests/test_unified_engine.py` |
| GEPA 集成 | `agent_evolve/algorithms/gepa/`、`tests/gepa/` |
| 已完成产物 | `artifacts/` |

