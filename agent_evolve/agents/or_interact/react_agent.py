"""A-Evolve wrapper around the OR-Claw ReAct baseline."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import signal
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from baseline.react.agent import AgentResult, ReActAgent, tool_schemas
from baseline.react.config import ReactConfig, load_config
from baseline.react.trace import TraceWriter
from baseline.heuristic.tools import RUN_HEURISTIC_DESCRIPTION, run_heuristic
from tools.seed_tools._common import configure_environment
from tools.tools_registery import ToolRegistry, ToolSpec, register_seed_tools

from ...protocol.base_agent import BaseAgent
from ...types import Task, Trajectory


FORBIDDEN_TOOL_STRINGS = (
    "oracle",
    "objective.json",
    "reference_solution",
    "reference_formulation",
    "or-interact-bench",
    "http://",
    "https://",
    "urllib",
    "requests",
    "socket",
    "subprocess",
    "os.system",
    "popen",
)
TASK_CATEGORY_ENV = "OR_INTERACT_TASK_CATEGORY"
OR_INTERACT_SETTINGS_FILE = "or_interact_settings.json"
RESERVED_DYNAMIC_TOOL_NAMES = {"run_heuristic"}
HEURISTIC_PROMPT_EXTENSION = """\
## Heuristic Algorithm Evolution

This run enables the `run_heuristic` tool. The agent may execute heuristic,
greedy, simulation, local-search, approximation, or metaheuristic Python code
when exact solver modeling is difficult. Evolution may improve prompts, skills,
memory, or tools that help design, validate, and refine such heuristic
algorithms, while still requiring final answers to be submitted with
`finalize`.

`run_heuristic` is a built-in runtime tool supplied by the harness. Do not
create, modify, or register workspace files named `tools/run_heuristic.py`.
"""


class ORReactAgent(BaseAgent):
    """Reloadable a-evolve agent that runs the OR-Claw ReAct baseline."""

    def __init__(self, workspace_dir: str | Path):
        self.or_interact_settings: dict[str, Any] = {}
        super().__init__(workspace_dir)
        self.config = self._load_config()

    def reload_from_fs(self) -> None:
        super().reload_from_fs()
        self.or_interact_settings = self._load_or_interact_settings()
        self.registry = self._build_registry()

    def solve(self, task: Task) -> Trajectory:
        runtime_dir, trace = self.start_task_run(task)
        start = time.monotonic()
        result = self.run_phase(task, runtime_dir=runtime_dir, trace=trace)
        elapsed = time.monotonic() - start
        return self.finish_task_run(
            task,
            runtime_dir=runtime_dir,
            result=result,
            elapsed=elapsed,
        )

    def start_task_run(self, task: Task) -> tuple[Path, TraceWriter]:
        task_dir = Path(task.metadata["task_dir"]).resolve()
        runtime_dir = self._prepare_runtime_dir(task.id)
        configure_environment(context_dir=task_dir, runtime_dir=runtime_dir)
        return runtime_dir, TraceWriter(runtime_dir)

    def run_phase(
        self,
        task: Task,
        *,
        runtime_dir: Path,
        trace: TraceWriter,
        phase: str = "solve",
        initial_messages: list[dict[str, Any]] | None = None,
        user_message: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        stop_after_tools: set[str] | None = None,
    ) -> AgentResult:
        task_dir = Path(task.metadata["task_dir"]).resolve()
        configure_environment(context_dir=task_dir, runtime_dir=runtime_dir)
        registry = self._build_registry(include_type_router=phase.endswith(":route"))
        self.registry = registry
        try:
            with _task_timeout(self.config.task_timeout_seconds):
                with _task_category_env(task.metadata.get("category")):
                    return ReActAgent(
                        config=self.config,
                        trace=trace,
                        registry=registry,
                        system_prompt=system_prompt or self._build_system_prompt(),
                    ).run(
                        initial_messages=initial_messages,
                        user_message=user_message,
                        phase=phase,
                        max_turns=max_turns,
                        stop_after_tools=stop_after_tools,
                    )
        except TimeoutError as exc:
            error = str(exc)
            trace.event("error", {"message": error})
            return AgentResult(
                status="task_timeout",
                turns=0,
                error=error,
                messages=initial_messages or [],
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trace.event("error", {"message": error})
            return AgentResult(
                status="error",
                turns=0,
                error=error,
                messages=initial_messages or [],
            )

    def finish_task_run(
        self,
        task: Task,
        *,
        runtime_dir: Path,
        result: AgentResult,
        elapsed: float,
    ) -> Trajectory:
        summary = {
            "task_id": task.id,
            "dataset": task.metadata.get("dataset"),
            "model": self.config.model,
            "status": result.status,
            "turns": result.turns,
            "objective": result.objective_value,
            "error": result.error,
            "elapsed_seconds": elapsed,
        }
        (runtime_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        steps = self._trace_steps(runtime_dir)
        steps.append(
            {
                "type": "run_summary",
                "runtime_dir": str(runtime_dir),
                "status": result.status,
                "turns": result.turns,
                "objective": result.objective_value,
                "error": result.error,
                "tools": self.registry.list_tools(),
            }
        )
        return Trajectory(
            task_id=task.id,
            output=json.dumps(summary, ensure_ascii=False),
            steps=steps,
            conversation=steps,
        )

    def _load_config(self) -> ReactConfig:
        config_path = os.environ.get("OR_REACT_CONFIG")
        default_config = Path.cwd() / "config" / "react.yaml"
        if config_path:
            base = load_config(config_path)
        elif default_config.is_file():
            base = load_config(default_config)
        else:
            base = ReactConfig(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                model=os.environ.get("OR_REACT_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
                temperature=0.0,
                max_turns=int(os.environ.get("OR_REACT_MAX_TURNS", "16")),
                task_timeout_seconds=int(os.environ.get("OR_REACT_TASK_TIMEOUT_SECONDS", "600")),
                parallelism=int(os.environ.get("OR_REACT_PARALLELISM", "1")),
                benchmark_dir=(Path.cwd() / "OR-Interact-Bench").resolve(),
                results_dir=self.workspace.root / "evolution" / "runs",
            )

        results_dir = os.environ.get("OR_REACT_RESULTS_DIR")
        return ReactConfig(
            api_key=os.environ.get("OR_REACT_API_KEY", base.api_key),
            base_url=os.environ.get("OR_REACT_BASE_URL", base.base_url),
            model=os.environ.get("OR_REACT_MODEL", base.model),
            temperature=_resolve_temperature(base.temperature),
            max_turns=int(os.environ.get("OR_REACT_MAX_TURNS", str(base.max_turns))),
            task_timeout_seconds=int(
                os.environ.get("OR_REACT_TASK_TIMEOUT_SECONDS", str(base.task_timeout_seconds))
            ),
            parallelism=int(os.environ.get("OR_REACT_PARALLELISM", str(base.parallelism))),
            benchmark_dir=base.benchmark_dir,
            results_dir=Path(results_dir).expanduser().resolve()
            if results_dir
            else self.workspace.root / "evolution" / "runs",
        )

    def _build_system_prompt(self) -> str:
        hook = self.harness_hook("build_system_prompt")
        if hook:
            return hook(self.system_prompt, self.skills, self.memories, self.registry)

        sections = [self.system_prompt.strip()]
        if self._heuristic_enabled():
            sections.append(HEURISTIC_PROMPT_EXTENSION.strip())

        skill_catalog = self._skill_catalog()
        if skill_catalog:
            sections.append("## Evolved Skill Catalog\n" + skill_catalog)

        memory_catalog = self._memory_catalog()
        if memory_catalog:
            sections.append("## Evolved Memory Catalog\n" + memory_catalog)

        tool_lines = []
        for spec in self.registry.as_dict().values():
            description = " ".join(spec.description.split())
            tool_lines.append(f"- {spec.name}: {description[:300]}")
        sections.append("## Available Tools\n" + "\n".join(tool_lines))
        return "\n\n".join(section for section in sections if section)

    def _memory_catalog(self) -> str:
        lines = []
        for index, memory in enumerate(self.memories[-20:], start=1):
            category = memory.get("_category", "memory")
            content = " ".join(str(memory.get("content") or "").split())
            if not content:
                continue
            lines.append(
                f"- memory:{index} category={category}; content={content[:300]}"
            )
        return "\n".join(lines)

    def _skill_catalog(self) -> str:
        lines = []
        for skill in self.skills:
            description = " ".join(skill.description.split())
            lines.append(
                f"- {skill.path or skill.name}: {skill.name}; description={description}"
            )
        return "\n".join(lines)

    def _build_registry(self, *, include_type_router: bool = False) -> ToolRegistry:
        registry = register_seed_tools(ToolRegistry())
        if self._heuristic_enabled():
            registry.add_tool(
                name="run_heuristic",
                function=run_heuristic,
                description=RUN_HEURISTIC_DESCRIPTION,
                metadata={"source": "or_interact_settings"},
            )
        for entry in self.workspace.read_tool_registry():
            if entry.get("kind") == "seed":
                continue
            name = str(entry.get("name") or "")
            if name == "type_router" and not include_type_router:
                continue
            if name == "answer_checker":
                continue
            if name in RESERVED_DYNAMIC_TOOL_NAMES:
                continue
            if not _is_evolved_tool_entry(entry):
                continue
            spec = self._load_evolved_tool(entry)
            registry.register(spec, overwrite=True)
        return registry

    def _load_or_interact_settings(self) -> dict[str, Any]:
        path = self.workspace.root / OR_INTERACT_SETTINGS_FILE
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _heuristic_enabled(self) -> bool:
        return bool(self.or_interact_settings.get("enable_heuristic_tool"))

    def _load_evolved_tool(self, entry: dict[str, Any]) -> ToolSpec:
        name = str(entry["name"])
        function_name = str(entry.get("function") or name)
        file_name = str(entry.get("file") or f"{name}.py")
        tool_path = _resolve_workspace_tool_path(self.workspace.tools_dir, file_name)
        _validate_tool_source(tool_path)

        module_name = f"or_interact_tool_{name}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, tool_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load evolved tool: {tool_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        function = getattr(module, function_name, None)
        if not callable(function):
            raise ValueError(f"evolved tool {name!r} missing callable {function_name!r}")
        _validate_tool_function(name, function)
        description = str(entry.get("description") or inspect.getdoc(function) or "")
        return ToolSpec(
            name=name,
            function=function,
            description=description,
            kind="evolved",
            metadata={"file": file_name, "function": function_name},
        )

    def _prepare_runtime_dir(self, task_id: str) -> Path:
        runtime_dir = self.config.results_dir / f"{task_id}-{uuid.uuid4().hex[:8]}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "trace.md",
            "trace.jsonl",
            "final_code.py",
            "submitted_answer.csv",
            "run_summary.json",
        ):
            (runtime_dir / filename).unlink(missing_ok=True)
        return runtime_dir

    def _trace_steps(self, runtime_dir: Path) -> list[dict[str, Any]]:
        trace_path = runtime_dir / "trace.jsonl"
        if not trace_path.is_file():
            return []
        steps = []
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    steps.append(json.loads(line))
        return steps

    def tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(self.registry)


def _is_evolved_tool_entry(entry: dict[str, Any]) -> bool:
    return bool(entry.get("name")) and (bool(entry.get("file")) or bool(entry.get("module")))


@contextmanager
def _task_category_env(category: Any):
    previous = os.environ.get(TASK_CATEGORY_ENV)
    text = str(category or "").strip()
    if text:
        os.environ[TASK_CATEGORY_ENV] = text
    else:
        os.environ.pop(TASK_CATEGORY_ENV, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TASK_CATEGORY_ENV, None)
        else:
            os.environ[TASK_CATEGORY_ENV] = previous


def _resolve_temperature(base_temperature: float | None) -> float | None:
    raw = os.environ.get("OR_REACT_TEMPERATURE")
    if raw is None:
        return base_temperature
    text = raw.strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(raw)


def _resolve_workspace_tool_path(tools_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"evolved tool file must be workspace-local: {file_name}")
    resolved = (tools_dir / path).resolve()
    tools_root = tools_dir.resolve()
    if not (resolved == tools_root or tools_root in resolved.parents):
        raise ValueError(f"evolved tool file escapes workspace tools dir: {file_name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"evolved tool file not found: {resolved}")
    return resolved


def _validate_tool_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in FORBIDDEN_TOOL_STRINGS:
        if forbidden in lowered:
            raise ValueError(f"evolved tool {path.name} contains forbidden string: {forbidden}")


def _validate_tool_function(name: str, function: Any) -> None:
    signature = inspect.signature(function)
    if any(
        param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}
        for param in signature.parameters.values()
    ):
        raise ValueError(f"evolved tool {name!r} cannot use *args or **kwargs")
    hints = get_type_hints(function)
    return_annotation = hints.get("return", signature.return_annotation)
    if return_annotation is inspect.Signature.empty:
        raise ValueError(f"evolved tool {name!r} must annotate return type as dict[str, Any]")
    origin = get_origin(return_annotation)
    if return_annotation is not dict and origin is not dict:
        raise ValueError(f"evolved tool {name!r} must return dict[str, Any]")
    if origin is dict:
        args = get_args(return_annotation)
        if args and args[0] is not str:
            raise ValueError(f"evolved tool {name!r} must use string keys")


@contextmanager
def _task_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"task exceeded timeout of {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_alarm = signal.alarm(0)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_alarm > 0:
            signal.alarm(previous_alarm)
