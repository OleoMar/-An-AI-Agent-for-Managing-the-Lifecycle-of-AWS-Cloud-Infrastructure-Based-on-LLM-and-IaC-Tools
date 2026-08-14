"""
evaluation/benchmark_runner.py

Запускает агента на всех сценариях и собирает метрики.

Использование:
  python -m evaluation.benchmark_runner                    # все сценарии
  python -m evaluation.benchmark_runner --scenario 001    # один сценарий
  python -m evaluation.benchmark_runner --dry-run         # без реального AWS

Формат сценария (evaluation/scenarios/XXX.json):
  {
    "id":          "scenario-001",
    "category":    "simple | networking | database | serverless | security",
    "input":       "Create an S3 bucket for user photos",
    "description": "что проверяет этот сценарий",
    "acceptance": {
      "min_resources":   1,
      "required_types":  ["aws_s3_bucket"],
      "should_ask_clarification": false
    }
  }
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import ScenarioResult, BenchmarkMetrics, compute_metrics
from config import EVAL_SCENARIOS_DIR, EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)


# ── Загрузка сценариев ────────────────────────────────────────────────────────

def load_scenarios(scenario_id: Optional[str] = None) -> list[dict]:
    """Загружает сценарии из папки evaluation/scenarios/."""
    EVAL_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = []

    for path in sorted(EVAL_SCENARIOS_DIR.glob("*.json")):
        with open(path) as f:
            scenario = json.load(f)
        if scenario_id and scenario.get("id") != scenario_id:
            continue
        scenarios.append(scenario)

    if not scenarios:
        logger.warning("Сценарии не найдены в %s", EVAL_SCENARIOS_DIR)
    return scenarios


# ── Запуск одного сценария ────────────────────────────────────────────────────

def run_scenario(scenario: dict, dry_run: bool = False, react: bool = False) -> ScenarioResult:
    """
    Запускает агента на одном сценарии и возвращает результат.

    dry_run=True: пропускает реальный деплой в AWS (для быстрого тестирования).
    """
    sid   = scenario["id"]
    sinp  = scenario["input"]
    logger.info("▶ [%s] %s", sid, sinp[:60])

    t_start = time.perf_counter()

    try:
        if dry_run:
            result = _run_dry(scenario)
        elif react:
            result = _run_react(sinp)
        else:
            result = _run_real(sinp)

        latency = time.perf_counter() - t_start

        # Проверяем acceptance criteria
        acceptance = scenario.get("acceptance", {})
        success = _check_acceptance(result, acceptance)

        # Определяем был ли security violation на первой попытке
        had_sec_violation = _check_security_violation(result)

        status = result.get("status", "unknown")
        sr = ScenarioResult(
            scenario_id=sid,
            input=sinp,
            success=success,
            status=status,
            generate_attempts=result.get("generate_attempts", 1),
            total_tokens=result.get("total_tokens", 0),
            total_cost_usd=result.get("total_cost_usd", 0.0),
            latency_sec=round(latency, 1),
            had_security_violation=had_sec_violation,
            validation_errors=result.get("validation_errors", []),
            error_message=result.get("error_message"),
            stack_name=result.get("plan_dict", {}).get("stack_name") if result.get("plan_dict") else None,
            resources_created=result.get("resources_created", []),
        )

        status_icon = "✓" if success else "✗"
        logger.info(
            "%s [%s] | tokens: %d | cost: $%.4f | %.1fs | attempts: %d",
            status_icon, sid,
            sr.total_tokens, sr.total_cost_usd,
            sr.latency_sec, sr.generate_attempts,
        )
        return sr

    except Exception as e:
        latency = time.perf_counter() - t_start
        logger.error("✗ [%s] исключение: %s", sid, e)
        return ScenarioResult(
            scenario_id=sid,
            input=sinp,
            success=False,
            status="error",
            generate_attempts=0,
            total_tokens=0,
            total_cost_usd=0.0,
            latency_sec=round(latency, 1),
            had_security_violation=False,
            error_message=str(e),
        )


def _run_real(user_input: str) -> dict:
    """Запускает реальный агент через LangGraph граф."""
    from orchestration.langgraph_graph import run_agent
    return run_agent(user_input)


def _run_react(user_input: str) -> dict:
    """Запускает агента через ReAct loop."""
    from orchestration.react_loop import run_react
    result = run_react(user_input)
    return {
        "status":             result.status,
        "plan_dict":          result.plan_dict,
        "workspace_path":     result.workspace_path,
        "validation_passed":  result.validation_passed,
        "validation_errors":  result.validation_errors,
        "generate_attempts":  result.generate_attempts,
        "total_tokens":       result.total_tokens,
        "total_cost_usd":     result.total_cost_usd,
        "error_message":      result.error_message,
        "clarification":      result.clarification,
    }


def _run_dry(scenario: dict) -> dict:
    """
    Dry-run: запускает только parse + generate, без deploy в AWS.
    Используется для быстрого тестирования без реального деплоя.
    """
    from agent.intent_parser import parse_intent, ClarificationNeeded
    from agent.iac_generator import generate_terraform
    from agent.validator import validate_terraform
    import uuid

    user_input = scenario["input"]
    request_id = str(uuid.uuid4())[:8]

    try:
        plan, parse_calls = parse_intent(user_input)
    except ClarificationNeeded as e:
        return {
            "status": "needs_clarification",
            "clarification": e.message,
            "total_tokens": sum(c.total_tokens for c in e.llm_calls),
            "total_cost_usd": sum(c.cost_usd for c in e.llm_calls),
            "generate_attempts": 0,
        }

    workspace, gen_calls = generate_terraform(plan, request_id)
    validation = validate_terraform(workspace)

    all_calls = parse_calls + gen_calls
    return {
        "status":             "done" if validation.passed else "failed",
        "plan_dict":          plan.model_dump(),
        "workspace_path":     str(workspace),
        "validation_passed":  validation.passed,
        "validation_errors":  validation.all_errors,
        "generate_attempts":  1,
        "total_tokens":       sum(c.total_tokens for c in all_calls),
        "total_cost_usd":     sum(c.cost_usd for c in all_calls),
        "resources_created":  [],
    }


def _check_acceptance(result: dict, acceptance: dict) -> bool:
    """Проверяет acceptance criteria сценария."""
    status = result.get("status", "")

    # Если сценарий проверяет что агент должен попросить уточнения
    if acceptance.get("should_ask_clarification"):
        return status == "needs_clarification"

    # Обычный сценарий — должен завершиться успешно
    if status not in ("done", "running"):
        return False

    # Проверяем минимальное количество ресурсов
    plan = result.get("plan_dict", {})
    resources = plan.get("resources", [])
    min_res = acceptance.get("min_resources", 1)
    if len(resources) < min_res:
        return False

    # Проверяем обязательные типы ресурсов
    required_types = acceptance.get("required_types", [])
    actual_types = {r.get("type") for r in resources}
    for req in required_types:
        if req not in actual_types:
            return False

    return True


def _check_security_violation(result: dict) -> bool:
    """Определяет был ли security violation на первой попытке."""
    errors = result.get("validation_errors", [])
    checkov_errors = [e for e in errors if e.startswith("CKV")]
    return len(checkov_errors) > 0


# ── Сохранение результатов ────────────────────────────────────────────────────

def save_results(metrics: BenchmarkMetrics, run_name: str) -> Path:
    """Сохраняет результаты в evaluation/results/{run_name}.json."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_RESULTS_DIR / f"{run_name}.json"

    output = {
        "run_name": run_name,
        "metrics":  metrics.to_dict(),
        "scenarios": [
            {
                "id":                r.scenario_id,
                "input":             r.input,
                "success":           r.success,
                "status":            r.status,
                "generate_attempts": r.generate_attempts,
                "tokens":            r.total_tokens,
                "cost_usd":          r.total_cost_usd,
                "latency_sec":       r.latency_sec,
                "had_sec_violation": r.had_security_violation,
                "error":             r.error_message,
            }
            for r in metrics.results
        ]
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Результаты сохранены: %s", path)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_benchmark(
    scenario_id: Optional[str] = None,
    dry_run: bool = False,
    react: bool = False,
    run_name: str = "run",
) -> BenchmarkMetrics:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scenarios = load_scenarios(scenario_id)
    if not scenarios:
        logger.error("Нет сценариев для запуска")
        return compute_metrics([])

    mode = "dry-run" if dry_run else ("ReAct" if react else "LangGraph")
    logger.info("Запуск бенчмарка | %d сценариев | режим: %s", len(scenarios), mode)
    logger.info("")

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario, dry_run=dry_run, react=react)
        results.append(result)

    metrics = compute_metrics(results)
    logger.info("")
    logger.info(metrics.report())
    save_results(metrics, run_name)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark runner для AWS агента")
    parser.add_argument("--scenario", help="ID конкретного сценария")
    parser.add_argument("--dry-run", action="store_true",
                        help="Без деплоя в AWS (только parse + generate)")
    parser.add_argument("--react", action="store_true",
                        help="Использовать ReAct вместо LangGraph")
    parser.add_argument("--run-name", default="run_langgraph",
                        help="Имя для файла результатов")
    args = parser.parse_args()

    run_benchmark(
        scenario_id=args.scenario,
        dry_run=args.dry_run,
        react=args.react,
        run_name=args.run_name,
    )
