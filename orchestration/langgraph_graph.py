"""
orchestration/langgraph_graph.py

Главный LangGraph граф — соединяет все стадии агента.

Текущий граф (стадии 1-2):
  lifecycle_check → parse → generate_validate → policy_gate → [END]

Стадии 3-4 будут добавлены в Неделях 5-6:
  ... → deploy → reconcile → lifecycle_save → [END]

Как читать этот файл:
  1. AgentState — словарь который путешествует через все узлы
  2. Узловые функции (node_*) — меняют State и возвращают его
  3. Условные рёбра (route_*) — смотрят на State и говорят куда идти
  4. build_graph() — собирает всё вместе
"""

import uuid
import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from agent.intent_parser import parse_intent, ClarificationNeeded, ParseError
from agent.generate_validate_loop import generate_and_validate, GenerationError
from agent.deployer import deploy, DeployResult
from agent.reconciler import reconcile, ReconcileResult
from agent.lifecycle_manager import check_existing_stack, save_stack
from security import policy_gate
from config import MAX_VALIDATE_RETRIES

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATE — словарь который путешествует через все узлы
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Состояние агента. Каждый узел читает нужные поля и дописывает свои.
    Никакой узел не удаляет чужие поля — только добавляет или обновляет.
    """
    # ── Вход ──────────────────────────────────────────────────────────────────
    user_input:      str              # исходный запрос пользователя
    request_id:      str              # уникальный ID сессии
    stack_context:   Optional[str]   # JSON существующего стэка (если изменение)

    # ── После parse ───────────────────────────────────────────────────────────
    plan_dict:       Optional[dict]   # Plan.model_dump() — сериализованный план
    parse_attempts:  int              # сколько раз вызывали LLM при парсинге

    # ── После generate_validate ───────────────────────────────────────────────
    workspace_path:       Optional[str]    # путь к папке с .tf файлами
    validation_passed:    bool            # прошёл ли tflint + checkov
    validation_errors:    list[str]       # список ошибок
    generate_attempts:    int             # сколько попыток generate→validate

    # ── После policy_gate ─────────────────────────────────────────────────────
    gate_approved:    Optional[bool]   # одобрил ли шлагбаум деплой
    gate_risk:        Optional[str]    # "LOW" или "HIGH"
    gate_reason:      Optional[str]    # почему одобрено/отклонено

    # ── Метрики (накапливаются) ────────────────────────────────────────────────
    total_tokens:    int
    total_cost_usd:  float
    total_latency:   float

    # ── После deploy ──────────────────────────────────────────────────────────
    deploy_result:     Optional[dict]   # terraform show -json после apply
    resources_created: list[str]        # адреса созданных ресурсов
    deployed_at:       Optional[str]    # ISO timestamp деплоя

    # ── После reconcile ───────────────────────────────────────────────────────
    is_reconciled:    bool             # совпадает ли реальное состояние с планом
    drift_detected:   bool             # был ли обнаружен дрифт
    drift_fields:     list[str]        # что именно дрифтовало

    # ── Управление потоком ────────────────────────────────────────────────────
    status:          str              # running / done / failed / needs_clarification
    error_message:   Optional[str]   # текст ошибки если status=failed
    clarification:   Optional[str]   # вопрос LLM если status=needs_clarification


# ══════════════════════════════════════════════════════════════════════════════
# 2. НАЧАЛЬНОЕ СОСТОЯНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def initial_state(user_input: str, stack_context: Optional[str] = None) -> AgentState:
    """Создаёт начальное состояние для нового запроса."""
    return AgentState(
        user_input=user_input,
        request_id=str(uuid.uuid4())[:8],
        stack_context=stack_context,
        plan_dict=None,
        parse_attempts=0,
        workspace_path=None,
        validation_passed=False,
        validation_errors=[],
        generate_attempts=0,
        gate_approved=None,
        gate_risk=None,
        gate_reason=None,
        total_tokens=0,
        total_cost_usd=0.0,
        total_latency=0.0,
        deploy_result=None,
        resources_created=[],
        deployed_at=None,
        is_reconciled=False,
        drift_detected=False,
        drift_fields=[],
        status="running",
        error_message=None,
        clarification=None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. УЗЛЫ ГРАФА
# ══════════════════════════════════════════════════════════════════════════════

def node_lifecycle_check(state: AgentState) -> AgentState:
    """
    Узел 1: проверяем существующий стэк в registry.
    Если нашли — добавляем контекст в State для parse.
    """
    logger.info(
        "[graph] ── lifecycle_check | request_id: %s", state["request_id"]
    )

    context = check_existing_stack(state["user_input"])

    if context:
        logger.info("[graph] существующий стэк найден — добавлен контекст")
        return {**state, "stack_context": context}

    logger.info("[graph] новый стэк")
    return state


def node_parse(state: AgentState) -> AgentState:
    """
    Узел 2: текст → JSON-план через LLM.
    При успехе записывает plan_dict в State.
    При ошибке — записывает status=failed или needs_clarification.
    """
    logger.info("[graph] ── parse | запрос: '%s'", state["user_input"][:60])

    try:
        from schemas.plan_schema import Plan

        plan, llm_calls = parse_intent(
            user_input=state["user_input"],
            stack_context=state["stack_context"],
        )

        tokens  = sum(c.total_tokens    for c in llm_calls)
        cost    = sum(c.cost_usd        for c in llm_calls)
        latency = sum(c.latency_seconds for c in llm_calls)

        logger.info(
            "[graph] parse ✓ | стэк: %s | ресурсов: %d | токены: %d",
            plan.stack_name, len(plan.resources), tokens,
        )

        return {
            **state,
            "plan_dict":     plan.model_dump(),
            "parse_attempts": len(llm_calls),
            "total_tokens":   state["total_tokens"]   + tokens,
            "total_cost_usd": state["total_cost_usd"] + cost,
            "total_latency":  state["total_latency"]  + latency,
            "status": "running",
        }

    except ClarificationNeeded as e:
        logger.info("[graph] parse → needs_clarification: %s", e.message[:80])
        return {
            **state,
            "status":        "needs_clarification",
            "clarification": e.message,
        }

    except ParseError as e:
        logger.error("[graph] parse → failed: %s", str(e)[:120])
        return {
            **state,
            "status":        "failed",
            "error_message": f"Не удалось разобрать запрос: {e}",
        }


def node_generate_validate(state: AgentState) -> AgentState:
    """
    Узел 3: план → .tf файлы → tflint + checkov → retry если нужно.
    """
    logger.info("[graph] ── generate_validate | стэк: %s", state["plan_dict"]["stack_name"])

    try:
        from schemas.plan_schema import Plan
        from pathlib import Path

        plan = Plan(**state["plan_dict"])
        result = generate_and_validate(plan, request_id=state["request_id"])

        logger.info(
            "[graph] generate_validate ✓ | попыток: %d | токены: %d",
            result.attempts, result.total_tokens,
        )

        return {
            **state,
            "workspace_path":    str(result.workspace),
            "validation_passed": result.validation.passed,
            "validation_errors": result.validation.all_errors,
            "generate_attempts": result.attempts,
            "total_tokens":      state["total_tokens"]   + result.total_tokens,
            "total_cost_usd":    state["total_cost_usd"] + result.total_cost_usd,
            "status": "running",
        }

    except GenerationError as e:
        logger.error("[graph] generate_validate → failed: %s", str(e)[:120])
        return {
            **state,
            "status":        "failed",
            "error_message": str(e),
        }


def node_policy_gate(state: AgentState) -> AgentState:
    """
    Узел 4: шлагбаум перед деплоем.
    Проверяет риск операций. HIGH риск → interactive=True.
    """
    logger.info("[graph] ── policy_gate")

    from pathlib import Path
    gate_result = policy_gate.check(
        workspace=Path(state["workspace_path"]),
        validation_passed=state["validation_passed"],
        interactive=True,
    )

    logger.info(
        "[graph] policy_gate: approved=%s | risk=%s | %s",
        gate_result.approved, gate_result.risk_level.value, gate_result.reason,
    )

    new_status = "running" if gate_result.approved else "failed"
    return {
        **state,
        "gate_approved": gate_result.approved,
        "gate_risk":     gate_result.risk_level.value,
        "gate_reason":   gate_result.reason,
        "status":        new_status,
        "error_message": None if gate_result.approved
                         else f"Policy Gate: {gate_result.reason}",
    }


def node_lifecycle_save(state: AgentState) -> AgentState:
    """
    Узел 7: сохраняем стэк в registry после успешного деплоя и reconcile.
    """
    logger.info("[graph] ── lifecycle_save")

    plan_dict = state["plan_dict"]
    save_stack(
        request_id=state["request_id"],
        stack_name=plan_dict["stack_name"],
        intent=state["user_input"],
        plan_dict=plan_dict,
        workspace=state.get("workspace_path"),
        tf_state=state.get("deploy_result"),
    )

    logger.info(
        "[graph] ✓ стэк '%s' сохранён в registry",
        plan_dict["stack_name"],
    )

    return {**state, "status": "done"}
    """
    Узел 4: шлагбаум перед деплоем.
    Проверяет риск операций. HIGH риск → interactive=True (спрашивает юзера).
    """
    logger.info("[graph] ── policy_gate")

    from pathlib import Path

    gate_result = policy_gate.check(
        workspace=Path(state["workspace_path"]),
        validation_passed=state["validation_passed"],
        interactive=True,   # спрашиваем пользователя при HIGH риске
    )

    logger.info(
        "[graph] policy_gate: approved=%s | risk=%s | %s",
        gate_result.approved, gate_result.risk_level.value, gate_result.reason,
    )

    new_status = "running" if gate_result.approved else "failed"

    return {
        **state,
        "gate_approved": gate_result.approved,
        "gate_risk":     gate_result.risk_level.value,
        "gate_reason":   gate_result.reason,
        "status":        new_status,
        "error_message": None if gate_result.approved
                         else f"Policy Gate: {gate_result.reason}",
    }


def node_deploy(state: AgentState) -> AgentState:
    """
    Узел 5: terraform apply → реальные ресурсы в AWS.
    """
    from pathlib import Path
    workspace = Path(state["workspace_path"])
    plan_dict = state["plan_dict"]

    logger.info(
        "[graph] ── deploy | стэк: %s | workspace: .../%s",
        plan_dict["stack_name"], workspace.name,
    )

    result = deploy(
        workspace=workspace,
        stack_name=plan_dict["stack_name"],
        request_id=state["request_id"],
    )

    if not result.success:
        logger.error("[graph] deploy ✗ | ошибка: %s", result.error)
        return {
            **state,
            "status":        "failed",
            "error_message": f"terraform apply ошибка: {result.error}",
        }

    logger.info(
        "[graph] deploy ✓ | ресурсов создано: %d | время: %.1fс",
        len(result.resources_created), result.duration_sec,
    )

    return {
        **state,
        "deploy_result":     result.actual_state,
        "resources_created": result.resources_created,
        "deployed_at":       result.applied_at,
        "status":            "running",
    }


def node_reconcile(state: AgentState) -> AgentState:
    """
    Узел 6: сравниваем план с реальным состоянием AWS.
    Если есть дрифт — LLM генерирует патч.
    """
    from pathlib import Path
    from schemas.plan_schema import Plan
    from agent.iac_generator import read_workspace_code

    logger.info("[graph] ── reconcile")

    plan      = Plan(**state["plan_dict"])
    workspace = Path(state["workspace_path"])
    actual    = state.get("deploy_result") or {}

    try:
        original_code = read_workspace_code(workspace)
    except FileNotFoundError:
        original_code = ""

    result = reconcile(
        plan=plan,
        actual_state=actual,
        workspace=workspace,
        original_tf_code=original_code,
    )

    logger.info(
        "[graph] reconcile: is_reconciled=%s | дрифт=%s | попыток=%d",
        result.is_reconciled, result.drift_report.has_drift, result.attempts,
    )

    tokens  = sum(c.total_tokens    for c in result.llm_calls)
    cost    = sum(c.cost_usd        for c in result.llm_calls)
    latency = sum(c.latency_seconds for c in result.llm_calls)

    return {
        **state,
        "is_reconciled":  result.is_reconciled,
        "drift_detected": result.drift_report.has_drift,
        "drift_fields":   result.drift_report.drifted_fields,
        "total_tokens":   state["total_tokens"]   + tokens,
        "total_cost_usd": state["total_cost_usd"] + cost,
        "total_latency":  state["total_latency"]  + latency,
        "status":         "running" if result.is_reconciled else "failed",
        "error_message":  result.error if not result.is_reconciled else None,
    }
    """
    Узел 5: terraform apply → реальные ресурсы в AWS.
    """
    from pathlib import Path
    workspace = Path(state["workspace_path"])
    plan_dict = state["plan_dict"]

    logger.info(
        "[graph] ── deploy | стэк: %s | workspace: .../%s",
        plan_dict["stack_name"], workspace.name,
    )

    result = deploy(
        workspace=workspace,
        stack_name=plan_dict["stack_name"],
        request_id=state["request_id"],
    )

    if not result.success:
        logger.error("[graph] deploy ✗ | ошибка: %s", result.error)
        return {
            **state,
            "status":        "failed",
            "error_message": f"terraform apply ошибка: {result.error}",
        }

    logger.info(
        "[graph] deploy ✓ | ресурсов создано: %d | время: %.1fс",
        len(result.resources_created), result.duration_sec,
    )

    return {
        **state,
        "deploy_result":     result.actual_state,
        "resources_created": result.resources_created,
        "deployed_at":       result.applied_at,
        "status":            "running",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. УСЛОВНЫЕ РЁБРА — куда идти после каждого узла
# ══════════════════════════════════════════════════════════════════════════════

def route_after_parse(state: AgentState) -> str:
    """После parse: успех → generate_validate, иначе → END."""
    if state["status"] == "running":
        return "generate_validate"
    return END   # failed или needs_clarification


def route_after_generate_validate(state: AgentState) -> str:
    """После generate_validate: успех → policy_gate, ошибка → END."""
    if state["status"] == "running":
        return "policy_gate"
    return END


def route_after_policy_gate(state: AgentState) -> str:
    """После policy_gate: одобрено → deploy, отклонено → END."""
    if state["gate_approved"]:
        return "deploy"
    return END


# ══════════════════════════════════════════════════════════════════════════════
# 5. СБОРКА ГРАФА
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Собирает и компилирует LangGraph граф.
    Возвращает скомпилированный граф готовый к вызову .invoke().
    """
    graph = StateGraph(AgentState)

    # ── Регистрируем узлы ──────────────────────────────────────────────────
    graph.add_node("lifecycle_check",    node_lifecycle_check)
    graph.add_node("parse",              node_parse)
    graph.add_node("generate_validate",  node_generate_validate)
    graph.add_node("policy_gate",        node_policy_gate)
    graph.add_node("deploy",             node_deploy)
    graph.add_node("reconcile",          node_reconcile)
    graph.add_node("lifecycle_save",     node_lifecycle_save)

    # ── Точка входа ───────────────────────────────────────────────────────
    graph.set_entry_point("lifecycle_check")

    # ── Рёбра ─────────────────────────────────────────────────────────────
    # lifecycle_check всегда идёт в parse
    graph.add_edge("lifecycle_check", "parse")

    # После parse — условный выбор
    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "generate_validate": "generate_validate",
            END: END,
        }
    )

    # После generate_validate — условный выбор
    graph.add_conditional_edges(
        "generate_validate",
        route_after_generate_validate,
        {
            "policy_gate": "policy_gate",
            END: END,
        }
    )

    # После policy_gate — условный выбор
    graph.add_conditional_edges(
        "policy_gate",
        route_after_policy_gate,
        {
            "deploy": "deploy",
            END: END,
        }
    )

    # deploy → reconcile (если успешно) или END (если ошибка)
    graph.add_conditional_edges(
        "deploy",
        lambda s: "reconcile" if s["status"] == "running" else END,
        {"reconcile": "reconcile", END: END}
    )

    # reconcile → lifecycle_save (если ок) или END (если ошибка)
    graph.add_conditional_edges(
        "reconcile",
        lambda s: "lifecycle_save" if s["status"] == "running" else END,
        {"lifecycle_save": "lifecycle_save", END: END}
    )

    # lifecycle_save → END
    graph.add_edge("lifecycle_save", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 6. УДОБНАЯ ФУНКЦИЯ ДЛЯ ВЫЗОВА
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(
    user_input: str,
    stack_context: Optional[str] = None,
) -> AgentState:
    """
    Запускает агента с текстовым запросом.
    Возвращает финальный State.

    Пример:
        result = run_agent("Create an S3 bucket for user photos")
        if result["status"] == "done":
            print(result["workspace_path"])
    """
    compiled = build_graph()
    state    = initial_state(user_input, stack_context)
    result   = compiled.invoke(state)
    return result
