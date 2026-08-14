"""
orchestration/react_loop.py

ReAct (Reasoning + Acting) — альтернативный orchestration для сравнения с LangGraph.

Разница с LangGraph:
  LangGraph: маршрут задан явно в коде (граф с рёбрами).
             После каждого узла функция-условие решает куда идти.
             Предсказуемо, легко аудировать, но менее гибко.

  ReAct:     LLM сама решает что делать следующим.
             Цикл Thought → Action → Observation.
             Гибче, но менее предсказуемо.

Для диплома:
  Прогоняем те же 20 сценариев через ReAct и сравниваем с LangGraph
  через McNemar's test. Гипотеза: LangGraph покажет более стабильный TCR
  потому что маршрут детерминирован, а ReAct может "заблудиться".

Реализация ReAct здесь упрощённая:
  Вместо полного tool-calling цикла — LLM решает только на этапе parse
  нужно ли генерировать план или спросить уточнения.
  Generate и validate — те же модули что в LangGraph.
  Разница: нет явного графа, решения принимаются через дополнительный
  LLM вызов на каждом шаге.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from agent.intent_parser import parse_intent, ClarificationNeeded, ParseError
from agent.generate_validate_loop import generate_and_validate, GenerationError
from agent.llm_client import call_llm, load_prompt, LLMResponse
from schemas.plan_schema import Plan

logger = logging.getLogger(__name__)

MAX_REACT_STEPS = 10   # защита от бесконечного цикла


# ── Структура результата ──────────────────────────────────────────────────────

@dataclass
class ReActResult:
    """Результат ReAct прогона — совместим с AgentState для метрик."""
    status:            str            # done / failed / needs_clarification
    plan_dict:         Optional[dict] = None
    workspace_path:    Optional[str]  = None
    validation_passed: bool = False
    validation_errors: list[str] = field(default_factory=list)
    generate_attempts: int = 0
    total_tokens:      int = 0
    total_cost_usd:    float = 0.0
    total_latency:     float = 0.0
    error_message:     Optional[str] = None
    clarification:     Optional[str] = None
    steps_taken:       int = 0
    llm_calls:         list[LLMResponse] = field(default_factory=list)


# ── ReAct PROMPT ──────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an AWS infrastructure agent using the ReAct (Reasoning + Acting) framework.

At each step you must output EXACTLY this format:
Thought: <your reasoning about what to do next>
Action: <one of: PARSE | GENERATE | VALIDATE | DONE | CLARIFY>
Action Input: <input for the action, or empty if not needed>

Available actions:
- PARSE: Convert the user request to a JSON infrastructure plan
- GENERATE: Generate Terraform code from the plan
- VALIDATE: Check the generated code with tflint and checkov
- DONE: The task is complete
- CLARIFY: Ask the user for more information (use when request is unclear)

Rules:
- Always start with PARSE
- After PARSE succeeds, use GENERATE
- After GENERATE, use VALIDATE
- After VALIDATE succeeds, use DONE
- If VALIDATE fails, use GENERATE again (with corrections)
- Use CLARIFY only if the request is too vague to create any plan
- Maximum {max_steps} steps allowed

Current state:
{state_summary}

User request: {user_request}
"""


# ── Главная функция ───────────────────────────────────────────────────────────

def run_react(
    user_input: str,
    request_id: Optional[str] = None,
) -> ReActResult:
    """
    Запускает ReAct цикл для обработки запроса пользователя.

    В отличие от LangGraph, здесь LLM принимает решения о следующем шаге
    через дополнительный вызов с REACT_SYSTEM_PROMPT.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    logger.info("[react] ── старт | request_id: %s", request_id)
    logger.info("[react] запрос: '%s'", user_input[:60])

    all_llm_calls: list[LLMResponse] = []
    t_start = time.perf_counter()

    # Состояние которое передаётся в каждый шаг
    state = {
        "user_input":    user_input,
        "request_id":    request_id,
        "plan":          None,
        "workspace":     None,
        "validation":    None,
        "last_error":    None,
        "step":          0,
    }

    plan_obj: Optional[Plan] = None
    workspace_path: Optional[str] = None

    for step in range(1, MAX_REACT_STEPS + 1):
        state["step"] = step
        logger.info("[react] шаг %d/%d", step, MAX_REACT_STEPS)

        # ── Решение LLM: что делать ────────────────────────────────────────
        action, action_input, react_call = _decide_next_action(state)
        all_llm_calls.append(react_call)
        logger.info("[react] Action: %s", action)

        # ── Выполняем действие ─────────────────────────────────────────────
        if action == "CLARIFY":
            return ReActResult(
                status="needs_clarification",
                clarification=action_input or "Пожалуйста уточни запрос.",
                steps_taken=step,
                total_tokens=sum(c.total_tokens for c in all_llm_calls),
                total_cost_usd=sum(c.cost_usd for c in all_llm_calls),
                total_latency=round(time.perf_counter() - t_start, 1),
                llm_calls=all_llm_calls,
            )

        elif action == "PARSE":
            try:
                plan_obj, parse_calls = parse_intent(user_input)
                all_llm_calls.extend(parse_calls)
                state["plan"] = plan_obj.model_dump()
                state["last_error"] = None
                logger.info("[react] PARSE ✓ | стэк: %s", plan_obj.stack_name)
            except ClarificationNeeded as e:
                all_llm_calls.extend(e.llm_calls)
                return ReActResult(
                    status="needs_clarification",
                    clarification=e.message,
                    steps_taken=step,
                    total_tokens=sum(c.total_tokens for c in all_llm_calls),
                    total_cost_usd=sum(c.cost_usd for c in all_llm_calls),
                    total_latency=round(time.perf_counter() - t_start, 1),
                    llm_calls=all_llm_calls,
                )
            except ParseError as e:
                return ReActResult(
                    status="failed",
                    error_message=str(e),
                    steps_taken=step,
                    total_tokens=sum(c.total_tokens for c in all_llm_calls),
                    total_cost_usd=sum(c.cost_usd for c in all_llm_calls),
                    total_latency=round(time.perf_counter() - t_start, 1),
                    llm_calls=all_llm_calls,
                )

        elif action == "GENERATE":
            if plan_obj is None:
                state["last_error"] = "Нет плана для генерации. Сначала выполни PARSE."
                continue
            try:
                gen_result = generate_and_validate(plan_obj, request_id)
                all_llm_calls.extend(gen_result.llm_calls)
                workspace_path = str(gen_result.workspace)
                state["workspace"] = workspace_path
                state["validation"] = {
                    "passed": gen_result.validation.passed,
                    "errors": gen_result.validation.all_errors,
                }
                state["last_error"] = None
                logger.info(
                    "[react] GENERATE ✓ | validation: %s",
                    "passed" if gen_result.validation.passed else "failed",
                )
            except GenerationError as e:
                state["last_error"] = str(e)
                return ReActResult(
                    status="failed",
                    error_message=str(e),
                    steps_taken=step,
                    total_tokens=sum(c.total_tokens for c in all_llm_calls),
                    total_cost_usd=sum(c.cost_usd for c in all_llm_calls),
                    total_latency=round(time.perf_counter() - t_start, 1),
                    llm_calls=all_llm_calls,
                )

        elif action == "DONE":
            val = state.get("validation", {})
            latency = round(time.perf_counter() - t_start, 1)
            logger.info("[react] DONE | шагов: %d | токены: %d",
                        step, sum(c.total_tokens for c in all_llm_calls))
            return ReActResult(
                status="done",
                plan_dict=state.get("plan"),
                workspace_path=workspace_path,
                validation_passed=val.get("passed", False),
                validation_errors=val.get("errors", []),
                generate_attempts=sum(
                    1 for c in all_llm_calls if c.call_type == "generate"
                ),
                steps_taken=step,
                total_tokens=sum(c.total_tokens for c in all_llm_calls),
                total_cost_usd=sum(c.cost_usd for c in all_llm_calls),
                total_latency=latency,
                llm_calls=all_llm_calls,
            )

        else:
            logger.warning("[react] неизвестный action: %s — пропускаем", action)

    # Исчерпали шаги
    return ReActResult(
        status="failed",
        error_message=f"Превышен лимит {MAX_REACT_STEPS} шагов ReAct",
        steps_taken=MAX_REACT_STEPS,
        total_tokens=sum(c.total_tokens for c in all_llm_calls),
        total_cost_usd=sum(c.cost_usd for c in all_llm_calls),
        total_latency=round(time.perf_counter() - t_start, 1),
        llm_calls=all_llm_calls,
    )


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _decide_next_action(state: dict) -> tuple[str, str, LLMResponse]:
    """
    Вызывает LLM для принятия решения о следующем действии.
    Возвращает (action, action_input, llm_response).
    """
    state_summary = _format_state(state)

    prompt = REACT_SYSTEM_PROMPT.format(
        max_steps=MAX_REACT_STEPS,
        state_summary=state_summary,
        user_request=state["user_input"],
    )

    response = call_llm(prompt, call_type="react_decide")

    action, action_input = _parse_react_response(response.text)
    return action, action_input, response


def _format_state(state: dict) -> str:
    """Форматирует текущее состояние для промпта."""
    lines = [f"Step: {state['step']}"]

    if state.get("plan"):
        plan = state["plan"]
        lines.append(f"Plan: {plan.get('stack_name')} ({len(plan.get('resources', []))} resources)")
    else:
        lines.append("Plan: not yet created")

    if state.get("workspace"):
        lines.append(f"Workspace: {state['workspace']}")

    if state.get("validation"):
        val = state["validation"]
        if val.get("passed"):
            lines.append("Validation: PASSED")
        else:
            errors = val.get("errors", [])
            lines.append(f"Validation: FAILED ({len(errors)} errors)")
            for e in errors[:3]:
                lines.append(f"  - {e}")

    if state.get("last_error"):
        lines.append(f"Last error: {state['last_error'][:100]}")

    return "\n".join(lines)


def _parse_react_response(text: str) -> tuple[str, str]:
    """
    Парсит ответ LLM в формате Thought/Action/Action Input.
    Возвращает (action, action_input).
    """
    text = text.strip()
    action = "PARSE"  # дефолт если не распарсили
    action_input = ""

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            action_part = line[len("Action:"):].strip().upper()
            # Берём первое слово (PARSE, GENERATE, etc.)
            action = action_part.split()[0] if action_part else "PARSE"
        elif line.startswith("Action Input:"):
            action_input = line[len("Action Input:"):].strip()

    # Нормализуем
    valid_actions = {"PARSE", "GENERATE", "VALIDATE", "DONE", "CLARIFY"}
    if action not in valid_actions:
        logger.warning("[react] неизвестный action '%s' → дефолт PARSE", action)
        action = "PARSE"

    return action, action_input
