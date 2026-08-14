"""
agent/generate_validate_loop.py

Цикл стадии 2: generate → validate → retry (до MAX_VALIDATE_RETRIES раз).

Это не отдельный узел LangGraph — это вспомогательная функция,
которую будет вызывать узел generate в графе.

Логика:
  попытка 1: generate_terraform(plan)           → workspace
             validate_terraform(workspace)       → result
             если passed → возвращаем workspace
             если нет    → идём на попытку 2

  попытка 2: generate_terraform(plan,
               validation_errors=result.all_errors,
               previous_tf_code=текущий main.tf)  → workspace
             validate_terraform(workspace)          → result
             ...и так до MAX_VALIDATE_RETRIES

  если все попытки исчерпаны → GenerationError
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent.iac_generator import generate_terraform, read_workspace_code
from agent.validator import validate_terraform, ValidationResult
from agent.llm_client import LLMResponse
from schemas.plan_schema import Plan
from config import MAX_VALIDATE_RETRIES

logger = logging.getLogger(__name__)


# ── Исключение ────────────────────────────────────────────────────────────────

class GenerationError(Exception):
    """Не удалось сгенерировать валидный Terraform после всех попыток."""
    pass


# ── Результат цикла ───────────────────────────────────────────────────────────

@dataclass
class GenerateValidateResult:
    """Всё что вернул цикл generate→validate."""
    workspace: Path                              # папка с финальными .tf файлами
    validation: ValidationResult                 # последний результат валидации
    attempts: int                                # сколько попыток потребовалось
    llm_calls: list[LLMResponse] = field(default_factory=list)  # все вызовы LLM

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)


# ── Главная функция ───────────────────────────────────────────────────────────

def generate_and_validate(
    plan: Plan,
    request_id: str,
) -> GenerateValidateResult:
    """
    Запускает цикл generate → validate с автоматическим retry.

    Параметры:
        plan        — валидный Plan от intent_parser
        request_id  — уникальный ID запроса (для имени workspace папки)

    Возвращает:
        GenerateValidateResult с финальным workspace и метриками

    Исключения:
        GenerationError — если все попытки исчерпаны
    """
    all_llm_calls: list[LLMResponse] = []
    last_validation: ValidationResult | None = None
    workspace: Path | None = None

    for attempt in range(1, MAX_VALIDATE_RETRIES + 1):
        logger.info(
            "[gen+val] попытка %d/%d", attempt, MAX_VALIDATE_RETRIES
        )

        # ── Генерация ─────────────────────────────────────────────────────────
        if attempt == 1:
            # Первая попытка — чистая генерация
            workspace, llm_calls = generate_terraform(
                plan=plan,
                request_id=request_id,
            )
        else:
            # Retry — передаём ошибки и предыдущий код
            previous_code = read_workspace_code(workspace)
            workspace, llm_calls = generate_terraform(
                plan=plan,
                request_id=request_id,
                validation_errors=last_validation.all_errors,
                previous_tf_code=previous_code,
            )

        all_llm_calls.extend(llm_calls)

        # ── Валидация ─────────────────────────────────────────────────────────
        last_validation = validate_terraform(workspace)

        if last_validation.passed:
            logger.info(
                "[gen+val] ✓ валидация прошла на попытке %d "
                "| токенов: %d | стоимость: $%.4f",
                attempt,
                sum(c.total_tokens for c in all_llm_calls),
                sum(c.cost_usd for c in all_llm_calls),
            )
            return GenerateValidateResult(
                workspace=workspace,
                validation=last_validation,
                attempts=attempt,
                llm_calls=all_llm_calls,
            )

        # Валидация не прошла — логируем и идём на следующую попытку
        logger.warning(
            "[gen+val] ✗ попытка %d: %s",
            attempt,
            last_validation.summary(),
        )
        for err in last_validation.all_errors:
            logger.warning("[gen+val]   - %s", err)

    # Все попытки исчерпаны
    raise GenerationError(
        f"Не удалось сгенерировать валидный Terraform "
        f"после {MAX_VALIDATE_RETRIES} попыток.\n"
        f"Последние ошибки:\n" +
        "\n".join(f"  - {e}" for e in last_validation.all_errors)
    )
