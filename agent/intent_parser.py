"""
agent/intent_parser.py

Стадия 1: текст пользователя → валидный объект Plan.

Что происходит внутри:
  1. Загружаем parser_prompt.txt, подставляем запрос пользователя
  2. Вызываем LLM через llm_client.call_llm()
  3. Парсим JSON из ответа
  4. Валидируем через pydantic Plan(**data)
  5. Если что-то пошло не так — повторяем с описанием ошибки
  6. После MAX_PARSE_RETRIES попыток — бросаем исключение
"""

import json
import logging
from typing import Optional

from pydantic import ValidationError

from agent.llm_client import call_llm, load_prompt, LLMResponse
from schemas.plan_schema import Plan
from security.sanitizer import sanitize
from config import MAX_PARSE_RETRIES

logger = logging.getLogger(__name__)


# ── Исключения ────────────────────────────────────────────────────────────────

class ParseError(Exception):
    """LLM не смогла разобрать запрос после всех попыток."""
    pass

class ClarificationNeeded(Exception):
    """
    LLM вернула {"error": "..."} — запрос слишком расплывчатый.
    Несёт llm_calls для подсчёта токенов в метриках.
    """
    def __init__(self, message: str, llm_calls: list = None):
        self.message = message
        self.llm_calls = llm_calls or []
        super().__init__(message)


# ── Главная функция ───────────────────────────────────────────────────────────

def parse_intent(
    user_input: str,
    stack_context: Optional[str] = None,
    sanitizer_enabled: bool = True,
) -> tuple[Plan, list[LLMResponse]]:
    """
    Разбирает запрос пользователя и возвращает валидный Plan.

    Параметры:
        user_input         — текст от пользователя
        stack_context       — если стэк уже существует, передаём его JSON сюда
                              чтобы LLM знала контекст при изменении.
                              Это единственный источник данных, пришедших
                              не напрямую от пользователя в этой текущей
                              сессии (а из ранее сохранённого стэка), и
                              потому единственная точка, где untrusted
                              data (например, вредоносный intent или тег
                              из прошлого запроса) может попасть в промпт.
        sanitizer_enabled   — включить/выключить обёртку stack_context в
                              <untrusted_data> теги. Используется для
                              security ablation study (RQ3); в normal
                              production-режиме всегда True.

    Возвращает:
        (plan, llm_calls) — объект Plan и список всех вызовов LLM
                            (нужен для метрик evaluation)

    Исключения:
        ClarificationNeeded — запрос неоднозначен, нужно уточнение
        ParseError          — не удалось получить валидный план
    """
    llm_calls: list[LLMResponse] = []
    last_error: str = ""

    # Если есть контекст существующего стэка — добавляем в запрос.
    # stack_context пришёл из ранее сохранённого стэка (agent/lifecycle_manager.py),
    # то есть это untrusted data относительно текущей сессии — оборачиваем
    # его в security/sanitizer.py перед тем, как подставить в промпт.
    full_request = user_input
    if stack_context:
        safe_context = (
            sanitize(stack_context, source="existing_stack_context")
            if sanitizer_enabled
            else stack_context
        )
        full_request = (
            f"{user_input}\n\n"
            f"[EXISTING STACK CONTEXT]\n{safe_context}"
        )

    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        logger.info("[parse] попытка %d/%d", attempt, MAX_PARSE_RETRIES)

        # На первой попытке — чистый промпт
        # На следующих — добавляем описание предыдущей ошибки
        if attempt == 1:
            prompt = load_prompt(
                "parser_prompt.txt",
                user_request=full_request,
            )
        else:
            # Добавляем ошибку предыдущей попытки в конец промпта
            prompt = load_prompt(
                "parser_prompt.txt",
                user_request=full_request,
            )
            prompt += (
                f"\n\n## PREVIOUS ATTEMPT FAILED\n\n"
                f"Your previous response caused this error:\n{last_error}\n\n"
                f"Fix the error and return valid JSON."
            )

        # Вызов LLM
        response = call_llm(prompt, call_type="parse")
        llm_calls.append(response)

        # Парсим и валидируем ответ
        result, error = _parse_response(response.text)

        if isinstance(result, Plan):
            logger.info("[parse] ✓ успех на попытке %d", attempt)
            return result, llm_calls

        if isinstance(result, str):
            # LLM вернула {"error": "..."} — нужно уточнение
            raise ClarificationNeeded(result, llm_calls=llm_calls)

        # Ошибка — запоминаем для следующей попытки
        last_error = error
        logger.warning("[parse] ✗ попытка %d: %s", attempt, error[:120])

    raise ParseError(
        f"Не удалось получить валидный план после {MAX_PARSE_RETRIES} попыток.\n"
        f"Последняя ошибка: {last_error}"
    )


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[Plan | str | None, str]:
    """
    Пытается извлечь Plan или сообщение об уточнении из ответа LLM.

    Возвращает одно из:
        (Plan,  "")      — успех
        (str,   "")      — LLM просит уточнения (вернула {"error": "..."})
        (None,  "текст ошибки") — что-то пошло не так, нужен retry
    """
    text = text.strip()

    # Убираем markdown-обёртки если LLM их добавила несмотря на инструкции
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])          # убираем первую строку (```json)
        text = text.rsplit("```", 1)[0]      # убираем закрывающие ```
        text = text.strip()

    # Парсим JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Невалидный JSON: {e}. Ответ LLM: {text[:200]}"

    # LLM вернула {"error": "..."} — нужно уточнение от пользователя
    if "error" in data and len(data) == 1:
        return data["error"], ""

    # Валидируем через pydantic
    try:
        plan = Plan(**data)
        return plan, ""
    except ValidationError as e:
        # Форматируем ошибки pydantic в читаемый текст для следующего промпта
        errors = []
        for err in e.errors():
            field = " → ".join(str(x) for x in err["loc"])
            errors.append(f"Field '{field}': {err['msg']}")
        return None, "Validation errors:\n" + "\n".join(errors)
