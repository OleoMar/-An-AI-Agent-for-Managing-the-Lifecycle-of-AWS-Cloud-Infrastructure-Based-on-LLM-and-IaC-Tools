"""
agent/llm_client.py

Единственное место в проекте откуда идут запросы к LLM API.
Все остальные модули (intent_parser, iac_generator, reconciler...)
используют только эту функцию — не вызывают API напрямую.

Зачем так:
  Если завтра сменить модель или провайдера — меняем один файл,
  а не шесть. Плюс логирование токенов и времени происходит
  автоматически для всех вызовов.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

import anthropic

from config import (
    LLM_MODEL,
    LLM_API_KEY,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

# ── Логгер ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Структура ответа ──────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Всё что возвращает вызов LLM.
    Используем dataclass вместо просто строки чтобы всегда
    иметь рядом метрики — для evaluation и логирования.
    """
    text: str                    # сам ответ от LLM
    input_tokens: int            # токенов в промпте
    output_tokens: int           # токенов в ответе
    latency_seconds: float       # сколько секунд ждали ответа
    model: str                   # какую модель использовали
    call_type: str               # "parse" / "generate" / "correct" / "reconcile" / "lifecycle"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """
        Примерная стоимость вызова для claude-sonnet-4-6.
        Обновить цены если поменяется тарифный план.
        Input:  $3.00 per 1M tokens
        Output: $15.00 per 1M tokens
        """
        input_cost  = self.input_tokens  * 3.00  / 1_000_000
        output_cost = self.output_tokens * 15.00 / 1_000_000
        return round(input_cost + output_cost, 6)


# ── Основная функция ──────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    call_type: str,
    system: Optional[str] = None,
) -> LLMResponse:
    """
    Отправляет prompt в LLM и возвращает LLMResponse.

    Параметры:
        prompt     — уже заполненный промпт (с подставленными значениями)
        call_type  — метка для логов: "parse", "generate", "correct",
                     "reconcile", "lifecycle"
        system     — системное сообщение (опционально, редко нужно)

    Исключения:
        anthropic.APIConnectionError  — нет сети
        anthropic.AuthenticationError — неверный API ключ
        anthropic.RateLimitError      — превышен лимит запросов
        Все пробрасываются наверх — intent_parser решает что делать.
    """
    if not LLM_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY не задан. "
            "Проверь файл .env — там должна быть строка ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=LLM_API_KEY)

    messages = [{"role": "user", "content": prompt}]

    # Фиксируем время ДО запроса
    t_start = time.perf_counter()

    logger.info(
        "[LLM] → %s | модель: %s | ~%d символов промпта",
        call_type, LLM_MODEL, len(prompt)
    )

    # Вызов API
    kwargs = dict(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    # Фиксируем время ПОСЛЕ ответа
    latency = time.perf_counter() - t_start

    # Извлекаем текст (первый content block типа text)
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
            break

    result = LLMResponse(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_seconds=round(latency, 2),
        model=response.model,
        call_type=call_type,
    )

    logger.info(
        "[LLM] ← %s | токены: %d in + %d out = %d total | "
        "время: %.1fс | стоимость: $%.4f",
        call_type,
        result.input_tokens,
        result.output_tokens,
        result.total_tokens,
        result.latency_seconds,
        result.cost_usd,
    )

    return result


# ── Вспомогательная функция для загрузки промптов ─────────────────────────────

def load_prompt(filename: str, **substitutions: str) -> str:
    """
    Читает файл из папки prompts/ и подставляет значения.

    Пример:
        prompt = load_prompt(
            "parser_prompt.txt",
            user_request="Deploy a Node.js API"
        )
    """
    from config import PROMPTS_DIR

    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Промпт не найден: {path}\n"
            f"Убедись что файл {filename} есть в папке prompts/"
        )

    text = path.read_text(encoding="utf-8")

    for key, value in substitutions.items():
        placeholder = "{" + key + "}"
        text = text.replace(placeholder, value)

    return text
