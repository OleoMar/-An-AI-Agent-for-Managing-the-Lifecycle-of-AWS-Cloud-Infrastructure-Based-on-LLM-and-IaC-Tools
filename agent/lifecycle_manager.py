"""
agent/lifecycle_manager.py

Стадия 4: управление жизненным циклом стэков.

Две задачи:
  1. При входе (вход в граф): определить новый это стэк или изменение
     существующего. Если существующий — добавить его контекст в State.

  2. При выходе (после успешного деплоя): сохранить стэк в registry.

Логика определения "новый или существующий":
  - Ищем в registry стэк с похожим именем или intent
  - Если нашли → это изменение, передаём контекст в parse
  - Если не нашли → новый стэк, план строится с нуля
"""

import json
import logging
from typing import Optional

from storage import stack_registry
from schemas.plan_schema import Plan

logger = logging.getLogger(__name__)


# ── Вход в граф: определяем контекст ─────────────────────────────────────────

def check_existing_stack(user_input: str) -> Optional[str]:
    """
    Ищет существующий стэк в registry по запросу пользователя.

    Возвращает:
        JSON строку с контекстом стэка если нашли,
        None если это новый запрос.

    Контекст передаётся в parse_intent() как stack_context —
    LLM видит предыдущий план и знает что изменять.
    """
    stack_registry.init_db()
    existing = stack_registry.find_by_intent(user_input)

    if not existing:
        logger.info("[lifecycle] новый стэк — контекст не найден")
        return None

    logger.info(
        "[lifecycle] найден существующий стэк '%s' (id: %s)",
        existing["stack_name"], existing["id"],
    )

    # Формируем контекст для LLM
    context = {
        "existing_stack_id":   existing["id"],
        "existing_stack_name": existing["stack_name"],
        "original_intent":     existing["intent"],
        "current_plan":        existing["plan_json"],
        "status":              existing["status"],
        "workspace":           existing["workspace"],
    }

    return json.dumps(context, indent=2)


def get_stack_by_name(stack_name: str) -> Optional[dict]:
    """
    Возвращает стэк по точному имени.
    Используется когда пользователь называет стэк явно.
    """
    stack_registry.init_db()
    return stack_registry.get_by_name(stack_name)


# ── Выход из графа: сохраняем результат ──────────────────────────────────────

def save_stack(
    request_id: str,
    stack_name: str,
    intent: str,
    plan_dict: dict,
    workspace: Optional[str] = None,
    tf_state: Optional[dict] = None,
) -> None:
    """
    Сохраняет стэк в registry после успешного деплоя.
    Вызывается в узле lifecycle_save графа.
    """
    stack_registry.init_db()
    stack_registry.save(
        request_id=request_id,
        stack_name=stack_name,
        intent=intent,
        plan_dict=plan_dict,
        workspace=workspace,
        tf_state=tf_state,
        status="active",
    )
    logger.info(
        "[lifecycle] стэк '%s' сохранён (id: %s)",
        stack_name, request_id,
    )


def mark_destroyed(request_id: str) -> None:
    """Помечает стэк как уничтоженный после terraform destroy."""
    stack_registry.update_status(request_id, "destroyed")
    logger.info("[lifecycle] стэк %s помечен как destroyed", request_id)


def list_active_stacks() -> list[dict]:
    """Возвращает все активные стэки — для отображения в UI."""
    stack_registry.init_db()
    return stack_registry.list_all(status="active")


# ── Формирование контекста для lifecycle_prompt ───────────────────────────────

def build_change_context(existing_stack: dict, change_request: str) -> str:
    """
    Формирует контекст для lifecycle_prompt.txt когда пользователь
    хочет изменить существующий стэк.

    Возвращает строку которая подставляется в {existing_plan}
    в lifecycle_prompt.txt.
    """
    plan = existing_stack.get("plan_json", {})
    return json.dumps({
        "stack_id":    existing_stack["id"],
        "stack_name":  existing_stack["stack_name"],
        "intent":      existing_stack["intent"],
        "plan":        plan,
        "status":      existing_stack["status"],
    }, indent=2)
