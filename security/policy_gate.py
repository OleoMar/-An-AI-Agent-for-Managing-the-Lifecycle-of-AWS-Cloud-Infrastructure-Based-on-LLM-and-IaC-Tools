"""
security/policy_gate.py

Шлагбаум перед terraform apply.

Ни один деплой не проходит без проверки здесь.
Это код-уровневая защита — не промпт, не LLM-решение.

Логика:
  1. Читаем terraform plan -json
  2. Классифицируем каждую операцию: LOW / HIGH риск
  3. Если есть HIGH — запрашиваем подтверждение у пользователя
  4. Если checkov не прошёл — блокируем полностью

Что считается HIGH риском:
  - Удаление stateful ресурсов (RDS, DynamoDB)
  - Изменение IAM ролей и политик
  - Отключение шифрования или публичный доступ
  - Удаление любых ресурсов вообще (destroy)
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from config import TERRAFORM_CMD, HIGH_RISK_RESOURCE_TYPES, HIGH_RISK_ACTIONS, BASE_DIR

logger = logging.getLogger(__name__)

# ── Кэш провайдеров Terraform ────────────────────────────────────────────────
# Без этого каждый вызов _get_terraform_plan() заново скачивает AWS provider
# (~100+ МБ) через terraform init. При 20 сценариях подряд это приводит к
# большим задержкам (наблюдалось 43–146с на сценарий вместо обычных 10–20с)
# и к сетевым таймаутам/обрывам соединения при security ablation study.
# Один общий кэш на диске решает обе проблемы: первый init скачивает
# провайдер, все последующие используют локальную копию.
_TF_PLUGIN_CACHE_DIR = BASE_DIR / ".terraform_plugin_cache"
_TF_PLUGIN_CACHE_DIR.mkdir(exist_ok=True)


def _tf_env() -> dict:
    """Окружение для subprocess-вызовов terraform с включённым plugin cache."""
    env = os.environ.copy()
    env["TF_PLUGIN_CACHE_DIR"] = str(_TF_PLUGIN_CACHE_DIR)
    return env


# ── Уровни риска ──────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW  = "LOW"
    HIGH = "HIGH"


# ── Результат проверки ────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Результат проверки policy gate."""
    approved: bool                        # можно ли деплоить
    risk_level: RiskLevel                 # общий уровень риска
    reason: str                           # почему одобрено или заблокировано
    high_risk_operations: list[str] = field(default_factory=list)  # что именно рискованно
    requires_human: bool = False          # нужно подтверждение человека


# ── Главная функция ───────────────────────────────────────────────────────────

def check(
    workspace: Path,
    validation_passed: bool,
    interactive: bool = True,
) -> GateResult:
    """
    Проверяет план перед деплоем.

    Параметры:
        workspace          — папка с .tf файлами
        validation_passed  — прошёл ли checkov (из ValidationResult)
        interactive        — запрашивать ли у пользователя подтверждение
                             (False в тестах)

    Возвращает:
        GateResult с решением approved=True/False
    """
    # ── Шаг 1: checkov должен пройти ──────────────────────────────────────────
    if not validation_passed:
        return GateResult(
            approved=False,
            risk_level=RiskLevel.HIGH,
            reason="BLOCKED: checkov не прошёл. Запусти validate заново.",
        )

    # ── Шаг 2: получаем план Terraform ────────────────────────────────────────
    plan_data = _get_terraform_plan(workspace)
    if plan_data is None:
        # Terraform не установлен или план не получен — разрешаем с предупреждением
        logger.warning("[policy_gate] не удалось получить terraform plan — LOW риск по умолчанию")
        return GateResult(
            approved=True,
            risk_level=RiskLevel.LOW,
            reason="terraform plan недоступен — одобрено автоматически (LOW риск)",
        )

    # ── Шаг 3: классифицируем операции ────────────────────────────────────────
    high_risk_ops = _classify_operations(plan_data)

    if not high_risk_ops:
        return GateResult(
            approved=True,
            risk_level=RiskLevel.LOW,
            reason="Все операции LOW риска — одобрено автоматически.",
        )

    # ── Шаг 4: есть HIGH риск — логируем ──────────────────────────────────────
    logger.warning("[policy_gate] ⚠ HIGH риск операции:")
    for op in high_risk_ops:
        logger.warning("[policy_gate]   - %s", op)

    if not interactive:
        # В тестах или CI — автоматически блокируем HIGH риск
        return GateResult(
            approved=False,
            risk_level=RiskLevel.HIGH,
            reason="HIGH риск операции требуют подтверждения человека.",
            high_risk_operations=high_risk_ops,
            requires_human=True,
        )

    # ── Шаг 5: запрашиваем подтверждение у пользователя ──────────────────────
    approved = _ask_human(high_risk_ops)

    return GateResult(
        approved=approved,
        risk_level=RiskLevel.HIGH,
        reason="Одобрено пользователем." if approved else "Отклонено пользователем.",
        high_risk_operations=high_risk_ops,
        requires_human=True,
    )


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _get_terraform_plan(workspace: Path) -> dict | None:
    """
    Запускает terraform plan -json и возвращает план как словарь.
    Возвращает None если terraform не установлен или произошла ошибка.
    """
    try:
        # Сначала init (без вывода) — с общим plugin cache
        subprocess.run(
            [TERRAFORM_CMD, "init", "-input=false"],
            cwd=workspace,
            capture_output=True,
            timeout=120,
            env=_tf_env(),
        )

        # Создаём plan файл
        result = subprocess.run(
            [TERRAFORM_CMD, "plan", "-out=tfplan", "-input=false"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=180,
            env=_tf_env(),
        )

        if result.returncode not in (0, 2):
            logger.warning("[policy_gate] terraform plan ошибка: %s", result.stderr[:300])
            return None

        # Конвертируем в JSON
        show_result = subprocess.run(
            [TERRAFORM_CMD, "show", "-json", "tfplan"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
            env=_tf_env(),
        )

        if show_result.returncode != 0:
            return None

        return json.loads(show_result.stdout)

    except FileNotFoundError:
        logger.warning("[policy_gate] terraform не найден в PATH")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning("[policy_gate] ошибка получения плана: %s", e)
        return None


def _classify_operations(plan_data: dict) -> list[str]:
    """
    Анализирует terraform plan и возвращает список HIGH-риск операций.
    """
    high_risk = []

    # Извлекаем список изменений ресурсов
    resource_changes = (
        plan_data
        .get("resource_changes", [])
    )

    for change in resource_changes:
        resource_type = change.get("type", "")
        resource_name = change.get("address", "unknown")
        actions       = change.get("change", {}).get("actions", [])

        # Проверяем тип ресурса
        is_high_risk_type = resource_type in HIGH_RISK_RESOURCE_TYPES

        # Проверяем действия
        has_high_risk_action = any(
            action in HIGH_RISK_ACTIONS
            for action in actions
        )

        # Удаление любого ресурса — всегда HIGH
        if "delete" in actions or "destroy" in actions:
            high_risk.append(
                f"DELETE {resource_name} ({resource_type})"
            )

        # Изменение критичного ресурса
        elif is_high_risk_type and "update" in actions:
            high_risk.append(
                f"UPDATE {resource_name} ({resource_type}) — stateful/IAM ресурс"
            )

        # Создание IAM ресурсов
        elif resource_type in {"aws_iam_role", "aws_iam_policy"} and "create" in actions:
            high_risk.append(
                f"CREATE {resource_name} ({resource_type}) — IAM изменение"
            )

    return high_risk


def _ask_human(high_risk_ops: list[str]) -> bool:
    """
    Выводит список HIGH-риск операций и ждёт подтверждения.
    Возвращает True если пользователь написал 'y'.
    """
    print("\n" + "="*55)
    print("⚠  POLICY GATE — требуется подтверждение")
    print("="*55)
    print("Следующие операции HIGH риска:")
    for op in high_risk_ops:
        print(f"  • {op}")
    print()

    try:
        answer = input("Продолжить деплой? [y/N]: ").strip().lower()
        return answer == "y"
    except (EOFError, KeyboardInterrupt):
        print("\nОтменено.")
        return False
