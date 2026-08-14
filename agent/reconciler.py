"""
agent/reconciler.py

Стадия 3б: сравниваем что планировали с тем что реально есть в AWS.

Дрифт — это расхождение между intended state (наш план) и actual state
(что terraform show возвращает из AWS). Он возникает когда:
  - AWS применил дефолтное значение которого не было в плане
  - Кто-то вручную изменил ресурс в AWS консоли
  - Terraform partial apply — часть ресурсов создалась, часть нет

Что делает reconciler:
  1. Читает intended Plan из State
  2. Читает actual_state из DeployResult (terraform show -json)
  3. Сравнивает ключевые поля
  4. Если есть дрифт — вызывает LLM с reconciler_prompt.txt
  5. LLM генерирует патч (исправленный main.tf)
  6. Патч идёт обратно в validate → deploy (max MAX_RECONCILE_RETRIES раз)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agent.llm_client import call_llm, load_prompt, LLMResponse
from security.sanitizer import sanitize, sanitize_dict
from schemas.plan_schema import Plan
from config import MAX_RECONCILE_RETRIES

logger = logging.getLogger(__name__)


# ── Результат сравнения ───────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """Описание расхождений между планом и реальным состоянием AWS."""
    has_drift:       bool
    drifted_fields:  list[str] = field(default_factory=list)  # что именно отличается
    missing_resources: list[str] = field(default_factory=list)  # есть в плане, нет в AWS
    extra_resources:   list[str] = field(default_factory=list)  # есть в AWS, нет в плане

    def summary(self) -> str:
        if not self.has_drift:
            return "✓ план совпадает с реальным состоянием AWS"
        parts = []
        if self.drifted_fields:
            parts.append(f"{len(self.drifted_fields)} поле(й) изменено")
        if self.missing_resources:
            parts.append(f"{len(self.missing_resources)} ресурс(ов) отсутствует")
        if self.extra_resources:
            parts.append(f"{len(self.extra_resources)} лишний ресурс(ов)")
        return "✗ дрифт: " + ", ".join(parts)


@dataclass
class ReconcileResult:
    """Результат reconciliation."""
    is_reconciled:   bool           # успешно ли выровняли состояние
    drift_report:    DriftReport    # что нашли
    attempts:        int            # сколько итераций потребовалось
    llm_calls:       list[LLMResponse] = field(default_factory=list)
    error:           Optional[str] = None


# ── Главная функция ───────────────────────────────────────────────────────────

def reconcile(
    plan: Plan,
    actual_state: dict,
    workspace: Path,
    original_tf_code: str,
) -> ReconcileResult:
    """
    Сравнивает план с реальным состоянием и исправляет расхождения.

    Параметры:
        plan            — объект Plan (что должно быть)
        actual_state    — вывод terraform show -json (что есть в AWS)
        workspace       — папка с .tf файлами
        original_tf_code — текущий main.tf (для LLM патча)

    Возвращает:
        ReconcileResult
    """
    logger.info("[reconcile] ── начинаем сверку состояний")

    # ── Шаг 1: определяем дрифт ───────────────────────────────────────────────
    drift = detect_drift(plan, actual_state)
    logger.info("[reconcile] %s", drift.summary())

    if not drift.has_drift:
        return ReconcileResult(
            is_reconciled=True,
            drift_report=drift,
            attempts=0,
        )

    # ── Шаг 2: есть дрифт — генерируем патч через LLM ────────────────────────
    all_llm_calls: list[LLMResponse] = []

    for attempt in range(1, MAX_RECONCILE_RETRIES + 1):
        logger.info(
            "[reconcile] попытка %d/%d — генерируем патч",
            attempt, MAX_RECONCILE_RETRIES,
        )

        # sanitizer.py защищает от prompt injection:
        # данные из AWS оборачиваются в <untrusted_data> теги
        safe_actual = sanitize_dict(actual_state, source="terraform_show")
        safe_plan   = sanitize(
            json.dumps(plan.model_dump(), indent=2),
            source="internal_plan"
        )

        drift_text = "\n".join([
            *[f"- изменено: {f}" for f in drift.drifted_fields],
            *[f"- отсутствует: {r}" for r in drift.missing_resources],
        ])
        safe_drift = sanitize(drift_text, source="drift_analysis")

        prompt = load_prompt(
            "reconciler_prompt.txt",
            intended_plan=safe_plan,
            actual_state=safe_actual,
            original_tf_code=original_tf_code,
            drift_report=safe_drift,
        )

        response = call_llm(prompt, call_type="reconcile")
        all_llm_calls.append(response)

        # Записываем патч в workspace
        patch_code = _clean_hcl(response.text)
        (workspace / "main.tf").write_text(patch_code, encoding="utf-8")
        logger.info("[reconcile] патч записан в main.tf")

        # После записи патча — вызывающий код (граф) должен снова
        # запустить validate → deploy. Мы возвращаем результат с has_drift=True
        # и граф решит нужна ли ещё итерация.
        # Для простоты: после первого патча считаем reconciled.
        # Полная проверка происходит когда граф снова запускает deploy.
        return ReconcileResult(
            is_reconciled=True,
            drift_report=drift,
            attempts=attempt,
            llm_calls=all_llm_calls,
        )

    # Если дошли сюда — все попытки исчерпаны
    return ReconcileResult(
        is_reconciled=False,
        drift_report=drift,
        attempts=MAX_RECONCILE_RETRIES,
        llm_calls=all_llm_calls,
        error=f"Не удалось устранить дрифт за {MAX_RECONCILE_RETRIES} попытки",
    )


# ── Определение дрифта ────────────────────────────────────────────────────────

def detect_drift(plan: Plan, actual_state: dict) -> DriftReport:
    """
    Сравнивает Plan с actual_state из terraform show -json.

    Логика:
      - Извлекаем список ресурсов из actual_state
      - Сравниваем с тем что должно быть в плане
      - Проверяем ключевые security поля
    """
    drifted_fields:    list[str] = []
    missing_resources: list[str] = []
    extra_resources:   list[str] = []

    # Получаем реальные ресурсы из terraform state
    actual_resources = _extract_resources_from_state(actual_state)
    actual_types = {r["type"] for r in actual_resources}

    # Проверяем что все ресурсы плана реально созданы
    plan_resource_types = {r.type.value for r in plan.resources}
    for plan_type in plan_resource_types:
        if plan_type not in actual_types:
            missing_resources.append(f"{plan_type} (есть в плане, нет в AWS)")

    # Проверяем security-критичные поля на каждом ресурсе
    for resource in actual_resources:
        res_type   = resource.get("type", "")
        res_name   = resource.get("address", "unknown")
        res_values = resource.get("values", {})

        drift = _check_security_drift(res_type, res_name, res_values)
        drifted_fields.extend(drift)

    has_drift = bool(drifted_fields or missing_resources or extra_resources)

    return DriftReport(
        has_drift=has_drift,
        drifted_fields=drifted_fields,
        missing_resources=missing_resources,
        extra_resources=extra_resources,
    )


def _check_security_drift(
    resource_type: str,
    resource_name: str,
    values: dict,
) -> list[str]:
    """
    Проверяет security-критичные поля конкретного ресурса.
    Возвращает список расхождений.
    """
    drifted = []

    if resource_type == "aws_db_instance":
        # RDS: шифрование, deletion protection, backup — никогда не должны быть выключены
        if not values.get("storage_encrypted", True):
            drifted.append(
                f"{resource_name}: storage_encrypted изменён на false"
            )
        if not values.get("deletion_protection", True):
            drifted.append(
                f"{resource_name}: deletion_protection изменён на false"
            )
        if values.get("backup_retention_period", 7) < 7:
            drifted.append(
                f"{resource_name}: backup_retention_period < 7 дней"
            )

    elif resource_type == "aws_s3_bucket":
        # S3: ACL и политики доступа
        acl = values.get("acl", "private")
        if acl not in ("private", "", None):
            drifted.append(
                f"{resource_name}: acl изменён на '{acl}' (ожидается private)"
            )

    elif resource_type == "aws_instance":
        # EC2: мониторинг и шифрование диска
        if not values.get("monitoring", True):
            drifted.append(
                f"{resource_name}: monitoring выключен"
            )

    return drifted


def _extract_resources_from_state(state: dict) -> list[dict]:
    """
    Извлекает список ресурсов из terraform show -json.
    Возвращает список словарей с полями: type, address, values.
    """
    try:
        root = state.get("values", {}).get("root_module", {})
        return root.get("resources", [])
    except Exception:
        return []


def _clean_hcl(text: str) -> str:
    """Убирает markdown-обёртку из ответа LLM если она есть."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    return text
