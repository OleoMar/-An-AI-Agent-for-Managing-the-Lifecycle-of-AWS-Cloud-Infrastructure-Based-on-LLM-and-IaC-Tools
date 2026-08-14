"""
agent/validator.py

Стадия 2б: запускает tflint и checkov на сгенерированных .tf файлах.

Что происходит:
  1. Получаем путь к workspace с .tf файлами
  2. Запускаем tflint --format json → ловим синтаксические ошибки HCL
  3. Запускаем checkov -d . -o json → ловим нарушения security rules
  4. Парсим JSON-вывод обоих инструментов
  5. Возвращаем ValidationResult со списком всех ошибок

Важно: этот модуль НЕ вызывает LLM. Только внешние инструменты.
Решение "идти дальше или retry" принимает LangGraph граф на основе
поля passed в ValidationResult.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from config import TFLINT_CMD, CHECKOV_CMD

logger = logging.getLogger(__name__)


# ── Структура результата валидации ────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Результат валидации .tf файлов.
    passed=True означает: можно идти на деплой.
    passed=False означает: нужен retry через generate.
    """
    passed: bool
    tflint_errors: list[str] = field(default_factory=list)
    checkov_errors: list[str] = field(default_factory=list)

    @property
    def all_errors(self) -> list[str]:
        """Все ошибки в одном списке — для передачи в correction_prompt."""
        return self.tflint_errors + self.checkov_errors

    @property
    def error_count(self) -> int:
        return len(self.tflint_errors) + len(self.checkov_errors)

    def summary(self) -> str:
        if self.passed:
            return "✓ валидация прошла"
        return (
            f"✗ {self.error_count} ошибок: "
            f"{len(self.tflint_errors)} tflint + "
            f"{len(self.checkov_errors)} checkov"
        )


# ── Главная функция ───────────────────────────────────────────────────────────

def validate_terraform(workspace: Path) -> ValidationResult:
    """
    Запускает tflint и checkov на папке workspace.

    Параметры:
        workspace — Path к папке с .tf файлами

    Возвращает:
        ValidationResult с полным списком ошибок
    """
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace не найден: {workspace}")

    logger.info("[validate] проверяем: %s", workspace)

    tflint_errors  = _run_tflint(workspace)
    checkov_errors = _run_checkov(workspace)

    passed = len(tflint_errors) == 0 and len(checkov_errors) == 0

    result = ValidationResult(
        passed=passed,
        tflint_errors=tflint_errors,
        checkov_errors=checkov_errors,
    )

    logger.info("[validate] %s", result.summary())
    if not passed:
        for err in result.all_errors:
            logger.warning("[validate]   - %s", err)

    return result


# ── tflint ────────────────────────────────────────────────────────────────────

def _run_tflint(workspace: Path) -> list[str]:
    """
    Запускает tflint и возвращает список ошибок.
    Возвращает [] если ошибок нет или tflint не установлен.
    """
    try:
        result = subprocess.run(
            [TFLINT_CMD, "--format", "json", "--chdir", str(workspace)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        logger.warning("[validate] tflint не найден — пропускаем синтаксическую проверку")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("[validate] tflint timeout — пропускаем")
        return []

    # tflint возвращает 0 (нет ошибок) или 2 (есть ошибки)
    # код 1 — ошибка самого tflint (не наш код)
    if result.returncode == 1:
        logger.warning("[validate] tflint завершился с ошибкой: %s", result.stderr[:200])
        return []

    if not result.stdout.strip():
        return []

    return _parse_tflint_json(result.stdout)


def _parse_tflint_json(output: str) -> list[str]:
    """
    Парсит JSON-вывод tflint и возвращает читаемые сообщения об ошибках.

    Формат вывода tflint:
    {
      "issues": [
        {
          "rule": {"name": "aws_instance_invalid_type"},
          "message": "\"t1.micro\" is invalid instance type.",
          "range": {"filename": "main.tf", "start": {"line": 5}}
        }
      ]
    }
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        logger.warning("[validate] не удалось распарсить вывод tflint: %s", output[:200])
        return []

    errors = []
    issues = data.get("issues", [])

    for issue in issues:
        rule    = issue.get("rule", {}).get("name", "unknown")
        message = issue.get("message", "")
        loc     = issue.get("range", {}).get("start", {})
        line    = loc.get("line", "?")
        file_   = issue.get("range", {}).get("filename", "?")

        errors.append(f"tflint [{rule}] {file_}:{line} — {message}")

    return errors


# ── checkov ───────────────────────────────────────────────────────────────────

def _run_checkov(workspace: Path) -> list[str]:
    """
    Запускает checkov и возвращает список нарушений.
    Возвращает [] если нарушений нет или checkov не установлен.
    """
    try:
        result = subprocess.run(
            [
                CHECKOV_CMD,
                "--directory", str(workspace),
                "--output", "json",
                "--quiet",          # только failed checks, без passed
                "--compact",        # без деталей кода в выводе
                "--framework", "terraform",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        logger.warning("[validate] checkov не найден — пропускаем security проверку")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("[validate] checkov timeout — пропускаем")
        return []

    # checkov возвращает 0 (всё прошло) или 1 (есть нарушения)
    if not result.stdout.strip():
        return []

    return _parse_checkov_json(result.stdout)


def _parse_checkov_json(output: str) -> list[str]:
    """
    Парсит JSON-вывод checkov и возвращает список нарушений.

    Формат вывода checkov:
    {
      "results": {
        "failed_checks": [
          {
            "check_id": "CKV_AWS_19",
            "check": {"name": "Ensure S3 bucket has server side encryption"},
            "resource": "aws_s3_bucket.my-bucket",
            "file_path": "/workspace/main.tf",
            "file_line_range": [1, 5]
          }
        ]
      }
    }
    """
    # Checkov иногда выводит несколько JSON объектов — берём первый валидный
    output = output.strip()

    # Если это массив результатов — берём первый элемент
    if output.startswith("["):
        try:
            data_list = json.loads(output)
            data = data_list[0] if data_list else {}
        except json.JSONDecodeError:
            logger.warning("[validate] не удалось распарсить вывод checkov")
            return []
    else:
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            logger.warning("[validate] не удалось распарсить вывод checkov: %s", output[:200])
            return []

    errors = []
    results = data.get("results", {})
    failed  = results.get("failed_checks", [])

    for check in failed:
        check_id   = check.get("check_id", "?")
        check_name = check.get("check", {}).get("name", "")
        resource   = check.get("resource", "?")

        errors.append(f"{check_id}: {check_name} [{resource}]")

    return errors
