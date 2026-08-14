"""
agent/iac_generator.py

Стадия 2a: объект Plan → файлы Terraform HCL на диске.

Что происходит:
  1. Получаем Plan из State
  2. Создаём папку terraform_workspace/{request_id}/
  3. Загружаем generator_prompt.txt или correction_prompt.txt (при retry)
  4. Вызываем LLM → получаем HCL код
  5. Записываем main.tf (и при необходимости variables.tf, outputs.tf)
  6. Возвращаем путь к папке и список вызовов LLM для метрик

Отдельные файлы:
  main.tf       — все ресурсы AWS
  variables.tf  — переменные (aws_region, stack_name)
  outputs.tf    — outputs если LLM их сгенерировала
  provider.tf   — настройка AWS провайдера (фиксированная, не от LLM)
"""

import re
import logging
import uuid
from pathlib import Path
from typing import Optional

from agent.llm_client import call_llm, load_prompt, LLMResponse
from schemas.plan_schema import Plan
from config import TERRAFORM_WORKSPACE, AWS_REGION

logger = logging.getLogger(__name__)


# ── Фиксированный provider.tf ─────────────────────────────────────────────────
# Не генерируется LLM — всегда одинаковый, безопасный.

PROVIDER_TF = '''terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
'''


# ── Главная функция ───────────────────────────────────────────────────────────

def generate_terraform(
    plan: Plan,
    request_id: Optional[str] = None,
    validation_errors: Optional[list[str]] = None,
    previous_tf_code: Optional[str] = None,
) -> tuple[Path, list[LLMResponse]]:
    """
    Генерирует Terraform файлы для плана.

    Параметры:
        plan               — валидный Plan от intent_parser
        request_id         — уникальный ID запроса (создаётся автоматически)
        validation_errors  — ошибки от предыдущей валидации (для retry)
        previous_tf_code   — предыдущий код (для correction_prompt при retry)

    Возвращает:
        (workspace_path, llm_calls)
        workspace_path — Path к папке с .tf файлами
        llm_calls      — список вызовов LLM для метрик
    """
    # Создаём уникальную рабочую папку для этого запроса
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    workspace = TERRAFORM_WORKSPACE / f"{plan.stack_name}-{request_id}"
    workspace.mkdir(parents=True, exist_ok=True)

    logger.info("[generate] рабочая папка: %s", workspace)

    # Выбираем промпт: первая попытка или correction
    is_retry = validation_errors is not None and previous_tf_code is not None

    if is_retry:
        # Разделяем ошибки tflint и checkov для correction_prompt
        tflint_errors, checkov_errors = _split_errors(validation_errors)

        prompt = load_prompt(
            "correction_prompt.txt",
            original_tf_code=previous_tf_code,
            tflint_errors=tflint_errors or "none",
            checkov_errors=checkov_errors or "none",
        )
        call_type = "correct"
        logger.info("[generate] режим: correction (было %d ошибок)", len(validation_errors))
    else:
        prompt = load_prompt(
            "generator_prompt.txt",
            plan_json=plan.model_dump_json(indent=2),
            aws_region=plan.aws_region or AWS_REGION,
            stack_name=plan.stack_name,
        )
        call_type = "generate"
        logger.info("[generate] режим: первая генерация")

    # Вызов LLM
    response = call_llm(prompt, call_type=call_type)
    llm_calls = [response]

    # Извлекаем и записываем файлы
    hcl_code = _clean_hcl(response.text)
    _write_terraform_files(workspace, plan, hcl_code)

    logger.info("[generate] ✓ файлы записаны в %s", workspace)
    return workspace, llm_calls


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _clean_hcl(text: str) -> str:
    """
    Убирает markdown-обёртки если LLM их добавила.
    Возвращает чистый HCL код.
    """
    text = text.strip()

    # Убираем ```hcl ... ``` или ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])       # убираем первую строку (```hcl)
        text = text.rsplit("```", 1)[0]   # убираем закрывающие ```
        text = text.strip()

    return text


def _write_terraform_files(workspace: Path, plan: Plan, hcl_code: str) -> None:
    """
    Разбивает HCL код на файлы и записывает их в workspace.

    Стратегия:
    - Если LLM вернула отдельные блоки variable/output — выносим их в
      variables.tf и outputs.tf соответственно
    - Всё остальное идёт в main.tf
    - provider.tf всегда фиксированный (не от LLM)
    """
    # provider.tf — всегда фиксированный
    (workspace / "provider.tf").write_text(PROVIDER_TF, encoding="utf-8")

    # Разделяем блоки по типам
    variable_blocks = _extract_blocks(hcl_code, "variable")
    output_blocks   = _extract_blocks(hcl_code, "output")
    main_code       = _remove_blocks(hcl_code, ["variable", "output"])

    # main.tf — ресурсы и всё остальное
    (workspace / "main.tf").write_text(main_code.strip() + "\n", encoding="utf-8")

    # variables.tf — если есть переменные
    if variable_blocks:
        (workspace / "variables.tf").write_text(
            "\n\n".join(variable_blocks) + "\n",
            encoding="utf-8"
        )

    # outputs.tf — если есть outputs
    if output_blocks:
        (workspace / "outputs.tf").write_text(
            "\n\n".join(output_blocks) + "\n",
            encoding="utf-8"
        )

    files = [f.name for f in workspace.iterdir()]
    logger.info("[generate] записаны файлы: %s", files)


def _extract_blocks(hcl: str, block_type: str) -> list[str]:
    """
    Извлекает все блоки заданного типа из HCL кода.

    Например, _extract_blocks(code, "variable") вернёт все
    блоки вида:  variable "имя" { ... }
    """
    pattern = rf'(^{block_type}\s+"[^"]+"\s*\{{[^}}]*\}})'
    matches = re.findall(pattern, hcl, re.MULTILINE | re.DOTALL)
    return matches


def _remove_blocks(hcl: str, block_types: list[str]) -> str:
    """
    Удаляет блоки заданных типов из HCL кода.
    Используется чтобы не дублировать variable/output в main.tf.
    """
    for block_type in block_types:
        pattern = rf'^{block_type}\s+"[^"]+"\s*\{{[^}}]*\}}'
        hcl = re.sub(pattern, "", hcl, flags=re.MULTILINE | re.DOTALL)
    return hcl


def _split_errors(errors: list[str]) -> tuple[str, str]:
    """
    Разделяет список ошибок на tflint и checkov.

    Ошибки checkov начинаются с CKV_ или CKV2_.
    Всё остальное — tflint.
    """
    tflint = []
    checkov = []

    for err in errors:
        if err.startswith("CKV"):
            checkov.append(err)
        else:
            tflint.append(err)

    return "\n".join(tflint), "\n".join(checkov)


def read_workspace_code(workspace: Path) -> str:
    """
    Читает main.tf из workspace и возвращает как строку.
    Используется validator.py и reconciler.py.
    """
    main_tf = workspace / "main.tf"
    if not main_tf.exists():
        raise FileNotFoundError(f"main.tf не найден в {workspace}")
    return main_tf.read_text(encoding="utf-8")
