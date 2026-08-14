"""
agent/deployer.py

Стадия 3: terraform apply → реальные ресурсы в AWS.

Что происходит:
  1. terraform init   — скачивает AWS провайдер
  2. terraform plan   — показывает что будет создано
  3. policy_gate      — проверяет риск (уже вызван в графе, но для audit)
  4. terraform apply  — создаёт ресурсы в AWS
  5. terraform show   — читает реальное состояние после деплоя
  6. audit log        — записывает в S3 что и когда было сделано

Результат:
  DeployResult с actual_state (реальное состояние в AWS после apply).
  Этот actual_state потом сравнивает reconciler.py.
"""

import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from config import (
    TERRAFORM_CMD,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AUDIT_LOG_BUCKET,
)

logger = logging.getLogger(__name__)


# ── Структура результата ──────────────────────────────────────────────────────

@dataclass
class DeployResult:
    """Результат terraform apply."""
    success:      bool
    stack_name:   str
    workspace:    Path
    actual_state: dict              # вывод terraform show -json
    outputs:      dict              # terraform outputs
    applied_at:   str               # ISO timestamp
    duration_sec: float
    error:        Optional[str] = None
    resources_created: list[str] = field(default_factory=list)


# ── Главная функция ───────────────────────────────────────────────────────────

def deploy(
    workspace: Path,
    stack_name: str,
    request_id: str,
) -> DeployResult:
    """
    Запускает полный цикл terraform: init → plan → apply → show.

    Параметры:
        workspace  — Path к папке с .tf файлами
        stack_name — имя стэка (для audit log)
        request_id — уникальный ID запроса

    Возвращает:
        DeployResult с actual_state и списком созданных ресурсов
    """
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace не найден: {workspace}")

    t_start = time.perf_counter()
    applied_at = datetime.now(timezone.utc).isoformat()

    logger.info("[deploy] ── начинаем деплой стэка '%s'", stack_name)
    logger.info("[deploy] workspace: %s", workspace)

    # ── Шаг 1: terraform init ─────────────────────────────────────────────────
    init_ok = _run_terraform_init(workspace)
    if not init_ok:
        return DeployResult(
            success=False, stack_name=stack_name, workspace=workspace,
            actual_state={}, outputs={}, applied_at=applied_at,
            duration_sec=time.perf_counter() - t_start,
            error="terraform init завершился с ошибкой",
        )

    # ── Шаг 2: terraform apply ────────────────────────────────────────────────
    apply_ok, apply_error = _run_terraform_apply(workspace)
    if not apply_ok:
        return DeployResult(
            success=False, stack_name=stack_name, workspace=workspace,
            actual_state={}, outputs={}, applied_at=applied_at,
            duration_sec=time.perf_counter() - t_start,
            error=apply_error,
        )

    # ── Шаг 3: читаем реальное состояние ─────────────────────────────────────
    actual_state = _get_terraform_state(workspace)
    outputs      = _get_terraform_outputs(workspace)
    resources    = _extract_resource_addresses(actual_state)

    duration = time.perf_counter() - t_start
    logger.info(
        "[deploy] ✓ деплой завершён | ресурсов: %d | время: %.1fс",
        len(resources), duration,
    )
    for r in resources:
        logger.info("[deploy]   + %s", r)

    result = DeployResult(
        success=True,
        stack_name=stack_name,
        workspace=workspace,
        actual_state=actual_state,
        outputs=outputs,
        applied_at=applied_at,
        duration_sec=round(duration, 1),
        resources_created=resources,
    )

    # ── Шаг 4: audit log в S3 ────────────────────────────────────────────────
    _write_audit_log(result, request_id)

    return result


def destroy(workspace: Path, stack_name: str) -> bool:
    """
    Удаляет все ресурсы стэка через terraform destroy.
    Возвращает True если успешно.

    ВНИМАНИЕ: необратимая операция. Вызывается только после
    подтверждения пользователя в policy_gate (HIGH risk).
    """
    logger.warning("[deploy] ⚠ DESTROY стэка '%s'", stack_name)

    try:
        result = subprocess.run(
            [TERRAFORM_CMD, "destroy", "-auto-approve", "-input=false"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=600,
            env=_aws_env(),
        )
        success = result.returncode == 0
        if success:
            logger.info("[deploy] ✓ destroy завершён")
        else:
            logger.error("[deploy] ✗ destroy ошибка: %s", result.stderr[:300])
        return success

    except Exception as e:
        logger.error("[deploy] destroy исключение: %s", e)
        return False


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _run_terraform_init(workspace: Path) -> bool:
    """terraform init — скачивает провайдер AWS."""
    logger.info("[deploy] terraform init...")
    try:
        result = subprocess.run(
            [TERRAFORM_CMD, "init", "-input=false", "-no-color"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=180,
            env=_aws_env(),
        )
        if result.returncode != 0:
            logger.error("[deploy] init ошибка: %s", result.stderr[:300])
            return False
        logger.info("[deploy] init ✓")
        return True
    except FileNotFoundError:
        logger.error("[deploy] terraform не найден в PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("[deploy] init timeout")
        return False


def _run_terraform_apply(workspace: Path) -> tuple[bool, Optional[str]]:
    """terraform apply — создаёт ресурсы в AWS."""
    logger.info("[deploy] terraform apply...")
    try:
        result = subprocess.run(
            [
                TERRAFORM_CMD, "apply",
                "-auto-approve",   # не спрашивает "yes" интерактивно
                "-input=false",
                "-no-color",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=600,           # 10 минут — RDS может долго создаваться
            env=_aws_env(),
        )

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            logger.error("[deploy] apply ошибка: %s", error_msg)
            return False, error_msg

        logger.info("[deploy] apply ✓")
        return True, None

    except FileNotFoundError:
        return False, "terraform не найден в PATH"
    except subprocess.TimeoutExpired:
        return False, "terraform apply timeout (>10 минут)"


def _get_terraform_state(workspace: Path) -> dict:
    """
    terraform show -json — читает реальное состояние ресурсов после apply.
    Возвращает полный JSON объект с values.root_module.resources.
    """
    try:
        result = subprocess.run(
            [TERRAFORM_CMD, "show", "-json", "-no-color"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
            env=_aws_env(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        logger.warning("[deploy] не удалось прочитать state: %s", e)
        return {}


def _get_terraform_outputs(workspace: Path) -> dict:
    """terraform output -json — читает outputs после apply."""
    try:
        result = subprocess.run(
            [TERRAFORM_CMD, "output", "-json", "-no-color"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            env=_aws_env(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return {}
    except Exception:
        return {}


def _extract_resource_addresses(state: dict) -> list[str]:
    """
    Извлекает список адресов всех созданных ресурсов из terraform state.
    Например: ["aws_s3_bucket.photos-bucket", "aws_vpc.main-vpc"]
    """
    resources = []
    try:
        root = state.get("values", {}).get("root_module", {})
        for r in root.get("resources", []):
            address = r.get("address", "")
            if address:
                resources.append(address)
    except Exception:
        pass
    return resources


def _aws_env() -> dict:
    """
    Возвращает переменные окружения для subprocess с AWS credentials.
    Terraform читает credentials из env vars.
    """
    import os
    env = os.environ.copy()
    if AWS_ACCESS_KEY_ID:
        env["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    if AWS_SECRET_ACCESS_KEY:
        env["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    env["AWS_DEFAULT_REGION"] = AWS_REGION
    return env


def _write_audit_log(result: DeployResult, request_id: str) -> None:
    """
    Записывает JSON-лог деплоя в S3 bucket для audit trail.
    Не падает если S3 недоступен — просто логирует предупреждение.
    """
    log_entry = {
        "request_id":       request_id,
        "stack_name":       result.stack_name,
        "success":          result.success,
        "applied_at":       result.applied_at,
        "duration_sec":     result.duration_sec,
        "resources_created": result.resources_created,
        "error":            result.error,
    }
    key = f"deploys/{result.stack_name}/{result.applied_at[:10]}/{request_id}.json"

    try:
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3.put_object(
            Bucket=AUDIT_LOG_BUCKET,
            Key=key,
            Body=json.dumps(log_entry, indent=2),
            ContentType="application/json",
        )
        logger.info("[deploy] audit log → s3://%s/%s", AUDIT_LOG_BUCKET, key)

    except NoCredentialsError:
        logger.warning("[deploy] нет AWS credentials — audit log пропущен")
    except ClientError as e:
        logger.warning("[deploy] S3 ошибка: %s — audit log пропущен", e)
    except Exception as e:
        logger.warning("[deploy] audit log ошибка: %s", e)
