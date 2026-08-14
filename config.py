"""
config.py — все константы проекта в одном месте.
Менять настройки только здесь, не разбрасывать по файлам.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # загружает переменные из .env

# ── Пути ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TERRAFORM_WORKSPACE = BASE_DIR / "terraform_workspace"
STACK_REGISTRY_PATH = BASE_DIR / "storage" / "stack_registry.db"
PROMPTS_DIR = BASE_DIR / "prompts"

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL = "claude-sonnet-4-6"          # модель Anthropic
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MAX_TOKENS = 4096                    # максимум токенов в ответе
LLM_TEMPERATURE = 0                      # 0 = детерминированный вывод (лучше для JSON/кода)

# ── AWS ───────────────────────────────────────────────────────────────────────
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Bucket для audit log (нужно создать вручную один раз)
AUDIT_LOG_BUCKET = os.getenv("AUDIT_LOG_BUCKET", "my-agent-audit-logs")

# ── Лимиты цикла самокоррекции ────────────────────────────────────────────────
MAX_PARSE_RETRIES = 3        # сколько раз переспрашивать LLM при ParseError
MAX_VALIDATE_RETRIES = 3     # сколько раз generate→validate до FAIL
MAX_RECONCILE_RETRIES = 2    # сколько раз patch→validate→deploy при дрифте

# ── Валидация (внешние инструменты) ──────────────────────────────────────────
TFLINT_CMD = "tflint"        # должен быть в PATH
CHECKOV_CMD = "checkov"      # должен быть в PATH
TERRAFORM_CMD = "terraform"  # должен быть в PATH

# ── Policy Gate — типы операций высокого риска ───────────────────────────────
HIGH_RISK_RESOURCE_TYPES = {
    "aws_db_instance",
    "aws_rds_cluster",
    "aws_dynamodb_table",
    "aws_iam_role",
    "aws_iam_policy",
    "aws_iam_user",
    "aws_s3_bucket_public_access_block",
    "aws_kms_key",
}

HIGH_RISK_ACTIONS = {
    "delete",
    "destroy",
    "remove",
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_SCENARIOS_DIR = BASE_DIR / "evaluation" / "scenarios"
EVAL_RESULTS_DIR = BASE_DIR / "evaluation" / "results"
