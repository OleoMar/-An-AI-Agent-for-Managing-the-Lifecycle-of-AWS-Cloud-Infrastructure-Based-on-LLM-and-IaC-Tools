"""
agent/clarification_engine.py

Определяет какие детали пропущены в запросе пользователя
и формирует список уточняющих вопросов.

Работает ДО основного pipeline — до parse_intent, generate и т.д.

Логика:
  1. Лёгкий LLM вызов чтобы определить типы ресурсов в запросе
  2. Для каждого типа — проверяем есть ли нужные детали в тексте
  3. Возвращаем список вопросов по недостающим деталям

Если все детали есть — возвращаем пустой список (вопросов нет).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from agent.llm_client import call_llm

logger = logging.getLogger(__name__)

# ── Структура вопроса ──────────────────────────────────────────────────────────

@dataclass
class ClarificationQuestion:
    """Один уточняющий вопрос."""
    id:       str            # уникальный ключ, напр. "region"
    question: str            # текст вопроса для пользователя
    type:     str            # "select" / "text"
    options:  list[str] = field(default_factory=list)  # варианты для select
    default:  Optional[str] = None   # дефолтное значение
    required: bool = True            # обязательный ли вопрос


# ── Промпт для определения ресурсов и пробелов ────────────────────────────────

_DETECT_PROMPT = """You are analyzing an AWS infrastructure request to find missing details.

User request: {user_request}

Identify:
1. What AWS resources are requested (e.g. S3 bucket, EC2, RDS, Lambda, VPC, SQS, SNS, DynamoDB)
2. What critical configuration details are MISSING from the request

Return ONLY valid JSON, nothing else:
{{
  "resources": ["s3", "ec2", "rds", "lambda", "vpc", "sqs", "sns", "dynamodb", "ebs", "iam"],
  "missing": {{
    "region": true/false,
    "ec2_instance_type": true/false,
    "ec2_purpose": true/false,
    "rds_engine": true/false,
    "rds_storage_gb": true/false,
    "rds_multi_az": true/false,
    "vpc_subnet_type": true/false,
    "vpc_port_access": true/false,
    "s3_access": true/false,
    "lambda_runtime": true/false,
    "lambda_trigger": true/false
  }}
}}

Rules:
- region: missing if user did not mention "us-east-1", "eu-north-1", "us-west-2" etc.
- ec2_instance_type: missing if user did not mention "t3.micro", "t3.small" etc.
- ec2_purpose: missing if user did not say what the EC2 will run
- rds_engine: missing if user did not mention postgres, mysql, aurora
- s3_access: missing ONLY if unclear whether bucket should be public or private
- Only mark as missing if the resource is actually requested

User request: {user_request}
"""


# ── Словарь вопросов по типам ─────────────────────────────────────────────────

_QUESTIONS = {
    "region": ClarificationQuestion(
        id="region",
        question="В каком регионе AWS создавать инфраструктуру?",
        type="select",
        options=["us-east-1 (US East, Вирджиния)", "eu-north-1 (Europe, Стокгольм)",
                 "eu-west-1 (Europe, Ирландия)", "us-west-2 (US West, Орегон)",
                 "ap-southeast-1 (Asia Pacific, Сингапур)"],
        default="eu-north-1 (Europe, Стокгольм)",
    ),
    "ec2_instance_type": ClarificationQuestion(
        id="ec2_instance_type",
        question="Какой тип EC2 инстанса нужен?",
        type="select",
        options=["t3.micro (1 vCPU, 1GB RAM — для тестов)",
                 "t3.small (2 vCPU, 2GB RAM — лёгкие приложения)",
                 "t3.medium (2 vCPU, 4GB RAM — веб-приложения)",
                 "t3.large (2 vCPU, 8GB RAM — средняя нагрузка)",
                 "t3.xlarge (4 vCPU, 16GB RAM — высокая нагрузка)"],
        default="t3.micro (1 vCPU, 1GB RAM — для тестов)",
    ),
    "ec2_purpose": ClarificationQuestion(
        id="ec2_purpose",
        question="Для чего используется EC2 инстанс?",
        type="select",
        options=["Веб-сервер (HTTP/HTTPS)",
                 "API сервер (бэкенд приложения)",
                 "Bastion host (SSH доступ в VPC)",
                 "Worker (фоновые задачи)",
                 "Другое"],
        default="Веб-сервер (HTTP/HTTPS)",
    ),
    "rds_engine": ClarificationQuestion(
        id="rds_engine",
        question="Какой движок базы данных?",
        type="select",
        options=["PostgreSQL 15.3",
                 "MySQL 8.0",
                 "Aurora PostgreSQL",
                 "Aurora MySQL"],
        default="PostgreSQL 15.3",
    ),
    "rds_storage_gb": ClarificationQuestion(
        id="rds_storage_gb",
        question="Сколько гигабайт хранилища для базы данных?",
        type="select",
        options=["20 GB (минимум)", "50 GB", "100 GB", "250 GB", "500 GB"],
        default="20 GB (минимум)",
    ),
    "rds_multi_az": ClarificationQuestion(
        id="rds_multi_az",
        question="Нужен ли Multi-AZ (отказоустойчивость)?",
        type="select",
        options=["Нет (dev/test окружение)",
                 "Да (production, автоматический failover)"],
        default="Нет (dev/test окружение)",
    ),
    "vpc_subnet_type": ClarificationQuestion(
        id="vpc_subnet_type",
        question="Какие подсети нужны в VPC?",
        type="select",
        options=["Только публичные (ресурсы доступны из интернета)",
                 "Только приватные (ресурсы только внутри VPC)",
                 "Публичные и приватные (стандартная архитектура)"],
        default="Публичные и приватные (стандартная архитектура)",
    ),
    "vpc_port_access": ClarificationQuestion(
        id="vpc_port_access",
        question="Какие порты открыть в Security Group?",
        type="select",
        options=["80 и 443 (HTTP/HTTPS — веб-сервер)",
                 "22 (SSH — только для управления)",
                 "80, 443 и 22 (веб + управление)",
                 "5432 (PostgreSQL)",
                 "3306 (MySQL)",
                 "Все закрыты (только внутренний трафик)"],
        default="80 и 443 (HTTP/HTTPS — веб-сервер)",
    ),
    "s3_access": ClarificationQuestion(
        id="s3_access",
        question="Какой должен быть доступ к S3 bucket?",
        type="select",
        options=["Приватный (только через AWS credentials)",
                 "Публичный (файлы доступны по URL)"],
        default="Приватный (только через AWS credentials)",
    ),
    "lambda_runtime": ClarificationQuestion(
        id="lambda_runtime",
        question="На каком языке написана Lambda функция?",
        type="select",
        options=["Python 3.12",
                 "Node.js 20.x",
                 "Java 21",
                 "Go 1.x",
                 "Ruby 3.3"],
        default="Python 3.12",
    ),
    "lambda_trigger": ClarificationQuestion(
        id="lambda_trigger",
        question="Что запускает Lambda функцию?",
        type="select",
        options=["SQS (обработка сообщений из очереди)",
                 "S3 (новый файл в bucket)",
                 "API Gateway (HTTP запросы)",
                 "EventBridge (расписание / события)",
                 "SNS (уведомления)",
                 "Вручную / другое"],
        default="SQS (обработка сообщений из очереди)",
    ),
}


# ── Главная функция ───────────────────────────────────────────────────────────

def get_clarification_questions(user_input: str) -> list[ClarificationQuestion]:
    """
    Анализирует запрос и возвращает список уточняющих вопросов.

    Возвращает [] если все детали уже указаны в запросе.
    """
    logger.info("[clarify] анализируем запрос: '%s'", user_input[:60])

    prompt = _DETECT_PROMPT.format(user_request=user_input)

    try:
        response = call_llm(prompt, call_type="clarify_detect")
        data = _parse_json_response(response.text)
    except Exception as e:
        logger.warning("[clarify] не удалось проанализировать запрос: %s", e)
        return []

    missing = data.get("missing", {})
    resources = data.get("resources", [])

    questions = []

    # Регион спрашиваем всегда если не указан
    if missing.get("region"):
        questions.append(_QUESTIONS["region"])

    # EC2 вопросы
    if "ec2" in resources:
        if missing.get("ec2_purpose"):
            questions.append(_QUESTIONS["ec2_purpose"])
        if missing.get("ec2_instance_type"):
            questions.append(_QUESTIONS["ec2_instance_type"])

    # RDS вопросы
    if "rds" in resources:
        if missing.get("rds_engine"):
            questions.append(_QUESTIONS["rds_engine"])
        if missing.get("rds_storage_gb"):
            questions.append(_QUESTIONS["rds_storage_gb"])
        if missing.get("rds_multi_az"):
            questions.append(_QUESTIONS["rds_multi_az"])

    # VPC вопросы
    if "vpc" in resources:
        if missing.get("vpc_subnet_type"):
            questions.append(_QUESTIONS["vpc_subnet_type"])
        if missing.get("vpc_port_access"):
            questions.append(_QUESTIONS["vpc_port_access"])

    # S3 вопросы
    if "s3" in resources:
        if missing.get("s3_access"):
            questions.append(_QUESTIONS["s3_access"])

    # Lambda вопросы
    if "lambda" in resources:
        if missing.get("lambda_runtime"):
            questions.append(_QUESTIONS["lambda_runtime"])
        if missing.get("lambda_trigger"):
            questions.append(_QUESTIONS["lambda_trigger"])

    logger.info("[clarify] вопросов: %d | ресурсов: %s", len(questions), resources)
    return questions


def build_enriched_request(user_input: str, answers: dict[str, str]) -> str:
    """
    Добавляет ответы пользователя к исходному запросу.
    Возвращает обогащённый запрос для parse_intent.
    """
    if not answers:
        return user_input

    # Маппинг id → читаемое описание для LLM
    _answer_templates = {
        "region":             lambda v: f"AWS region: {v.split('(')[0].strip()}",
        "ec2_instance_type":  lambda v: f"EC2 instance type: {v.split('(')[0].strip()}",
        "ec2_purpose":        lambda v: f"EC2 purpose: {v}",
        "rds_engine":         lambda v: f"Database engine: {v}",
        "rds_storage_gb":     lambda v: f"Database storage: {v.split(' ')[0]} GB",
        "rds_multi_az":       lambda v: f"Multi-AZ: {'yes' if 'Да' in v else 'no'}",
        "vpc_subnet_type":    lambda v: f"Subnet type: {v.split('(')[0].strip()}",
        "vpc_port_access":    lambda v: f"Security group ports: {v.split('(')[0].strip()}",
        "s3_access":          lambda v: f"S3 access: {'public' if 'Публичный' in v else 'private'}",
        "lambda_runtime":     lambda v: f"Lambda runtime: {v}",
        "lambda_trigger":     lambda v: f"Lambda trigger: {v.split('(')[0].strip()}",
    }

    details = []
    for key, value in answers.items():
        if key in _answer_templates and value:
            details.append(_answer_templates[key](value))

    if not details:
        return user_input

    enriched = (
        f"{user_input}\n\n"
        f"Additional details provided by user:\n"
        + "\n".join(f"- {d}" for d in details)
    )
    logger.info("[clarify] обогащённый запрос: %d символов", len(enriched))
    return enriched


def _parse_json_response(text: str) -> dict:
    """Парсит JSON из ответа LLM."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)
