"""
schemas/plan_schema.py

Pydantic-схемы для JSON-плана который LLM возвращает после парсинга запроса.

Зачем pydantic:
  Если LLM вернула JSON без обязательного поля, с неверным типом, или
  с ресурсом которого нет в нашем списке — pydantic бросит ValidationError
  с точным описанием что не так. Это сообщение идёт обратно в LLM как
  correction hint (correction_prompt.txt). Без pydantic мы бы узнали об
  ошибке только когда Terraform упал бы при apply.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ── 1. Разрешённые типы ресурсов ─────────────────────────────────────────────
# Закрытый список — LLM не может выдумать произвольный ресурс.
# Добавляй сюда по мере расширения проекта.

class ResourceType(str, Enum):
    # Вычисления
    EC2_INSTANCE       = "aws_ec2_instance"
    # Хранилище
    S3_BUCKET          = "aws_s3_bucket"
    EBS_VOLUME         = "aws_ebs_volume"
    # Сети
    VPC                = "aws_vpc"
    SUBNET             = "aws_subnet"
    SECURITY_GROUP     = "aws_security_group"
    INTERNET_GATEWAY   = "aws_internet_gateway"
    # Базы данных
    RDS_INSTANCE       = "aws_db_instance"
    DYNAMODB_TABLE     = "aws_dynamodb_table"
    # IAM
    IAM_ROLE           = "aws_iam_role"
    IAM_POLICY         = "aws_iam_policy"
    # Балансировщики
    ALB                = "aws_lb"
    ALB_TARGET_GROUP   = "aws_lb_target_group"
    # Lambda
    LAMBDA_FUNCTION    = "aws_lambda_function"
    # Очереди
    SQS_QUEUE          = "aws_sqs_queue"
    SNS_TOPIC          = "aws_sns_topic"


# ── 2. Ограничения на ресурс ──────────────────────────────────────────────────

class Constraint(BaseModel):
    """Необязательные лимиты и параметры для конкретного ресурса."""

    max_cost_usd_per_month: Optional[float] = Field(
        default=None,
        ge=0,
        description="Максимальная стоимость ресурса в месяц в долларах."
    )
    instance_type: Optional[str] = Field(
        default=None,
        description="Тип инстанса EC2, например 't3.micro'."
    )
    storage_gb: Optional[int] = Field(
        default=None,
        ge=1,
        le=65536,
        description="Размер хранилища в гигабайтах."
    )
    multi_az: Optional[bool] = Field(
        default=None,
        description="Использовать Multi-AZ (для RDS, ALB)."
    )
    public: Optional[bool] = Field(
        default=None,
        description="Ресурс должен быть публично доступен."
    )
    engine: Optional[str] = Field(
        default=None,
        description="Движок базы данных: 'postgres', 'mysql', 'aurora' и т.д."
    )
    engine_version: Optional[str] = Field(
        default=None,
        description="Версия движка, например '15.3'."
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="AWS теги ресурса."
    )


# ── 3. Один ресурс ────────────────────────────────────────────────────────────

class Resource(BaseModel):
    """Описание одного AWS-ресурса которого нужно создать."""

    name: str = Field(
        min_length=1,
        max_length=64,
        description="Логическое имя ресурса внутри плана. Только a-z, 0-9, дефис.",
        pattern=r'^[a-z][a-z0-9\-]*$'
    )
    type: ResourceType = Field(
        description="Тип ресурса — только из разрешённого списка ResourceType."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Имена других ресурсов из этого же плана от которых зависит этот."
    )
    constraints: Constraint = Field(
        default_factory=Constraint,
        description="Ограничения и параметры для этого ресурса."
    )
    description: str = Field(
        default="",
        max_length=256,
        description="Человекочитаемое описание зачем нужен этот ресурс."
    )

    @model_validator(mode='after')
    def check_depends_on_not_self(self) -> Resource:
        """Ресурс не может зависеть от самого себя."""
        if self.name in self.depends_on:
            raise ValueError(
                f"Ресурс '{self.name}' не может зависеть от самого себя в depends_on."
            )
        return self


# ── 4. Весь план ──────────────────────────────────────────────────────────────

class Plan(BaseModel):
    """
    Полный план инфраструктуры — выход intent_parser.py.
    Это то что LLM должна вернуть в виде JSON после разбора запроса.
    """

    stack_name: str = Field(
        min_length=1,
        max_length=64,
        description="Уникальное имя стэка. Только a-z, 0-9, дефис.",
        pattern=r'^[a-z][a-z0-9\-]*$'
    )
    description: str = Field(
        description="Краткое описание что делает весь стэк."
    )
    resources: list[Resource] = Field(
        min_length=1,
        description="Список ресурсов. Не может быть пустым."
    )
    aws_region: str = Field(
        default="us-east-1",
        description="Регион AWS для всех ресурсов плана."
    )
    estimated_cost_usd_per_month: Optional[float] = Field(
        default=None,
        ge=0,
        description="LLM может указать примерную стоимость всего стэка."
    )

    @model_validator(mode='after')
    def check_depends_on_valid(self) -> Plan:
        """
        Все имена в depends_on должны ссылаться на реальные ресурсы в этом плане.
        Защита от галлюцинаций LLM ('depends_on': ['database'] а ресурс назван 'db').
        """
        resource_names = {r.name for r in self.resources}
        for resource in self.resources:
            for dep in resource.depends_on:
                if dep not in resource_names:
                    raise ValueError(
                        f"Ресурс '{resource.name}' ссылается на '{dep}' в depends_on, "
                        f"но такого ресурса нет в плане. "
                        f"Доступные имена: {sorted(resource_names)}"
                    )
        return self

    @model_validator(mode='after')
    def check_unique_names(self) -> Plan:
        """Имена ресурсов в плане должны быть уникальными."""
        names = [r.name for r in self.resources]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Дублирующиеся имена ресурсов: {duplicates}. "
                f"Каждый ресурс должен иметь уникальное имя."
            )
        return self

    def get_resource(self, name: str) -> Optional[Resource]:
        """Найти ресурс по имени."""
        return next((r for r in self.resources if r.name == name), None)

    def resource_names(self) -> list[str]:
        """Список всех имён ресурсов."""
        return [r.name for r in self.resources]


# ── 5. Примеры для тестов и few-shot промптов ─────────────────────────────────

EXAMPLE_PLAN_SIMPLE = {
    "stack_name": "static-website",
    "description": "Статический сайт на S3",
    "aws_region": "us-east-1",
    "resources": [
        {
            "name": "website-bucket",
            "type": "aws_s3_bucket",
            "depends_on": [],
            "description": "S3 bucket для хранения статических файлов сайта",
            "constraints": {
                "public": True,
                "tags": {"Project": "thesis", "Environment": "dev"}
            }
        }
    ]
}

EXAMPLE_PLAN_WITH_DEPS = {
    "stack_name": "web-app",
    "description": "Веб-приложение с EC2 и RDS",
    "aws_region": "us-east-1",
    "resources": [
        {
            "name": "app-vpc",
            "type": "aws_vpc",
            "depends_on": [],
            "description": "Изолированная сеть для приложения",
            "constraints": {}
        },
        {
            "name": "app-subnet",
            "type": "aws_subnet",
            "depends_on": ["app-vpc"],
            "description": "Подсеть в VPC",
            "constraints": {}
        },
        {
            "name": "app-server",
            "type": "aws_ec2_instance",
            "depends_on": ["app-subnet", "app-db"],
            "description": "EC2 сервер приложения",
            "constraints": {"instance_type": "t3.micro"}
        },
        {
            "name": "app-db",
            "type": "aws_db_instance",
            "depends_on": ["app-subnet"],
            "description": "PostgreSQL база данных",
            "constraints": {
                "engine": "postgres",
                "engine_version": "15.3",
                "storage_gb": 20,
                "multi_az": False
            }
        }
    ]
}