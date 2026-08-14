"""
tests/unit/test_intent_parser.py

Unit-тесты для agent/intent_parser.py.

Ключевой принцип: тесты НЕ вызывают реальный LLM API.
Вместо этого мы "подменяем" call_llm фиктивным ответом (mock).
Это позволяет:
  - запускать тесты без API ключа (в CI, у коллег)
  - тестировать конкретные сценарии (ошибки, retry, edge cases)
  - работать быстро (нет сетевых запросов)

Запуск:  python -m pytest tests/unit/test_intent_parser.py -v
"""

import json
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '.')

from agent.intent_parser import (
    parse_intent,
    _parse_response,
    ParseError,
    ClarificationNeeded,
)
from agent.llm_client import LLMResponse
from schemas.plan_schema import Plan


# ── Фабрика фейковых LLMResponse ─────────────────────────────────────────────

def make_llm_response(text: str, call_type: str = "parse") -> LLMResponse:
    """Создаёт фейковый LLMResponse с нужным текстом."""
    return LLMResponse(
        text=text,
        input_tokens=500,
        output_tokens=100,
        latency_seconds=1.5,
        model="claude-sonnet-4-6",
        call_type=call_type,
    )


# ── Готовые JSON-планы для тестов ─────────────────────────────────────────────

VALID_PLAN_JSON = json.dumps({
    "stack_name": "user-photos",
    "description": "S3 bucket for storing user profile photos",
    "aws_region": "us-east-1",
    "resources": [
        {
            "name": "photos-bucket",
            "type": "aws_s3_bucket",
            "depends_on": [],
            "description": "Stores user profile photos",
            "constraints": {
                "public": False,
                "tags": {"Environment": "dev", "ManagedBy": "agent"}
            }
        }
    ],
    "estimated_cost_usd_per_month": 2.0,
})

VALID_COMPLEX_PLAN_JSON = json.dumps({
    "stack_name": "nodejs-api",
    "description": "Node.js API with PostgreSQL database",
    "aws_region": "us-east-1",
    "resources": [
        {
            "name": "api-vpc",
            "type": "aws_vpc",
            "depends_on": [],
            "description": "VPC for the API stack",
            "constraints": {}
        },
        {
            "name": "api-server",
            "type": "aws_ec2_instance",
            "depends_on": ["api-vpc"],
            "description": "EC2 instance running Node.js",
            "constraints": {"instance_type": "t3.micro"}
        },
        {
            "name": "api-db",
            "type": "aws_db_instance",
            "depends_on": ["api-vpc"],
            "description": "PostgreSQL database",
            "constraints": {
                "engine": "postgres",
                "engine_version": "15.3",
                "storage_gb": 20
            }
        }
    ]
})

CLARIFICATION_JSON = json.dumps({
    "error": "Request is unclear. Please specify: what type of application?"
})


# ══════════════════════════════════════════════════════════════════════════════
# ГРУППА 1: тесты _parse_response (без mock, тестируем чистую логику)
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponse:

    def test_valid_json_returns_plan(self):
        """Валидный JSON → объект Plan."""
        result, error = _parse_response(VALID_PLAN_JSON)
        assert isinstance(result, Plan)
        assert result.stack_name == "user-photos"
        assert len(result.resources) == 1
        assert error == ""

    def test_valid_complex_plan_with_deps(self):
        """Сложный план с depends_on → Plan с правильными зависимостями."""
        result, error = _parse_response(VALID_COMPLEX_PLAN_JSON)
        assert isinstance(result, Plan)
        assert result.stack_name == "nodejs-api"
        assert len(result.resources) == 3
        server = result.get_resource("api-server")
        assert server is not None
        assert "api-vpc" in server.depends_on

    def test_clarification_json_returns_string(self):
        """LLM вернула {error: ...} → строка с пояснением."""
        result, error = _parse_response(CLARIFICATION_JSON)
        assert isinstance(result, str)
        assert "unclear" in result.lower()
        assert error == ""

    def test_markdown_wrapped_json_is_cleaned(self):
        """JSON обёрнутый в ```json ... ``` → парсится нормально."""
        wrapped = f"```json\n{VALID_PLAN_JSON}\n```"
        result, error = _parse_response(wrapped)
        assert isinstance(result, Plan)
        assert result.stack_name == "user-photos"

    def test_invalid_json_returns_none_with_error(self):
        """Сломанный JSON → (None, описание ошибки)."""
        result, error = _parse_response("This is not JSON at all")
        assert result is None
        assert "JSON" in error
        assert len(error) > 0

    def test_unknown_resource_type_returns_validation_error(self):
        """Несуществующий тип ресурса → (None, ошибка валидации)."""
        bad = json.dumps({
            "stack_name": "bad",
            "description": "test",
            "resources": [{
                "name": "thing",
                "type": "aws_unicorn_server",
                "depends_on": []
            }]
        })
        result, error = _parse_response(bad)
        assert result is None
        assert "Validation" in error

    def test_depends_on_nonexistent_resource_fails(self):
        """depends_on ссылается на несуществующий ресурс → ValidationError."""
        bad = json.dumps({
            "stack_name": "bad",
            "description": "test",
            "resources": [{
                "name": "server",
                "type": "aws_ec2_instance",
                "depends_on": ["db-that-does-not-exist"]
            }]
        })
        result, error = _parse_response(bad)
        assert result is None
        assert "db-that-does-not-exist" in error

    def test_empty_resources_list_fails(self):
        """Пустой список ресурсов → ValidationError."""
        bad = json.dumps({
            "stack_name": "empty",
            "description": "no resources",
            "resources": []
        })
        result, error = _parse_response(bad)
        assert result is None
        assert error != ""

    def test_duplicate_resource_names_fails(self):
        """Два ресурса с одинаковым именем → ValidationError."""
        bad = json.dumps({
            "stack_name": "dup",
            "description": "test",
            "resources": [
                {"name": "bucket", "type": "aws_s3_bucket", "depends_on": []},
                {"name": "bucket", "type": "aws_s3_bucket", "depends_on": []},
            ]
        })
        result, error = _parse_response(bad)
        assert result is None
        assert "bucket" in error


# ══════════════════════════════════════════════════════════════════════════════
# ГРУППА 2: тесты parse_intent (с mock — LLM не вызывается реально)
# ══════════════════════════════════════════════════════════════════════════════

class TestParseIntent:

    @patch("agent.intent_parser.call_llm")
    def test_success_on_first_attempt(self, mock_call):
        """LLM сразу возвращает валидный JSON → план с первой попытки."""
        mock_call.return_value = make_llm_response(VALID_PLAN_JSON)

        plan, calls = parse_intent("Create an S3 bucket for user photos")

        assert isinstance(plan, Plan)
        assert plan.stack_name == "user-photos"
        assert len(calls) == 1          # ровно один вызов LLM
        mock_call.assert_called_once()  # убеждаемся что вызвали ровно один раз

    @patch("agent.intent_parser.call_llm")
    def test_retry_on_bad_json_then_success(self, mock_call):
        """
        Первый ответ — сломан. Второй — валидный.
        → план получен на 2-й попытке, было 2 вызова LLM.
        """
        mock_call.side_effect = [
            make_llm_response("not json at all"),   # попытка 1 — провал
            make_llm_response(VALID_PLAN_JSON),     # попытка 2 — успех
        ]

        plan, calls = parse_intent("Create S3 bucket")

        assert isinstance(plan, Plan)
        assert len(calls) == 2
        assert mock_call.call_count == 2

    @patch("agent.intent_parser.call_llm")
    def test_raises_parse_error_after_max_retries(self, mock_call):
        """
        LLM возвращает сломанный JSON все 3 раза →
        ParseError после исчерпания попыток.
        """
        mock_call.return_value = make_llm_response("{ broken json }")

        with pytest.raises(ParseError) as exc_info:
            parse_intent("Deploy something")

        assert "3" in str(exc_info.value)     # упомянуто число попыток
        assert mock_call.call_count == 3       # было ровно 3 вызова

    @patch("agent.intent_parser.call_llm")
    def test_raises_clarification_needed(self, mock_call):
        """
        LLM вернула {"error": "..."} →
        ClarificationNeeded (не ParseError — это другой случай).
        """
        mock_call.return_value = make_llm_response(CLARIFICATION_JSON)

        with pytest.raises(ClarificationNeeded) as exc_info:
            parse_intent("Set up something for my app")

        assert "unclear" in exc_info.value.message.lower()
        assert mock_call.call_count == 1  # сразу поняли — не retry

    @patch("agent.intent_parser.call_llm")
    def test_stack_context_appended_to_request(self, mock_call):
        """
        Если передан stack_context — он добавляется в промпт.
        """
        mock_call.return_value = make_llm_response(VALID_PLAN_JSON)

        parse_intent(
            "Add a DynamoDB table",
            stack_context='{"stack_name": "existing-app"}',
        )

        # Проверяем что в промпте был контекст
        actual_prompt = mock_call.call_args[0][0]  # первый позиционный аргумент
        assert "EXISTING STACK CONTEXT" in actual_prompt
        assert "existing-app" in actual_prompt

    @patch("agent.intent_parser.call_llm")
    def test_llm_calls_accumulate_across_retries(self, mock_call):
        """
        После retry список calls содержит ВСЕ вызовы, не только последний.
        Это важно для подсчёта метрик (суммарные токены, стоимость).
        """
        mock_call.side_effect = [
            make_llm_response("not json"),       # попытка 1
            make_llm_response(VALID_PLAN_JSON),  # попытка 2
        ]

        plan, calls = parse_intent("Create S3 bucket")

        assert len(calls) == 2
        total_tokens = sum(c.total_tokens for c in calls)
        assert total_tokens == 1200  # 600 * 2 вызова

    @patch("agent.intent_parser.call_llm")
    def test_error_from_previous_attempt_added_to_next_prompt(self, mock_call):
        """
        На второй попытке промпт содержит ошибку из первой.
        Так LLM знает что именно исправить.
        """
        mock_call.side_effect = [
            make_llm_response("{ invalid }"),    # попытка 1 — плохой JSON
            make_llm_response(VALID_PLAN_JSON),  # попытка 2 — успех
        ]

        parse_intent("Create S3 bucket")

        # Второй вызов должен содержать "PREVIOUS ATTEMPT FAILED"
        second_call_prompt = mock_call.call_args_list[1][0][0]
        assert "PREVIOUS ATTEMPT FAILED" in second_call_prompt
