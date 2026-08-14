"""
app.py — веб-интерфейс агента.

Запуск: python app.py
Открыть в браузере: http://localhost:5000
"""

import sys
import logging
import traceback
sys.path.insert(0, '.')

from flask import Flask, render_template, request, jsonify

from orchestration.langgraph_graph import run_agent
from agent.clarification_engine import (
    get_clarification_questions,
    build_enriched_request,
)
from schemas.plan_schema import Plan

logging.basicConfig(level=logging.INFO, format='%(message)s')

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/clarify", methods=["POST"])
def api_clarify():
    """
    Шаг 1: анализирует запрос и возвращает список уточняющих вопросов.
    Если вопросов нет — возвращает {"questions": []}.
    """
    data = request.get_json()
    user_input = (data or {}).get("input", "").strip()

    if not user_input:
        return jsonify({"error": "Введи запрос"}), 400

    try:
        questions = get_clarification_questions(user_input)

        return jsonify({
            "questions": [
                {
                    "id":       q.id,
                    "question": q.question,
                    "type":     q.type,
                    "options":  q.options,
                    "default":  q.default,
                    "required": q.required,
                }
                for q in questions
            ]
        })

    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/plan", methods=["POST"])
def api_plan():
    """
    Шаг 2: запускает агента с обогащённым запросом.
    Принимает исходный запрос + ответы пользователя на уточняющие вопросы.
    """
    data = request.get_json()
    user_input = (data or {}).get("input", "").strip()
    answers    = (data or {}).get("answers", {})

    if not user_input:
        return jsonify({"error": "Введи запрос"}), 400

    try:
        # Обогащаем запрос ответами пользователя
        enriched_input = build_enriched_request(user_input, answers)

        result = run_agent(enriched_input)

        if result["status"] == "needs_clarification":
            return jsonify({
                "status":  "clarify",
                "message": result["clarification"],
            }), 200

        if result["status"] == "failed":
            return jsonify({
                "status":  "error",
                "message": result["error_message"],
            }), 500

        plan = Plan(**result["plan_dict"])

        return jsonify({
            "status":      "ok",
            "stack_name":  plan.stack_name,
            "description": plan.description,
            "aws_region":  plan.aws_region,
            "resources": [
                {
                    "name":        r.name,
                    "type":        r.type.value,
                    "depends_on":  r.depends_on,
                    "description": r.description,
                    "constraints": r.constraints.model_dump(exclude_none=True),
                }
                for r in plan.resources
            ],
            "estimated_cost": plan.estimated_cost_usd_per_month,
            "tokens_used":    result["total_tokens"],
            "cost_usd":       round(result["total_cost_usd"], 4),
            "latency_sec":    round(result["total_latency"], 1),
            "workspace":      result.get("workspace_path"),
            "gate_risk":      result.get("gate_risk"),
        })

    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
