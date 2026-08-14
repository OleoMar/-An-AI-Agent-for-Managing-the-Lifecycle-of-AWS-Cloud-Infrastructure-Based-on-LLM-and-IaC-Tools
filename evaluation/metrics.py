"""
evaluation/metrics.py

Метрики для оценки агента. Используются в дипломе.

Пять метрик:
  TCR  — Task Completion Rate: доля успешных сценариев
  ERD  — Error Recovery Depth: среднее число итераций self-correction
  LAT  — Latency: среднее время от запроса до готовой инфраструктуры
  COST — Token Cost: средняя стоимость в долларах за один сценарий
  SVR  — Security Violation Rate: доля сценариев с нарушениями безопасности
         на первой попытке (до correction)

Зачем McNemar's test:
  Сравниваем LangGraph vs ReAct на одних и тех же сценариях.
  Данные парные (один сценарий прогоняется дважды) и бинарные
  (success/fail). McNemar's test — правильный выбор для этого случая.
  t-test не подходит (нужны непрерывные данные), chi-square не подходит
  (предполагает независимые выборки).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional


# ── Результат одного сценария ─────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Результат прогона агента на одном сценарии."""
    scenario_id:       str
    input:             str
    success:           bool            # завершился ли сценарий успешно
    status:            str             # done / failed / needs_clarification
    generate_attempts: int             # ERD: сколько итераций generate→validate
    total_tokens:      int             # для COST
    total_cost_usd:    float           # для COST
    latency_sec:       float           # для LAT
    had_security_violation: bool       # для SVR: были ли ошибки checkov на первой попытке
    validation_errors: list[str] = field(default_factory=list)
    error_message:     Optional[str] = None
    stack_name:        Optional[str] = None
    resources_created: list[str] = field(default_factory=list)


# ── Агрегированные метрики ─────────────────────────────────────────────────────

@dataclass
class BenchmarkMetrics:
    """Агрегированные метрики по всем сценариям."""
    n_total:     int    # всего сценариев
    n_success:   int    # успешных

    tcr:  float         # Task Completion Rate = n_success / n_total
    erd:  float         # Error Recovery Depth = среднее generate_attempts
    lat:  float         # Latency (сек) = среднее по успешным
    cost: float         # Token Cost ($) = среднее по всем
    svr:  float         # Security Violation Rate = доля с checkov ошибками

    # Детали для анализа
    n_failed:         int = 0
    n_clarification:  int = 0
    n_security_violations: int = 0
    total_tokens:     int = 0
    total_cost_usd:   float = 0.0
    results:          list[ScenarioResult] = field(default_factory=list)

    def report(self) -> str:
        """Форматированный отчёт для вывода в консоль и диплом."""
        lines = [
            "=" * 55,
            "BENCHMARK RESULTS",
            "=" * 55,
            f"Scenarios:  {self.n_total} total | {self.n_success} passed | {self.n_failed} failed",
            "",
            f"TCR  (Task Completion Rate):    {self.tcr:.1%}",
            f"ERD  (Error Recovery Depth):    {self.erd:.2f} iterations avg",
            f"LAT  (Latency):                 {self.lat:.1f}s avg",
            f"COST (Token Cost):              ${self.cost:.4f} avg per scenario",
            f"SVR  (Security Violation Rate): {self.svr:.1%}",
            "",
            f"Total tokens used:  {self.total_tokens:,}",
            f"Total cost:         ${self.total_cost_usd:.4f}",
            "=" * 55,
        ]

        if self.n_failed > 0:
            lines.append("FAILED SCENARIOS:")
            for r in self.results:
                if not r.success:
                    lines.append(f"  ✗ [{r.scenario_id}] {r.input[:50]}")
                    if r.error_message:
                        lines.append(f"    → {r.error_message[:80]}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Для сохранения в JSON файл."""
        return {
            "n_total":   self.n_total,
            "n_success": self.n_success,
            "n_failed":  self.n_failed,
            "tcr":       round(self.tcr,  4),
            "erd":       round(self.erd,  4),
            "lat":       round(self.lat,  4),
            "cost":      round(self.cost, 6),
            "svr":       round(self.svr,  4),
            "total_tokens":   self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
        }


# ── Подсчёт метрик ────────────────────────────────────────────────────────────

def compute_metrics(results: list[ScenarioResult]) -> BenchmarkMetrics:
    """
    Вычисляет все пять метрик по списку результатов сценариев.
    """
    n = len(results)
    if n == 0:
        return BenchmarkMetrics(n_total=0, n_success=0,
                                tcr=0, erd=0, lat=0, cost=0, svr=0)

    success = [r for r in results if r.success]
    failed  = [r for r in results if not r.success]
    clarify = [r for r in results if r.status == "needs_clarification"]
    sec_viol = [r for r in results if r.had_security_violation]

    # TCR
    tcr = len(success) / n

    # ERD — среднее число попыток generate→validate по ВСЕМ сценариям
    erd = sum(r.generate_attempts for r in results) / n if n > 0 else 0

    # LAT — только по успешным (у failed latency неполная)
    lat = (sum(r.latency_sec for r in success) / len(success)
           if success else 0)

    # COST — по всем (даже failed тратят токены)
    total_cost = sum(r.total_cost_usd for r in results)
    cost = total_cost / n

    # SVR
    svr = len(sec_viol) / n

    return BenchmarkMetrics(
        n_total=n,
        n_success=len(success),
        n_failed=len(failed),
        n_clarification=len(clarify),
        n_security_violations=len(sec_viol),
        tcr=tcr,
        erd=erd,
        lat=lat,
        cost=cost,
        svr=svr,
        total_tokens=sum(r.total_tokens for r in results),
        total_cost_usd=total_cost,
        results=results,
    )


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(
    results_a: list[ScenarioResult],
    results_b: list[ScenarioResult],
) -> dict:
    """
    McNemar's test для сравнения LangGraph (A) vs ReAct (B)
    на одних и тех же сценариях.

    Почему McNemar's а не t-test:
      - Данные парные (один сценарий в обоих условиях)
      - Данные бинарные (success/fail), не непрерывные
      - McNemar's именно для этого — парные бинарные данные

    Возвращает словарь с:
      chi2      — статистика хи-квадрат
      p_value   — p-значение (< 0.05 → значимая разница)
      b         — сценарии где A успешен, B нет
      c         — сценарии где B успешен, A нет
      significant — True если p < 0.05
    """
    if len(results_a) != len(results_b):
        raise ValueError("Списки результатов должны быть одной длины")

    # Подсчёт b и c (discordant pairs)
    b = 0  # A=success, B=fail
    c = 0  # A=fail,    B=success

    for ra, rb in zip(results_a, results_b):
        if ra.success and not rb.success:
            b += 1
        elif not ra.success and rb.success:
            c += 1

    # McNemar statistic с continuity correction
    n = b + c
    if n == 0:
        return {"chi2": 0, "p_value": 1.0, "b": 0, "c": 0,
                "significant": False, "note": "нет расхождений"}

    # Формула с поправкой Йейтса (для малых выборок)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value из chi-square с df=1
    p_value = _chi2_p_value(chi2)

    return {
        "chi2":        round(chi2, 4),
        "p_value":     round(p_value, 4),
        "b":           b,
        "c":           c,
        "significant": p_value < 0.05,
        "note": (
            f"LangGraph лучше на {b} сценариях, ReAct лучше на {c}"
            if b != c else "одинаковые результаты"
        )
    }


def _chi2_p_value(chi2: float) -> float:
    """
    Приближённое p-value из chi-square распределения с df=1.
    Используем регуляризованную неполную гамма-функцию.
    """
    if chi2 <= 0:
        return 1.0
    # Для df=1: p = 1 - CDF(chi2) = erfc(sqrt(chi2/2)) / 1
    x = math.sqrt(chi2 / 2)
    return math.erfc(x)
