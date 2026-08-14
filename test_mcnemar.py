# test_mcnemar.py
import json, sys
sys.path.insert(0, '.')
from evaluation.metrics import ScenarioResult, mcnemar_test, compute_metrics

def load_results(path):
    data = json.load(open(path))
    return [ScenarioResult(
        scenario_id=s['id'], input=s['input'],
        success=s['success'], status=s['status'],
        generate_attempts=s['generate_attempts'],
        total_tokens=s['tokens'], total_cost_usd=s['cost_usd'],
        latency_sec=s['latency_sec'],
        had_security_violation=s['had_sec_violation'],
        error_message=s['error']
    ) for s in data['scenarios']]

langgraph = load_results('evaluation/results/run_langgraph_final.json')
react     = load_results('evaluation/results/run_react_final.json')

test = mcnemar_test(langgraph, react)
print(f"chi2:        {test['chi2']}")
print(f"p_value:     {test['p_value']}")
print(f"significant: {test['significant']}")
print(f"note:        {test['note']}")