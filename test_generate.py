# test_generate.py
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(message)s')

from schemas.plan_schema import Plan, EXAMPLE_PLAN_SIMPLE
from agent.iac_generator import generate_terraform

plan = Plan(**EXAMPLE_PLAN_SIMPLE)
workspace, calls = generate_terraform(plan)

print(f"\nworkspace: {workspace}")
print(f"токенов: {sum(c.total_tokens for c in calls)}")
print("\n--- main.tf ---")
print((workspace / 'main.tf').read_text())