# test_parse.py
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(message)s')

from agent.intent_parser import parse_intent

plan, calls = parse_intent("Create an S3 bucket for storing user photos")
print(f"\nstack_name: {plan.stack_name}")
print(f"ресурсов: {len(plan.resources)}")
print(f"токенов: {sum(c.total_tokens for c in calls)}")