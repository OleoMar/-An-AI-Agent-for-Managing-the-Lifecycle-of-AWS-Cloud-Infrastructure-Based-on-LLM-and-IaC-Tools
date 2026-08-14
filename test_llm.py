# test_llm.py  — запускать: python test_llm.py
import sys
sys.path.insert(0, '.')

from agent.llm_client import call_llm

r = call_llm("Say hello in one word.", "parse")
print("Ответ:  ", r.text)
print("Токены: ", r.total_tokens)
print("Цена:   $", r.cost_usd)