"""
main.py — CLI точка входа для агента.

Использование:
  python main.py --intent "Create an S3 bucket for user photos"
  python main.py --list
  python main.py --destroy my-stack
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")


def cmd_plan(args):
    from orchestration.langgraph_graph import run_agent
    print(f"\n🚀 Запрос: {args.intent}\n")
    result = run_agent(args.intent)
    status = result.get("status")

    if status == "done":
        plan = result.get("plan_dict", {})
        print(f"\n✅ ГОТОВО")
        print(f"   Стэк:      {plan.get('stack_name')}")
        print(f"   Регион:    {plan.get('aws_region')}")
        print(f"   Ресурсов:  {len(plan.get('resources', []))}")
        for r in result.get("resources_created", []):
            print(f"     + {r}")
        print(f"   Токены:    {result.get('total_tokens', 0)}")
        print(f"   Стоимость: ${result.get('total_cost_usd', 0):.4f}")
        print(f"   Время:     {result.get('total_latency', 0):.1f}с")

    elif status == "needs_clarification":
        print(f"\n❓ Уточни запрос:\n   {result.get('clarification')}")

    elif status == "failed":
        print(f"\n❌ Ошибка: {result.get('error_message')}")
        sys.exit(1)


def cmd_list(args):
    from agent.lifecycle_manager import list_active_stacks
    stacks = list_active_stacks()
    if not stacks:
        print("\nНет активных стэков.")
        return
    print(f"\nАктивные стэки ({len(stacks)}):\n")
    for s in stacks:
        plan = s.get("plan_json", {})
        print(f"  {s['stack_name']}  (id: {s['id']})")
        print(f"    intent:   {s['intent'][:60]}")
        print(f"    ресурсов: {len(plan.get('resources', []))}")
        print(f"    создан:   {s['created_at'][:10]}\n")


def cmd_destroy(args):
    from agent.lifecycle_manager import get_stack_by_name, mark_destroyed
    from agent.deployer import destroy

    stack = get_stack_by_name(args.destroy)
    if not stack:
        print(f"❌ Стэк '{args.destroy}' не найден.")
        sys.exit(1)

    print(f"\n⚠  Удаление стэка '{args.destroy}'")
    if input("   Продолжить? [y/N]: ").strip().lower() != "y":
        print("   Отменено.")
        return

    workspace = Path(stack["workspace"]) if stack.get("workspace") else None
    if workspace and workspace.exists():
        if destroy(workspace, args.destroy):
            mark_destroyed(stack["id"])
            print(f"✅ Стэк '{args.destroy}' удалён.")
        else:
            print("❌ Ошибка при удалении.")
            sys.exit(1)
    else:
        print(f"❌ Workspace не найден: {workspace}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="AWS Infrastructure Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --intent "Create S3 bucket for photos"
  python main.py --list
  python main.py --destroy my-stack
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--intent", "-i", help="Запрос на естественном языке")
    group.add_argument("--list",   "-l", action="store_true", help="Список стэков")
    group.add_argument("--destroy","-d", metavar="NAME", help="Удалить стэк")

    args = parser.parse_args()
    if args.intent:  cmd_plan(args)
    elif args.list:  cmd_list(args)
    elif args.destroy: cmd_destroy(args)


if __name__ == "__main__":
    main()
