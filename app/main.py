import argparse
import json
from datetime import datetime
from pathlib import Path

from app.graph import build_graph

OUTPUT_DIR = Path("output")


def _serialize_state(state: dict) -> dict:
    serialized = dict(state)
    serialized["messages"] = [
        {"type": message.type, "content": message.content} for message in state["messages"]
    ]
    return serialized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the home assistant a question and get a routed, RAG-backed answer."
    )
    parser.add_argument("question", help="The homeowner's question")
    parser.add_argument(
        "--household-type",
        choices=["apartment", "house", "rental"],
        default="house",
        help="Household type (default: house)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    initial_state = {
        "household_type": args.household_type,
        "messages": [("human", args.question)],
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    run_dir = OUTPUT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(_serialize_state(final_state), indent=2), encoding="utf-8")

    print(final_state["messages"][-1].content)
    print(f"\nFull result written to {result_path}")


if __name__ == "__main__":
    main()
