"""Interactive manual test: play the customer yourself and watch the agent respond.

This is for hands-on testing / recording a demo -- the official scoring path
is evaluator.local_evaluator, which simulates the customer automatically.
Here, YOU type the customer's messages.

Usage:
    python -m scripts.chat_demo
"""

from __future__ import annotations

from starter.agent import Agent

MAX_TURNS = 10
TOP_K = 10
SHOWN = 10  # how many of the top-10 recommendations to print per turn


def describe(index, parent_asin: str) -> str:
    product = index.products.get(parent_asin, {})
    title = str(product.get("title", "?"))[:70]
    price = product.get("price")
    price_text = f"${price}" if isinstance(price, (int, float)) else "price n/a"
    return f"{parent_asin}  {price_text:>10}  {title}"


def main() -> None:
    print("Loading catalog and building indexes (first run may take a few seconds)...")
    agent = Agent("data/catalog.jsonl")

    user_profile = {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.5,
        "rating_style": "usually positive",
        "preference_tags": ["comfort", "fit"],
        "summary": "Prior purchases emphasize comfort and fit; ratings are usually positive.",
    }
    session_id = "manual-test-session"
    agent.reset(session_id, user_profile)

    print("\nSession started. Type as the customer (or 'quit' to stop).")
    print(f"Demo user_profile: {user_profile}\n")

    for turn in range(1, MAX_TURNS + 1):
        message = input(f"[turn {turn}] you> ").strip()
        if message.lower() in ("quit", "exit"):
            break

        response = agent.respond(session_id, message, turn, TOP_K)
        diagnostics = agent.session_diagnostics(session_id)

        print(f"\nagent> {response['message']}")
        if response["ask_attribute"]:
            print(f"       (asking about: {response['ask_attribute']})")
        # The route and the pool size are what actually drive this turn, so
        # show them: they explain both the ranking blend and the decision to
        # ask (or not ask) a clarifying question.
        print(
            f"       route: {diagnostics['route']} "
            f"(buying weight {diagnostics['buying_weight']}), "
            f"candidate pool ~{diagnostics['pool_size']}"
        )
        if diagnostics["requirements"]:
            print(f"       heard so far: {'; '.join(diagnostics['requirements'])}")
        if diagnostics["hard_filters"]:
            filters = ", ".join(
                f"{attribute}={'/'.join(variants)}"
                for attribute, variants in diagnostics["hard_filters"].items()
            )
            print(f"       hard filters: {filters}")
        print(f"       top {SHOWN} recommendations:")
        for item in response["recommendations"][:SHOWN]:
            print("         " + describe(agent.index, item["parent_asin"]))
        print()

    print("Session ended.")


if __name__ == "__main__":
    main()
