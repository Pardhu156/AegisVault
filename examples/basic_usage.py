"""Minimal AegisVault Stage 1 example."""

from __future__ import annotations

from pathlib import Path

from aegisvault import AegisVault


def ecommerce_assistant(prompt: str) -> str:
    return f"Demo application received: {prompt}"


def drifting_assistant(prompt: str) -> str:
    return "Here is Python code for merge sort: def merge_sort(items): return items"


def main() -> None:
    policy_path = Path(__file__).resolve().parents[1] / "policies" / "ecommerce.yaml"
    guard = AegisVault.from_policy(policy_path)
    protected_app = guard.wrap(ecommerce_assistant)
    protected_drifting_app = guard.wrap(drifting_assistant)

    examples = [
        ("In-domain request", protected_app, "Where is my order?"),
        ("Out-of-domain request", protected_app, "Write Python code for merge sort."),
        ("Ambiguous request", protected_app, "Can you help me calculate something?"),
        ("Out-of-domain generated response", protected_drifting_app, "Can you help me with my order?"),
    ]

    for label, app, prompt in examples:
        result = app(prompt)
        print(f"\n{label}")
        print(f"Prompt: {prompt}")
        print(f"Application called: {result.application_called}")
        print(f"Terminated by: {result.terminated_by.value}")
        print(f"Final response: {result.final_response}")


if __name__ == "__main__":
    main()
