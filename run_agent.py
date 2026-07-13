from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegisvault.agent_runtime import AgentRuntime, JsonlTraceLogger, OllamaChatClient, default_tool_registry
from aegisvault.agent_runtime.runtime import SYSTEM_PROMPT
from aegisvault.email_agent import EMAIL_AGENT_SYSTEM_PROMPT, build_email_tool_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Qwen tool-calling runtime.")
    parser.add_argument("prompt", nargs="?", help="Prompt to run once.")
    parser.add_argument("--task", help="Prompt/task to run once. Useful for the Email Agent CLI.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace", default="logs/agent_runtime_traces.jsonl")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--email-agent", action="store_true", help="Use Stage 4.1 synthetic email tools.")
    parser.add_argument("--dataset", default="datasets/email", help="Email dataset path when --email-agent is used.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_email_tool_registry(args.dataset) if args.email_agent else default_tool_registry()
    if args.list_tools:
        for tool in registry.list_tools():
            print(f"{tool.name}: {tool.description}")
        return 0
    runtime = AgentRuntime(
        client=OllamaChatClient(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            num_predict=args.num_predict,
        ),
        tools=registry,
        trace_logger=JsonlTraceLogger(args.trace),
        system_prompt=EMAIL_AGENT_SYSTEM_PROMPT if args.email_agent else SYSTEM_PROMPT,
    )
    if args.interactive:
        mode = "Stage 4.1 Email Agent" if args.email_agent else "Stage 4.0 runtime"
        print(f"AegisVault {mode} using {args.model}. Type 'exit' to quit.")
        while True:
            prompt = input("> ").strip()
            if prompt.lower() in {"exit", "quit"}:
                return 0
            _run_once(runtime, prompt, args.verbose)
    elif args.task or args.prompt:
        _run_once(runtime, args.task or args.prompt, args.verbose)
        return 0
    else:
        print("Provide a prompt or use --interactive.")
        return 2


def _run_once(runtime: AgentRuntime, prompt: str, verbose: bool) -> None:
    result = runtime.run(prompt)
    print(result.final_response)
    if verbose:
        print(json.dumps(result.trace.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
