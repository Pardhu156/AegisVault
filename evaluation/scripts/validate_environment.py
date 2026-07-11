"""Validate Stage 2 evaluation environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.scripts.eval_lib import DEFAULT_DOMAINS, DEFAULT_GEMINI_MODEL, load_domain_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gemini, Ollama, and AegisVault evaluation configuration.")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--policy-dir", type=Path, default=Path("evaluation/policies"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("evaluation/datasets"))
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = True

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("GEMINI_API_KEY: set")
    else:
        print("GEMINI_API_KEY: missing")
        ok = False

    try:
        assets = load_domain_assets(args.domains, args.policy_dir, args.dataset_dir)
        print(f"Policies and datasets: loaded for {', '.join(assets)}")
    except Exception as exc:
        print(f"Policies and datasets: failed: {exc}")
        return 1

    ollama_base_url = next(iter(assets.values())).policy.evaluator.base_url
    ollama_model = next(iter(assets.values())).policy.evaluator.model
    try:
        response = requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = {model.get("name", "").split(":")[0] for model in models}
        full_names = {model.get("name", "") for model in models}
        print(f"Ollama server: reachable at {ollama_base_url}")
        if ollama_model in model_names or any(name.startswith(f"{ollama_model}:") for name in full_names):
            print(f"Ollama evaluator model: found {ollama_model}")
        else:
            print(f"Ollama evaluator model: missing {ollama_model}; run `ollama pull {ollama_model}`")
            ok = False
    except Exception as exc:
        print(f"Ollama server: unavailable: {exc}")
        ok = False

    if api_key:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            result = client.models.generate_content(
                model=args.gemini_model,
                contents="Reply with the single word ok.",
                config={"temperature": 0},
            )
            if getattr(result, "text", None):
                print(f"Gemini model: reachable ({args.gemini_model})")
            else:
                print(f"Gemini model: reached but returned empty text ({args.gemini_model})")
                ok = False
        except Exception as exc:
            print(f"Gemini model: unavailable ({args.gemini_model}): {exc}")
            print("Set a different model with: export GEMINI_MODEL=\"your_model_name\"")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
