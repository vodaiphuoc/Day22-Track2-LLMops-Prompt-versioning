from __future__ import annotations

import subprocess
import sys


def run(step: str) -> None:
    print("\n" + "=" * 80)
    print(f"Running: {step}")
    print("=" * 80)
    subprocess.check_call([sys.executable, step])


def main() -> None:
    run("01_langsmith_rag_pipeline.py")
    run("02_prompt_hub_ab_routing.py")
    run("03_ragas_evaluation.py")
    run("04_guardrails_validator.py")
    print("\n✅ All steps completed.")


if __name__ == "__main__":
    main()

