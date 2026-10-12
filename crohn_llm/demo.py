# -*- coding: utf-8 -*-
"""Run episodes through one model and prompting strategy; --dry-run prints the messages without calling a model.

Examples:
  python crohn_llm/demo.py --dry-run
  python crohn_llm/demo.py --model gpt55 --prompt v0_zero
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_client import load_dotenv, run_one
from paths import ROOT
from prompt import PROMPT_VERSIONS, build_messages
from run_experiment import MAX_CHARS, MODELS, TEMPERATURE, make_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, default=ROOT / "examples" / "synthetic_episodes.jsonl")
    ap.add_argument("--model", choices=list(MODELS), default="gpt55")
    ap.add_argument("--prompt", choices=PROMPT_VERSIONS, default="v0_zero")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    eps = [json.loads(l) for l in args.episodes.open(encoding="utf-8") if l.strip()]

    if args.dry_run:
        for m in build_messages(eps[0]["prompt_text"], args.prompt):
            print(f"##### {m['role']}\n{m['content']}\n")
        return

    load_dotenv()
    cfg = MODELS[args.model]
    backend = make_backend(cfg)
    for ep in eps:
        r = run_one(backend, cfg["model"], ep, MAX_CHARS, False, args.prompt, TEMPERATURE)
        print(json.dumps({k: r[k] for k in ("episode_id", "gold", "pred", "probability", "evidence",
                                            "differential", "api_error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
