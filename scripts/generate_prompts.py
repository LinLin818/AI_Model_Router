"""Generate a varied set of prompts for training the router.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python scripts/generate_prompts.py --n 1000 --out data/prompts.jsonl

Then feed the output to collect_data.py:
    python scripts/collect_data.py --prompts data/prompts.jsonl --out data/real.jsonl
"""
import argparse
import itertools
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic()

CATEGORIES = [
    "factual questions", "writing and editing (emails, essays, rewrites)", "summarization",
    "translation", "coding: writing functions", "coding: debugging and code review",
    "SQL and data analysis", "math word problems", "proofs and formal reasoning",
    "system design and architecture", "brainstorming and ideas", "advice (career, personal, health)",
    "explaining concepts", "comparing options and tradeoffs", "role-play and creative writing",
    "classification and data extraction", "planning and scheduling", "business and finance",
]
DIFFICULTIES = {
    "easy": "simple, short, one clear answer, a strong model is not needed",
    "medium": "moderately involved, needs some knowledge or a few steps",
    "hard": "complex, multi-step reasoning, or expert-level, easy to get wrong",

}
STYLES = [
    "terse and lowercase, like a quick chat message",
    "polite and detailed, with background context",
    "includes pasted text, code, or data to work on",
    "casual with typos",
    "formal and professional",
]

TEMPLATE = """Write {k} different prompts that real users might send to an AI assistant.

Category: {category}
Difficulty: {difficulty} ({difficulty_desc})
Style: {style}

Rules:
- Each prompt must be self-contained and realistic, with all needed details included.
- Make them different from each other in topic and wording.
- Do not number them or add commentary.
Reply with ONLY a JSON array of strings."""


def generate(job, model, k):
    category, difficulty, style = job
    msg = TEMPLATE.format(
        k=k, category=category, difficulty=difficulty,
        difficulty_desc=DIFFICULTIES[difficulty], style=style,
    )
    try:
        r = client.messages.create(
            model=model, max_tokens=4000,
            messages=[{"role": "user", "content": msg}],

        )
        text = "".join(b.text for b in r.content if b.type == "text")
        arr = json.loads(re.search(r"\[.*\]", text, re.S).group())
        return [
            {"prompt": p.strip(), "category": category, "difficulty_hint": difficulty}
            for p in arr if isinstance(p, str) and p.strip()
        ]
    except Exception as e:
        print(f"skipped a batch: {e}")
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="target number of unique prompts")
    ap.add_argument("--out", default="data/prompts.jsonl")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--per-call", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    combos = list(itertools.product(CATEGORIES, DIFFICULTIES, STYLES))
    seen, rows = set(), []

    # keep going in rounds until we reach the target (dedupe removes some)
    for round_no in range(10):
        if len(rows) >= a.n:
            break
        n_calls = max(1, (a.n - len(rows)) // a.per_call + 1)
        jobs = rng.sample(combos, min(n_calls, len(combos)))

        with ThreadPoolExecutor(a.workers) as ex:
            batches = list(ex.map(lambda j: generate(j, a.model, a.per_call), jobs))
        for batch in batches:
            for row in batch:
                key = re.sub(r"\W+", " ", row["prompt"].lower()).strip()
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        print(f"round {round_no + 1}: {len(rows)} unique prompts")
        if not any(batches):
            raise SystemExit("Every batch failed. Fix the error above before retrying.")

    rows = rows[: a.n]
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} prompts to {a.out}")


if __name__ == "__main__":
    main()