"""Generate a SYNTHETIC demo dataset so the pipeline runs end to end.

The scores come from a made-up difficulty model, not from real LLM outputs, so the
numbers you get from training on this say nothing about real-world routing quality.
Replace it with data from scripts/collect_data.py.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

COUNTRIES = ["France", "Japan", "Brazil", "Kenya", "Canada", "Norway", "Chile", "India", "Egypt", "Peru"]
SENTS = [
    "the meeting got moved to thursday cuz the boss is out",
    "our team have shipped the feature last week",
    "i cant make it tomorow, sorry for the short notice",
    "the quarterly numbers looks better then expected",
    "please send me the report as soon as you can",
    "the new office has alot of natural light and plants",
]
WORDS = ["happy", "difficult", "quick", "important", "begin", "buy", "large", "angry"]
LANGS = ["Spanish", "German", "Italian", "Portuguese"]
UNITS = [("miles", "kilometers"), ("pounds", "kilograms"), ("gallons", "liters"), ("feet", "meters")]
PY_TASKS = [
    "checks whether a string is a palindrome",
    "merges two sorted lists",
    "counts word frequencies in a text file",
    "validates an email address with a regex",
    "flattens a nested list",
]
SQL_TASKS = [
    "find the top 5 customers by total order value",
    "list employees earning more than their department average",
    "count orders per month for the last year",
]
CONCEPT_PAIRS = [("a list", "a tuple"), ("TCP", "UDP"), ("a process", "a thread"), ("REST", "GraphQL"), ("SQL", "NoSQL")]
ASKS = ["a day off next Friday", "a small budget for a conference", "feedback on my last project"]
CLAIMS = [
    "the sum of the first n odd numbers is n^2 for every positive integer n",
    "there are infinitely many primes of the form 4k+3",
    "every connected graph with n vertices has at least n-1 edges",
    "sqrt(2) is irrational",
]
SYSTEMS = ["rate limiter", "job scheduler", "key-value store", "notification service", "leader-election service"]


def easy(r):
    return r.choice(
        [
            lambda: f"What is the capital of {r.choice(COUNTRIES)}?",
            lambda: f'Rewrite this more formally: "{r.choice(SENTS)}"',
            lambda: f'Give me three synonyms for "{r.choice(WORDS)}".',
            lambda: f'Translate "{r.choice(SENTS)}" into {r.choice(LANGS)}.',
            lambda: f"Summarize in one sentence: {r.choice(SENTS)}. {r.choice(SENTS)}.",
            lambda: "Convert {} {} to {}.".format(r.randint(2, 500), *r.choice(UNITS)),
        ]
    )(), r.uniform(0.02, 0.25)


def medium(r):
    return r.choice(
        [
            lambda: f"Write a Python function that {r.choice(PY_TASKS)}, with a couple of unit tests.",
            lambda: "Explain the difference between {} and {} and when to use each.".format(*r.choice(CONCEPT_PAIRS)),
            lambda: f"Draft a polite email to my manager asking for {r.choice(ASKS)}. Keep it under 150 words.",
            lambda: f"Write a SQL query to {r.choice(SQL_TASKS)}.",
        ]
    )(), r.uniform(0.28, 0.55)


def hard(r):
    return r.choice(
        [
            lambda: f"Prove that {r.choice(CLAIMS)}. Then generalize the result and explain where the argument breaks on edge cases.",
            lambda: (
                "This service deadlocks under load: two threads take locks in different orders and a retry loop "
                "holds a mutex. Debug it step by step and propose a redesign, analyzing the trade-offs."
            ),
            lambda: (
                f"Design a distributed {r.choice(SYSTEMS)} that handles {r.randint(5, 500)}k requests per second "
                "with strong consistency. Analyze failure modes, trade-offs and the complexity of each component."
            ),
            lambda: (
                f"Derive the closed form of T(n) = {r.randint(2, 5)}T(n/{r.randint(2, 4)}) + n^{r.randint(1, 3)}, "
                "justify each step, and compare it with an iterative implementation's runtime."
            ),
        ]
    )(), r.uniform(0.6, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/demo.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    r = random.Random(a.seed)
    rng = np.random.default_rng(a.seed)
    rows = []
    for _ in range(a.n):
        gen = r.choices([easy, medium, hard], weights=[0.5, 0.3, 0.2])[0]
        prompt, d = gen(r)
        if r.random() < 0.25:  # some prompts carry extra pasted context
            prompt += "\n\nContext: " + " ".join(r.choice(SENTS) + "." for _ in range(r.randint(2, 8)))
            d = min(d + 0.05, 1.0)
        cheap = float(np.clip(0.96 - 0.9 * max(0.0, d - 0.3) + rng.normal(0, 0.04), 0, 1))
        strong = float(np.clip(0.96 - 0.15 * max(0.0, d - 0.3) + rng.normal(0, 0.04), 0, 1))
        rows.append({"prompt": prompt, "cheap_score": round(cheap, 3), "strong_score": round(strong, 3)})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} synthetic rows to {a.out}")


if __name__ == "__main__":
    main()