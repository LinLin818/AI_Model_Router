"""Build a real training set: run every prompt through both models and have a judge score the answers.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python scripts/collect_data.py --prompts my_prompts.txt --out data/real.jsonl

--prompts: .txt (one prompt per line) or .jsonl with a "prompt" field.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic

from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()

JUDGE_PROMPT = """You are grading an AI assistant's answer.

[Question]
{q}

[Answer]
{a}

Score the answer from 0 to 10 for correctness, completeness and helpfulness.
Reply with only the integer."""


def ask(model: str, prompt: str, max_tokens: int = 1024) -> str:
    r = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")



def judge(model: str, question: str, answer: str):
    m = re.search(r"\d+", ask(model, JUDGE_PROMPT.format(q=question, a=answer), max_tokens=10))
    return min(int(m.group()), 10) / 10 if m else None


def process(prompt: str, a):
    try:
        cheap_answer = ask(a.cheap_model, prompt)
        strong_answer = ask(a.strong_model, prompt)
        cs = judge(a.judge_model, prompt, cheap_answer)
        ss = judge(a.judge_model, prompt, strong_answer)
        if cs is None or ss is None:
            return None
        return {"prompt": prompt, "cheap_score": cs, "strong_score": ss}
    except Exception as e:  # keep going; one bad prompt shouldn't kill the run
        print(f"skipped one prompt: {e}")
        return None


def read_prompts(path: str):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if path.endswith(".jsonl"):
        return [json.loads(l)["prompt"] for l in lines]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)

    ap.add_argument("--out", default="data/real.jsonl")
    ap.add_argument("--cheap-model", default="claude-haiku-5-5")
    ap.add_argument("--strong-model", default="claude-sonnet-5-5")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    prompts = read_prompts(a.prompts)
    with ThreadPoolExecutor(a.workers) as ex:
        results = [r for r in ex.map(lambda p: process(p, a), prompts) if r]
    with open(a.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)}/{len(prompts)} rows to {a.out}")


if __name__ == "__main__":
    main()