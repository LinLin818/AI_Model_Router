AI Router Model

I created this project because I want to solve a personal problem I currently have. When I use Claude or any other types of AI models, often times my tokens ran out pretty fast. I am pretty sure you had experienced this some time in your life. One token is burned for every 0.75 words. Humans typically write about 10 to 15 words in a single sentence and 100 to 200 words for a paragraph. One uploaded short file burned around 50 to 500 tokens, while 100 page-PDF can consume tens of thousands of tokens. Imagine how much tokens you can burn up too? Ok, enough rambling. Let’s get to the point of the project. The core idea is to create a system that can optimize the cost of token usage by analyzing the input prompts,  define the difficulty level and routing them to the most suitable AI model. For example, a simple question could be handled by smaller, less expensive AI models, while more complex questions or reasoning tasks could be routed to a more capable model. The goal is to avoid using the advance models without sacrificing the quality of responses.

How it works:
1. Each prompt is turned into features (TF-IDF text features plus simple signals like length, code, and reasoning keywords).
2. A logistic regression model learns `P(cheap model is good enough | prompt)` from training data.
3. At inference time, the prompt goes to the cheap model if that probability clears a threshold. Otherwise it goes to the strong model.
4. The threshold is tuned on out-of-fold predictions, so it targets a chosen share of the strong model's quality (97% by default) at the lowest cost.

The router is compared against four policies: always strong, always cheap, the router, and an oracle (perfect routing, as an upper bound).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## Using real data

Training data is JSONL, one row per prompt:

```json
{"prompt": "...", "cheap_score": 0.8, "strong_score": 0.9}
```

`scripts/collect_data.py` builds this by running each prompt through two models and having a judge score the answers. It needs an Anthropic API key and makes paid API calls.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
python scripts/collect_data.py --prompts my_prompts.txt --out data/real.jsonl
python -m router.train --data data/real.jsonl
```



## Project structure

```
router/
  features.py   # prompt -> feature vector
  train.py      # training and threshold selection
  evaluate.py   # cost/quality evaluation
  router.py     # inference-time Router class
scripts/
  make_demo_data.py   # synthetic training data
  collect_data.py     # real data from live models
```

## Tuning

| Flag | What it does |
|---|---|
| `--target-retention` | Share of strong-model quality to keep. Lower saves more money. |
| `--margin` | How much worse than the strong model still counts as "good enough". |
| `--cheap-cost` / `--strong-cost` | Relative cost per call for each model. |
| `--backend embeddings` | Use sentence embeddings instead of TF-IDF. |

<img src="Screenshot 2026-10-27 at 10.34.49 PM.png" alt="App Screenshot" width="500">

## Ideas for next steps

- Try sentence embeddings or gradient boosting.
- Add more than two model tiers.
- Log routing decisions in production and retrain as traffic changes.
- Decrease latency for generating time