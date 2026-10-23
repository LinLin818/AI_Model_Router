"""Train the router.

    python -m router.train --data data/demo.jsonl --out artifacts/router.joblib

Input is JSONL with one row per prompt:
    {"prompt": "...", "cheap_score": 0.0-1.0, "strong_score": 0.0-1.0}

Label: the cheap model is "good enough" when cheap_score >= strong_score - margin.
The router learns P(good enough | prompt). The routing threshold is picked on
out-of-fold predictions (never on the test set) as the cheapest operating point that
still keeps `--target-retention` of always-strong quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from .evaluate import format_table, pick_threshold, policy_metrics, sweep
from .features import build_featurizer


def load_jsonl(path: str):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    prompts = [r["prompt"] for r in rows]
    cheap = np.array([r["cheap_score"] for r in rows], dtype=float)
    strong = np.array([r["strong_score"] for r in rows], dtype=float)
    return prompts, cheap, strong


def build_pipeline(backend: str, embedding_model: str) -> Pipeline:
    return Pipeline(
        [
            ("features", build_featurizer(backend, embedding_model)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/demo.jsonl")
    ap.add_argument("--out", default="artifacts/router.joblib")
    ap.add_argument("--backend", choices=["tfidf", "embeddings"], default="tfidf")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--margin", type=float, default=0.05, help="tolerated quality gap vs strong model")
    ap.add_argument("--target-retention", type=float, default=0.97, help="min share of strong-model quality to keep")
    ap.add_argument("--cheap-cost", type=float, default=1.0, help="relative cost per call, cheap model")
    ap.add_argument("--strong-cost", type=float, default=15.0, help="relative cost per call, strong model")
    ap.add_argument("--cheap-model", default="claude-haiku-5-5")
    ap.add_argument("--strong-model", default="claude-sonnet-5-5")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    prompts, cheap, strong = load_jsonl(a.data)

    y = (cheap >= strong - a.margin).astype(int)
    print(f"{len(prompts)} prompts | cheap model good enough on {y.mean():.1%}")

    idx = np.arange(len(prompts))
    tr, te = train_test_split(idx, test_size=a.test_size, random_state=a.seed, stratify=y)
    pick = lambda xs, ii: [xs[i] for i in ii]  # noqa: E731
    X_tr, X_te = pick(prompts, tr), pick(prompts, te)

    pipe = build_pipeline(a.backend, a.embedding_model)

    # 1) choose the threshold on out-of-fold predictions from the training split
    oof = cross_val_predict(pipe, X_tr, y[tr], cv=5, method="predict_proba")[:, 1]
    curve = sweep(oof, cheap[tr], strong[tr], a.cheap_cost, a.strong_cost)
    chosen = pick_threshold(curve, a.target_retention)
    threshold = chosen["threshold"]

    # 2) report on the held-out test split
    pipe.fit(X_tr, y[tr])
    p_te = pipe.predict_proba(X_te)[:, 1]
    args = (cheap[te], strong[te], a.cheap_cost, a.strong_cost)
    n = len(te)
    table = {
        "always strong": policy_metrics(np.zeros(n, bool), *args),
        "always cheap": policy_metrics(np.ones(n, bool), *args),
        f"router (t={threshold:.2f})": policy_metrics(p_te >= threshold, *args),
        "oracle (upper bound)": policy_metrics(y[te].astype(bool), *args),
    }
    print(f"\nHeld-out test set ({n} prompts), AUC of P(cheap ok) = {roc_auc_score(y[te], p_te):.3f}\n")
    print(format_table(table))

    # 3) refit on all data and save
    pipe.fit(prompts, y)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "threshold": threshold,
            "cheap_model": a.cheap_model,
            "strong_model": a.strong_model,
            "meta": {
                "backend": a.backend,
                "margin": a.margin,
                "target_retention": a.target_retention,
                "n_train": len(prompts),
                "test_table": table,
            },
        },
        a.out,
    )
    print(f"\nSaved router to {a.out}")


if __name__ == "__main__":
    main()