"""Cost / quality evaluation for routing policies.

Terminology
  quality    - mean judged score of the answers a policy would actually serve
  retention  - quality / quality of "always use the strong model"
  rel_cost   - mean cost / cost of "always use the strong model"
"""
from __future__ import annotations

import numpy as np


def policy_metrics(use_cheap, cheap_score, strong_score, cheap_cost, strong_cost) -> dict:
    use_cheap = np.asarray(use_cheap, dtype=bool)
    c, s = np.asarray(cheap_score, float), np.asarray(strong_score, float)
    quality = np.where(use_cheap, c, s).mean()
    cost = np.where(use_cheap, cheap_cost, strong_cost).mean()
    return {
        "cheap_share": float(use_cheap.mean()),
        "quality": float(quality),
        "retention": float(quality / s.mean()),
        "rel_cost": float(cost / strong_cost),
    }


def sweep(p_cheap_ok, cheap_score, strong_score, cheap_cost, strong_cost, thresholds=None) -> list[dict]:
    """Metrics for 'route to cheap iff p >= t' across thresholds. t=1.01 means always-strong."""
    p = np.asarray(p_cheap_ok, float)
    if thresholds is None:
        thresholds = list(np.round(np.linspace(0, 1, 101), 2)) + [1.01]
    out = []
    for t in thresholds:

        m = policy_metrics(p >= t, cheap_score, strong_score, cheap_cost, strong_cost)
        m["threshold"] = float(t)
        out.append(m)
    return out


def pick_threshold(curve: list[dict], target_retention: float) -> dict:
    """Cheapest operating point that keeps at least `target_retention` of strong-model quality."""
    feasible = [r for r in curve if r["retention"] >= target_retention]
    if not feasible:  # target above 1.0 etc. -> fall back to always-strong
        return curve[-1]
    return min(feasible, key=lambda r: (r["rel_cost"], -r["retention"]))


def format_table(rows: dict[str, dict]) -> str:
    head = f"{'policy':<22}{'cheap share':>12}{'quality':>10}{'retention':>11}{'rel. cost':>11}"
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(
            f"{name:<22}{m['cheap_share']:>12.1%}{m['quality']:>10.3f}{m['retention']:>11.1%}{m['rel_cost']:>11.1%}"
        )
    return "\n".join(lines)