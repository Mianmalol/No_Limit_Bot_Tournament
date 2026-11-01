"""Fit the Game 1 field to the score curve and real allocation-score pairs.

The component library includes zero-castle pools alongside the earlier components.
Fits are judged by predicting a real entry they were not given, not by how well they
match what they were given.
"""
import json
import sys

import numpy as np

import calibrate
from calibrate import GRID, compose, curve, library
from riddler import best_response, scores

V5 = np.arange(1, 6)
SIZE = calibrate.SIZE
ANCHORS = {(1, 26, 29, 43, 1): 8.0, (0, 0, 37, 6, 57): 8.1}
PICKS = {"pick 3,5,4,43,45": (3, 5, 4, 43, 45), "challenger 7,18,6,3,66": (7, 18, 6, 3, 66),
         "earlier 6,16,1,36,41": (6, 16, 1, 36, 41)}


def dirichlet_allocs(rng, alpha, n):
    """Whole-number allocations over the castles with non-zero alpha."""
    alpha = np.array(alpha, dtype=float)
    keep = alpha > 0
    raw = np.zeros((n, 5))
    raw[:, keep] = rng.dirichlet(alpha[keep], n) * 100
    out = np.floor(raw).astype(np.int64)
    for row, frac in zip(out, raw - out):
        idx = np.argsort(-frac)[: 100 - row.sum()]
        row[idx] += 1
    return out


def zero_pools(rng, n=SIZE):
    return {
        "zero 4+5, spread": dirichlet_allocs(rng, (3, 6, 9, 0, 0), n),
        "zero 4+5, tight": dirichlet_allocs(rng, (12, 30, 45, 0, 0), n),
        "zero 5": dirichlet_allocs(rng, (3, 6, 9, 12, 0), n),
        "zero 5, tight": dirichlet_allocs(rng, (8, 16, 30, 40, 0), n),
        "zero 4": dirichlet_allocs(rng, (3, 6, 9, 0, 15), n),
        "zero 3+5": dirichlet_allocs(rng, (5, 20, 0, 40, 0), n),
        "zero 3+4": dirichlet_allocs(rng, (5, 10, 0, 0, 40), n),
        "zero 1+2": dirichlet_allocs(rng, (0, 0, 9, 12, 15), n),
    }


def objective(field, target, anchors, strength):
    err = np.abs(curve(field) - target).mean()
    if anchors:
        a = np.array(list(anchors))
        err += strength * np.abs(scores(a, field, V5) - np.array(list(anchors.values()))).mean()
    return err


def fit(target, pools, anchors, rng, strength=1.0, steps=1500):
    k = len(pools)
    w = rng.dirichlet(np.ones(k))
    err = objective(compose(pools, w), target, anchors, strength)
    for _ in range(steps):
        trial = w * np.exp(rng.normal(0, 0.35, k) * (rng.random(k) < 0.3))
        trial /= trial.sum()
        e = objective(compose(pools, trial), target, anchors, strength)
        if e < err:
            w, err = trial, e
    return w, err


def holdout(target, pools, rng):
    keys = list(ANCHORS)
    print("Hold-out: predict a real entry the fit never saw (actual in brackets)")
    print("given                         predicting                 predictions from 5 fits")
    for given, hidden in (([], keys[0]), ([], keys[1]), ([keys[1]], keys[0]), ([keys[0]], keys[1])):
        anchors = {k: ANCHORS[k] for k in given}
        preds = []
        for _ in range(5):
            w, _ = fit(target, pools, anchors, rng)
            preds.append(scores(np.array([hidden]), compose(pools, w), V5)[0])
        print(f"{str(given or 'curve only'):30s}{str(hidden):20s}[{ANCHORS[hidden]}]  "
              f"{np.round(preds, 2)}  mean {np.mean(preds):.2f}", flush=True)


def main():
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_latest.json"))["g1"]])
    target = np.quantile(pub, GRID)
    rng = np.random.default_rng(1234)
    pools = {**library(rng), **zero_pools(rng)}
    holdout(target, pools, rng)
    if "--holdout" in sys.argv:
        return
    print("\nFull fit: curve plus both entries")
    fields = []
    for i in range(10):
        w, err = fit(target, pools, ANCHORS, rng, strength=2.0)
        f = compose(pools, w)
        fields.append(f)
        pred = scores(np.array(list(ANCHORS)), f, V5)
        own = scores(f, f, V5)
        print(f"err {err:.3f} entries {np.round(pred, 2)} top {own.max():.2f} "
              f"zero on 1-5 {np.round((f == 0).mean(0), 2)} mean {f.mean(0).round(0)}", flush=True)
        print("   ", calibrate.show_weights(pools, w, 6))
    best = best_response(np.vstack(fields), V5)
    print("\nconsensus best response", tuple(int(v) for v in best))
    rows = {**PICKS, "consensus": tuple(int(v) for v in best)}
    for name, c in rows.items():
        v = np.array([scores(np.array(c)[None, :], f, V5)[0] for f in fields])
        print(f"{name:26s} mean {v.mean():.3f} range {v.min():.2f}-{v.max():.2f}, "
              f"board percentile {(pub < v.mean()).mean():.3f}")
    np.savez("results/g1_anchored.npz", *fields)


if __name__ == "__main__":
    main()
