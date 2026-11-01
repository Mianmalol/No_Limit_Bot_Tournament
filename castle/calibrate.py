"""Fit a low-dimensional mixture of field components to the published score curve.

FAILED its synthetic check, and very different mixtures fit the real curve equally
well, so its picks aren't evidence. Kept as a record.

Unlike the full reconstructions (invert_g1, transport), which fail their synthetic
checks, this fits about 25 weights. Validation truths use clusters absent from the
component library, so the check can't pass by construction.
"""
import json
import sys

import numpy as np

from field import ARCHETYPES, jitter, random_allocs
from riddler import best_response, scores
from stress import CONCENTRATED
from stress2 import sharp_g1

V5 = np.arange(1, 6)
ROUNDS = {int(k[1:]): v for k, v in np.load("results/riddler_merged.npz").items()}
REAL = np.vstack(list(ROUNDS.values()))
SIZE = 3478
GRID = (np.arange(200) + 0.5) / 200
FAMILIES = [(6, 16, 1, 36, 41), (6, 5, 3, 42, 44), (7, 20, 36, 0, 37), (1, 16, 1, 41, 41),
            (0, 0, 33, 33, 34), (0, 25, 25, 25, 25), (10, 30, 0, 60, 0)]


def noisy(base, n, rng, moves=3):
    return np.vstack([jitter(np.array(base), rng.poisson(moves), rng)[None, :] for _ in range(n)])


def library(rng, n=SIZE):
    pools = {f"riddler{k}": v[rng.integers(len(v), size=n)] for k, v in ROUNDS.items()}
    for name, base, _ in ARCHETYPES:
        pools[name] = noisy(base, n, rng)
    pools["concentrated"] = np.vstack([noisy(CONCENTRATED[rng.integers(8)], 1, rng) for _ in range(n)])
    pools["equilibrium"] = sharp_g1(rng, n)
    pools["random"] = random_allocs(n, rng)
    for f in FAMILIES:
        pools[f"near{f}"] = noisy(f, n, rng, moves=2)
    return pools


def compose(pools, weights):
    names = list(pools)
    counts = np.floor(weights * SIZE).astype(int)
    counts[np.argmax(weights)] += SIZE - counts.sum()
    return np.vstack([pools[k][:c] for k, c in zip(names, counts) if c])


def curve(field):
    return np.quantile(scores(field, field, V5), GRID)


def calibrate(target, pools, rng, steps=1500):
    k = len(pools)
    w = rng.dirichlet(np.ones(k))
    err = np.abs(curve(compose(pools, w)) - target).mean()
    for _ in range(steps):
        trial = w * np.exp(rng.normal(0, 0.35, k) * (rng.random(k) < 0.3))
        trial /= trial.sum()
        e = np.abs(curve(compose(pools, trial)) - target).mean()
        if e < err:
            w, err = trial, e
    return w, err


def ensemble(target, rng, runs=6):
    pools = library(rng)
    return [(calibrate(target, pools, rng), pools) for _ in range(runs)]


def show_weights(pools, w, top=7):
    names = list(pools)
    return ", ".join(f"{names[i]} {w[i]:.2f}" for i in np.argsort(-w)[:top])


def validate():
    rng = np.random.default_rng(8)
    print("Synthetic check. Scores on the truth")
    print("field              fit   true-best  consensus  naive  old-pick")
    cases = {
        "hidden clusters": np.vstack([REAL[rng.integers(len(REAL), size=1400)], sharp_g1(rng, 700),
                                      noisy((9, 9, 18, 30, 34), 700, rng),
                                      noisy((2, 2, 30, 30, 36), 678, rng)]),
        "sharp heavy": np.vstack([sharp_g1(rng, 1800), noisy((5, 14, 2, 38, 41), 900, rng),
                                  REAL[rng.integers(len(REAL), size=778)]]),
    }
    naive = best_response(REAL, V5)
    old = np.array([6, 5, 3, 42, 44])
    for name, truth in cases.items():
        res = ensemble(curve(truth), rng, runs=4)
        fields = [compose(p, w) for (w, _), p in res]
        pick = best_response(np.vstack(fields), V5)
        val = lambda x: scores(np.array(x)[None, :], truth, V5)[0]
        print(f"{name:16s} {np.mean([e for (_, e), _ in res]):6.3f} "
              f"{val(best_response(truth, V5)):9.3f} {val(pick):9.3f} {val(naive):7.3f} {val(old):7.3f}",
              flush=True)


def main():
    validate()
    if "--validate" in sys.argv:
        return
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_scores.json"))["g1"]])
    target = np.quantile(pub, GRID)
    res = ensemble(target, np.random.default_rng(11), runs=8)
    fields = []
    print("\nReal Game 1 calibration")
    for (w, err), pools in res:
        f = compose(pools, w)
        fields.append(f)
        own = scores(f, f, V5)
        print(f"fit {err:.3f} top {own.max():.2f} mean {f.mean(0).round(1)}")
        print("   ", show_weights(pools, w))
    pick = best_response(np.vstack(fields), V5)
    print("consensus BR", tuple(int(v) for v in pick))
    for name, c in (("consensus", pick), ("old (6,5,3,42,44)", (6, 5, 3, 42, 44)),
                    ("(6,16,1,36,41)", (6, 16, 1, 36, 41)), ("(3,16,1,37,43)", (3, 16, 1, 37, 43))):
        vals = np.array([scores(np.array(c)[None, :], f, V5)[0] for f in fields])
        print(f"{name:20s} predicted {vals.mean():.3f} range {vals.min():.3f}-{vals.max():.3f}, "
              f"real percentile at mean {(pub < vals.mean()).mean():.3f}")
    np.savez("results/g1_calibrated.npz", *fields)


if __name__ == "__main__":
    main()
