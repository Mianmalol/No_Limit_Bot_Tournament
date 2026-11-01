"""Second reconstruction: move a population of entries until it earns the published scores.

FAILED its synthetic check: picks from this field lose to the plain Riddler field.
Kept as a record; don't use its output.

Each distinct published score owns one allocation (ties are copies of it). Sweeps nudge
allocations toward their target score under the population's own marginals. Fields
matching the scores aren't unique, so runs start from very different populations and the
final answer is a best response to all of them together. The check on synthetic fields
measures that pick on the truth.
"""
import json
import sys
from collections import Counter

import numpy as np
from numba import njit

from field import scenario
from invert_g1 import marginals
from riddler import best_response, scores
from stress2 import sharp_g1

V5 = np.arange(1, 6)
ROUNDS = {int(k[1:]): v for k, v in np.load("results/riddler_merged.npz").items()}
REAL = np.vstack(list(ROUNDS.values()))


@njit(cache=True)
def lookup(table, x):
    s = 0.0
    for c in range(5):
        s += table[c, x[c]]
    return s


@njit(cache=True)
def sweep(pop, counts, targets, table, share, rng_seed):
    """Moves a random share of allocations toward their target; returns mean abs gap."""
    np.random.seed(rng_seed)
    gap = 0.0
    n = pop.shape[0]
    for j in range(n):
        x = pop[j]
        best, best_err = x.copy(), abs(lookup(table, x) - targets[j])
        gap += best_err
        if np.random.random() > share:
            continue
        for _ in range(60):
            y = x.copy()
            for _ in range(np.random.randint(1, 3)):
                a, b = np.random.randint(5), np.random.randint(5)
                d = min(np.random.randint(1, 9), y[a])
                y[a] -= d
                y[b] += d
            err = abs(lookup(table, y) - targets[j])
            if err < best_err:
                best, best_err = y, err
        pop[j] = best
    return gap / n


def expand(pop, counts):
    return np.repeat(pop, counts, axis=0)


def fit(published, start, seed, sweeps=400):
    """Population matching the published scores, seeded by quantile-matching `start`."""
    rng = np.random.default_rng(seed)
    values = Counter(np.round(published, 3).tolist())
    targets = np.array(sorted(values))
    counts = np.array([values[t] for t in targets], dtype=np.int64)
    s = scores(start, start, V5)
    order = np.argsort(s)
    quantile = (np.cumsum(counts) - counts / 2) / counts.sum()
    pop = start[order[(quantile * (len(start) - 1)).astype(int)]].copy()
    gap = 9.0
    for k in range(sweeps):
        table = marginals(expand(pop, counts))
        gap = sweep(pop, counts, targets, table, 0.25, int(rng.integers(1 << 31)))
        if gap < 0.002:
            break
    return expand(pop, counts), gap


def starts(rng, n):
    """Deliberately different beliefs about what the field looks like."""
    return {
        "riddler": REAL[rng.integers(len(REAL), size=n)],
        "archetypes": scenario(rng, n),
        "equilibrium": sharp_g1(rng, n),
        "half+half": np.vstack([REAL[rng.integers(len(REAL), size=n // 2)], sharp_g1(rng, n - n // 2)]),
    }


def solve(published, rng, seeds=2):
    fields, gaps = [], []
    for name, start in starts(rng, 3000).items():
        for s in range(seeds):
            f, gap = fit(published, start, int(rng.integers(1 << 31)))
            fields.append(f)
            gaps.append(gap)
    return fields, gaps


def consensus(fields):
    pooled = np.vstack(fields)
    return best_response(pooled, V5)


def validate():
    rng = np.random.default_rng(6)
    print("Synthetic check. Columns: scores on the truth")
    print("field              gap   true-best  consensus  naive  old-pick")
    cases = {
        "archetypes": scenario(rng, 3478),
        "two clusters": np.vstack([scenario(rng, 2800), np.tile([6, 5, 3, 42, 44], (300, 1)),
                                   np.tile([1, 16, 1, 41, 41], (378, 1))]),
        "sharp half": np.vstack([scenario(rng, 1700), sharp_g1(rng, 1778)]),
    }
    naive = best_response(REAL, V5)
    old = np.array([6, 5, 3, 42, 44])
    for name, truth in cases.items():
        pub = np.round(scores(truth, truth, V5), 3)
        fields, gaps = solve(pub, rng, seeds=1)
        val = lambda x: scores(np.array(x)[None, :], truth, V5)[0]
        print(f"{name:16s} {np.mean(gaps):6.3f} {val(best_response(truth, V5)):9.3f} "
              f"{val(consensus(fields)):9.3f} {val(naive):7.3f} {val(old):7.3f}", flush=True)


def main():
    validate()
    if "--validate" in sys.argv:
        return
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_scores.json"))["g1"]])
    rng = np.random.default_rng(2026)
    fields, gaps = solve(pub, rng, seeds=3)
    print(f"\nReal field: {len(pub)} scores, {len(fields)} reconstructions")
    print("start order: riddler, archetypes, equilibrium, half+half (3 of each)")
    for f, g in zip(fields, gaps):
        br = best_response(f, V5)
        print(f"gap {g:.3f} mean {f.mean(0).round(1)} own-BR {tuple(int(v) for v in br)} "
              f"{scores(br[None, :], f, V5)[0]:.3f}", flush=True)
    pick = consensus(fields)
    vals = [scores(pick[None, :], f, V5)[0] for f in fields]
    print("consensus", tuple(int(v) for v in pick), "scores", np.round(vals, 2))
    for name, c in (("old pick", (6, 5, 3, 42, 44)), ("(6,16,1,36,41)", (6, 16, 1, 36, 41))):
        print(name, np.round([scores(np.array(c)[None, :], f, V5)[0] for f in fields], 2))
    np.savez("results/g1_fields.npz", *fields)


if __name__ == "__main__":
    main()
