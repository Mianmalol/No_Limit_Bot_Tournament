"""Reconstruct the Game 1 field from published scores alone.

FAILED its synthetic check: picks from this field lose to the plain Riddler field.
Kept as a record; don't use its output.

A Game 1 score is a sum of five lookups into the field's castle marginals, so a guess
at the marginals scores all 4.6M allocations at once. Each published score then gets
an allocation that earns it under that guess, drawn in proportion to a prior; those
allocations give new marginals, and the loop repeats until the field reproduces its
own scores. Equal scores are taken to be one allocation entered several times.

Run directly: validates on synthetic fields with known answers, then solves the real one.
"""
import json
import sys
from collections import Counter

import numpy as np
from numba import njit

from field import scenario
from riddler import best_response, scores

V5 = np.arange(1, 6)
ROUNDS = {int(k[1:]): v for k, v in np.load("results/riddler_merged.npz").items()}


def enumerate_allocs():
    out = np.zeros((4598126, 5), dtype=np.int8)
    i = 0
    for a in range(101):
        for b in range(101 - a):
            for c in range(101 - a - b):
                for d in range(101 - a - b - c):
                    out[i] = (a, b, c, d, 100 - a - b - c - d)
                    i += 1
    return out


def make_prior(allocs):
    """Log weight of each allocation: smoothed real castle marginals plus a round-number bonus."""
    real = np.vstack(list(ROUNDS.values()))
    log_w = np.zeros(len(allocs))
    for c in range(5):
        h = np.bincount(real[:, c], minlength=101)[:101].astype(float)
        h = np.convolve(h, np.ones(9) / 9, mode="same") + 0.5
        log_w += np.log(h / h.sum())[allocs[:, c]]
    # Real entries are all-multiples-of-5 about 40x as often as chance.
    log_w += np.log(40.0) * np.all(allocs % 5 == 0, axis=1)
    return log_w


def marginals(field):
    """Score lookup (5, 101): points a bid of s earns at each castle."""
    table = np.zeros((5, 101))
    for c in range(5):
        pmf = np.bincount(field[:, c], minlength=101)[:101] / len(field)
        below = np.concatenate([[0.0], np.cumsum(pmf)[:-1]])
        table[c] = (c + 1) * (below + 0.5 * pmf)
    return table


@njit(cache=True, parallel=False)
def score_table(allocs, table):
    out = np.zeros(allocs.shape[0])
    for i in range(allocs.shape[0]):
        s = 0.0
        for c in range(5):
            s += table[c, allocs[i, c]]
        out[i] = s
    return out


def assign(allocs, s, log_w, targets, counts, rng, tol):
    """For each distinct published score, draw an allocation earning it, by prior weight."""
    order = np.argsort(s)
    sorted_s, sorted_w = s[order], np.exp(log_w[order] - log_w.max())
    cum = np.concatenate([[0.0], np.cumsum(sorted_w)])
    lo = np.searchsorted(sorted_s, targets - tol)
    hi = np.searchsorted(sorted_s, targets + tol)
    picks = []
    for t, k, a, b in zip(targets, counts, lo, hi):
        if b <= a:  # nothing in the window: use the nearest allocation
            a = min(np.searchsorted(sorted_s, t), len(s) - 1)
            b = a + 1
        u = cum[a] + rng.random() * (cum[b] - cum[a])
        i = min(np.searchsorted(cum, u, side="right") - 1, len(s) - 1)
        picks.append(np.repeat(allocs[order[i]][None, :], k, axis=0))
    return np.vstack(picks).astype(np.int64)


def invert(published, allocs, log_w, rng, iters=40, start=None, verbose=False):
    values = Counter(np.round(published, 3).tolist())
    targets = np.array(sorted(values))
    counts = np.array([values[t] for t in targets])
    table = marginals(start if start is not None else np.vstack(list(ROUNDS.values())))
    for it in range(iters):
        s = score_table(allocs, table)
        field = assign(allocs, s, log_w, targets, counts, rng, 0.0015)
        new = marginals(field)
        rate = max(0.15, 0.6 * 0.95 ** it)
        table = (1 - rate) * table + rate * new
        if verbose and it % 10 == 9:
            print(f"  iter {it + 1} fit error {fit_error(field, published):.4f}", flush=True)
    return field


def fit_error(field, published):
    """Mean gap between sorted self-scores of a field and the published scores."""
    own = np.sort(scores(field, field, V5))
    return float(np.abs(own - np.sort(published)).mean())


def compare(true, found):
    a, b = marginals(true), marginals(found)
    return float(np.abs(a - b).sum(1).mean() / 101)


def validate(allocs, log_w):
    rng = np.random.default_rng(5)
    print("Synthetic check (truth unknown to the method)")
    print("field                 fit err  marginal err  true-BR  found-BR  naive-BR")
    for name, truth in (("archetypes", scenario(rng, 3478)),
                        ("archetype sharp", np.vstack([scenario(rng, 2000),
                                                       np.tile([6, 5, 3, 42, 44], (300, 1)),
                                                       np.tile([1, 16, 1, 41, 41], (300, 1)),
                                                       scenario(rng, 878)]))):
        pub = np.round(scores(truth, truth, V5), 3)
        found = invert(pub, allocs, log_w, rng)
        best_true = scores(best_response(truth, V5)[None, :], truth, V5)[0]
        best_found = scores(best_response(found, V5)[None, :], truth, V5)[0]
        naive = np.vstack(list(ROUNDS.values()))
        best_naive = scores(best_response(naive, V5)[None, :], truth, V5)[0]
        print(f"{name:20s} {fit_error(found, pub):7.3f} {compare(truth, found):10.3f}"
              f" {best_true:9.3f} {best_found:8.3f} {best_naive:8.3f}", flush=True)


def main():
    allocs = enumerate_allocs()
    log_w = make_prior(allocs)
    validate(allocs, log_w)
    if "--validate" in sys.argv:
        return
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_scores.json"))["g1"]])
    print(f"\nReal field: {len(pub)} scores")
    runs = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        found = invert(pub, allocs, log_w, rng, iters=60)
        runs.append(found)
        br = best_response(found, V5)
        common = Counter(map(tuple, found.tolist())).most_common(4)
        print(f"seed {seed} fit {fit_error(found, pub):.3f} mean {found.mean(0).round(1)} "
              f"BR {tuple(int(v) for v in br)} scores {scores(br[None, :], found, V5)[0]:.3f}",
              flush=True)
        print("   clusters", [(tuple(int(v) for v in a), n) for a, n in common], flush=True)
    np.savez("results/g1_fields.npz", *runs)


if __name__ == "__main__":
    main()
