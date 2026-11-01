"""Scenario generator for the applicant field.

There is no data on real entries: the archetypes and weights below are judgement
calls. Scenarios vary them widely so conclusions don't rest on any one guess.
"""
import numpy as np

ARCHETYPES = [
    ("proportional", (7, 13, 20, 27, 33), 16),
    ("round_proportional", (10, 15, 20, 25, 30), 10),
    ("round_steep", (5, 10, 20, 30, 35), 8),
    ("round_mid", (5, 15, 20, 25, 35), 4),
    ("round_flat", (10, 10, 20, 30, 30), 5),
    ("uniform", (20, 20, 20, 20, 20), 3),
    ("top_three", (0, 0, 33, 33, 34), 8),
    ("top_three_steep", (0, 0, 25, 35, 40), 5),
    ("skip_one", (0, 10, 20, 30, 40), 8),
    ("skip_one_steep", (0, 5, 15, 30, 50), 5),
    ("token_low", (1, 1, 32, 33, 33), 5),
    ("token_low_steep", (2, 3, 25, 35, 35), 5),
    ("undercut", (3, 8, 22, 30, 37), 10),
    ("undercut_flat", (2, 12, 22, 28, 36), 4),
    ("cheap_castles", (25, 25, 25, 25, 0), 2),
    ("no_top", (0, 30, 35, 35, 0), 2),
]
BASES = np.array([a[1] for a in ARCHETYPES], dtype=np.int64)
WEIGHTS = np.array([a[2] for a in ARCHETYPES], dtype=np.float64)


def jitter(alloc, moves, rng):
    """Move one soldier between random castles, `moves` times."""
    out = alloc.copy()
    for _ in range(moves):
        src = rng.choice(np.flatnonzero(out > 0))
        out[src] -= 1
        out[rng.integers(5)] += 1
    return out


def random_allocs(n, rng):
    raw = rng.dirichlet(np.ones(5), n) * 100
    out = np.floor(raw).astype(np.int64)
    for row, frac in zip(out, raw - out):
        row[np.argsort(-frac)[: 100 - row.sum()]] += 1
    return out


def scenario(rng, size=1500):
    """One plausible field: reweighted archetypes, noise and a random-entry share."""
    probs = rng.dirichlet(WEIGHTS / WEIGHTS.sum() * rng.choice([3, 8, 20]))
    noise = rng.choice([0.5, 2, 5, 10])
    random_share = rng.uniform(0, 0.15)
    n_random = int(size * random_share)
    picks = rng.choice(len(BASES), size - n_random, p=probs)
    rows = [jitter(BASES[p], rng.poisson(noise), rng) for p in picks]
    return np.vstack(rows + [random_allocs(n_random, rng)])
