"""Second search round: the attacks that broke round one's winners become regimes.

    python3 round2.py prep         build the extra Game 2 regimes
    ./game2 screen ...             (printed by prep) scores to results/search/round2
    python3 round2.py refine       re-test on all 21 regimes
    python3 round2.py game1        same for Game 1
"""
import json
import sys
import tempfile

import numpy as np

import search
from attack import WINNER, arms_race, countered
from search import OUT, Q, blend, copies, near, run, sharp_g1, sharp_g2

G1_LEADER = (3, 5, 5, 42, 45)
base_g1, base_g2 = search.g1_regimes, search.g2_regimes


def g1_race(rng, n):
    bases = [(1, 2, 3, 47, 47), (2, 2, 2, 44, 50), (0, 4, 4, 46, 46), (3, 3, 3, 40, 51)]
    return np.vstack([near(rng, bases[rng.integers(len(bases))], 1) for _ in range(n)])


def extra_g2():
    return {
        "10% copy 28-72": blend((0.2, sharp_g2), (0.1, copies(WINNER))),
        "20% copy 28-72": blend((0.2, sharp_g2), (0.2, copies(WINNER))),
        "10% castle-5 race": blend((0.2, sharp_g2), (0.1, arms_race)),
        "25% castle-5 race": blend((0.2, sharp_g2), (0.25, arms_race)),
        "Copies + counters": lambda rng: countered(rng, 0.2)[0],
    }


def extra_g1():
    return {
        "20% copy leader": blend((0.2, sharp_g1), (0.2, copies(G1_LEADER))),
        "10% castle 4-5 race": blend((0.2, sharp_g1), (0.1, g1_race)),
        "25% castle 4-5 race": blend((0.2, sharp_g1), (0.25, g1_race)),
    }


search.g2_regimes = lambda: {**base_g2(), **extra_g2()}
search.g1_regimes = lambda: {**base_g1(), **extra_g1()}


def prep():
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_scores.json"))["g2"]])
    errors = json.load(open(OUT / "errors.json"))
    rng = np.random.default_rng(22)
    paths = []
    with tempfile.TemporaryDirectory() as d:
        for i, (name, make) in enumerate(extra_g2().items()):
            f = make(rng)
            path = OUT / f"extra{i}.txt"
            np.savetxt(path, f, fmt="%d")
            paths.append(str(path))
            sample = f[rng.integers(len(f), size=1200)]
            own = np.array([float(l.split()[0]) for l in run("eval", f, sample, 30000, 5, d)])
            errors[name] = float(np.abs(np.quantile(own, Q) - np.quantile(pub, Q)).mean())
            print(f"{name:20s} curve error {errors[name]:.3f}", flush=True)
    json.dump(errors, open(OUT / "errors2.json", "w"))
    print(f"\nnext: ./game2 screen 2000 {OUT}/round2 {' '.join(paths)}")


def refine():
    errors = json.load(open(OUT / "errors2.json"))
    n = 4598126
    first = np.fromfile(OUT / "all.scores", dtype=np.float32).reshape(-1, n)
    second = np.fromfile(OUT / "round2.scores", dtype=np.float32).reshape(-1, n)
    np.vstack([first, second]).tofile(OUT / "all.scores.combined")
    del first, second
    search.OUT_SCORES = OUT / "all.scores.combined"
    search.refine(errors)


def check():
    """Final attack: the Game 2 winner as the crowd, plus players countering that crowd."""
    pick = (0, 39, 0, 61, 0)
    rng = np.random.default_rng(55)
    fx = np.array([pick, search.NEW2, search.OLD2, WINNER])
    tests = {f"20% sharp + {s:.0%} copies": blend((0.2, sharp_g2), (s, copies(pick)))(rng)
             for s in (0.1, 0.2, 0.4)}
    tests["20% sharp baseline"] = blend((0.2, sharp_g2))(rng)
    for s in (0.1, 0.2):
        popular = blend((0.2, sharp_g2), (s, copies(pick)))(rng)
        top = search.g2_br(popular)
        print(f"counter to {s:.0%} copies:", top.tolist())
        n = int(search.SIZE * 0.15)
        tests[f"{s:.0%} copies + counters"] = np.vstack([popular[:search.SIZE - n], near(rng, top, n, 2)])
    print(f"{'':28s}" + "".join(str(tuple(int(v) for v in x)).replace(" ", "").rjust(26) for x in fx))
    with tempfile.TemporaryDirectory() as d:
        for i, (name, f) in enumerate(tests.items()):
            rows = [[float(v) for v in l.split()[:5]] for l in run("places", f, fx, 200000, 60 + i, d)]
            print(f"{name:28s}" + "".join(f"{r[0]:5.2f} [1st {r[1]:.0%} last {r[4]:.0%}]".rjust(26) for r in rows),
                  flush=True)


if __name__ == "__main__":
    {"prep": prep, "refine": refine, "game1": search.game1, "check": check}[sys.argv[1]]()
