"""Search every allocation against a broad set of contestant regimes.

    python3 search.py prep       build the regime fields and their fit to the leaderboard
    ./game2 screen ...           (printed by prep) score all 4.6M allocations in Game 2
    python3 search.py refine     re-test finalists on fresh fields; compare place odds
    python3 search.py game1      the same search for Game 1, by exact calculation

Regimes include fields that counter the search's own favourites, so the winner has to
survive more than a field that happens to suit it. Fit weights come from how closely a
regime reproduces the published score curve. That is a weak signal, so equal weighting
and the worst regime are reported beside it.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from field import jitter, random_allocs, scenario
from riddler import best_response, scores
from stress import CONCENTRATED
from stress2 import EXE, REAL, V5, sharp_g1, sharp_g2

SIZE = 3000
OUT = Path("results/search")
NEW2, OLD2 = (13, 37, 48, 0, 2), (6, 31, 0, 63, 0)
NEW1, OLD1 = (6, 16, 1, 36, 41), (6, 5, 3, 42, 44)
Q = (np.arange(100) + 0.5) / 100


def real(rng, n):
    return REAL[rng.integers(len(REAL), size=n)]


def near(rng, base, n, moves=3):
    return np.vstack([jitter(np.array(base), rng.poisson(moves), rng)[None, :] for _ in range(n)])


def heavy(rng, n):
    return np.vstack([near(rng, CONCENTRATED[rng.integers(len(CONCENTRATED))], 1) for _ in range(n)])


def low_castles(rng, n):
    """Contrarians who skip castles 4 and 5, like the Game 2 pick does."""
    bases = [(20, 33, 47, 0, 0), (13, 37, 48, 0, 2), (10, 40, 50, 0, 0), (0, 45, 55, 0, 0)]
    return np.vstack([near(rng, bases[rng.integers(len(bases))], 1) for _ in range(n)])


def blend(*parts):
    """parts are (share, rows-function); the remaining share is real entries."""
    def make(rng):
        used = sum(int(SIZE * s) for s, _ in parts)
        return np.vstack([f(rng, int(SIZE * s)) for s, f in parts] + [real(rng, SIZE - used)])
    return make


def copies(base):
    return lambda rng, n: near(rng, base, n, 2)


def regimes(sharp, new, old, br):
    """Name -> field maker. `br` best-responds to a field, for the levelled regimes."""
    fit = blend((0.2, sharp))
    def level(rng, base):
        return np.vstack([base[:int(SIZE * 0.7)], near(rng, br(base), SIZE - int(SIZE * 0.7), 2)])
    def level2(rng):
        return level(rng, fit(rng))
    def level3(rng):
        return level(rng, level2(rng))
    return {
        "Riddler": lambda rng: real(rng, SIZE),
        "Focal": lambda rng: scenario(rng, SIZE),
        "Heavy": lambda rng: heavy(rng, SIZE),
        "Equilibrium": lambda rng: sharp(rng, SIZE),
        "Random": lambda rng: random_allocs(SIZE, rng),
        "20% sharp": fit,
        "40% sharp": blend((0.4, sharp)),
        "60% sharp": blend((0.6, sharp)),
        "20% heavy": blend((0.2, heavy)),
        "40% heavy": blend((0.4, heavy)),
        "20% low-castle": blend((0.2, low_castles)),
        "40% low-castle": blend((0.4, low_castles)),
        "20% copy new": blend((0.2, sharp), (0.2, copies(new))),
        "20% copy old": blend((0.2, sharp), (0.2, copies(old))),
        "Level 2": level2,
        "Level 3": level3,
    }


def g2_br(field):
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(f"{d}/f", field, fmt="%d")
        out = subprocess.run([EXE, "br", f"{d}/f", "4000", "24", "1", "7"], capture_output=True,
                             text=True, check=True).stdout
    return np.array([int(v) for v in out.split()[1:]])


def g1_br(field):
    return best_response(field, V5)


def g2_regimes():
    return regimes(sharp_g2, NEW2, OLD2, g2_br)


def g1_regimes():
    return regimes(sharp_g1, NEW1, OLD1, g1_br)


def run(mode, field, x, tables, seed, d):
    np.savetxt(f"{d}/f", field, fmt="%d")
    np.savetxt(f"{d}/c", x, fmt="%d")
    out = subprocess.run([EXE, mode, f"{d}/f", f"{d}/c", str(tables), str(seed)], capture_output=True,
                         text=True, check=True).stdout
    return out.splitlines()


def weights_from_fit(errors, scale):
    w = np.exp(-np.array(errors) / scale)
    return w / w.sum()


def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_scores.json"))["g2"]])
    rng = np.random.default_rng(21)
    errors, paths = {}, []
    with tempfile.TemporaryDirectory() as d:
        for i, (name, make) in enumerate(g2_regimes().items()):
            f = make(rng)
            path = OUT / f"regime{i}.txt"
            np.savetxt(path, f, fmt="%d")
            paths.append(str(path))
            sample = f[rng.integers(len(f), size=1200)]
            own = np.array([float(l.split()[0]) for l in run("eval", f, sample, 30000, 5, d)])
            errors[name] = float(np.abs(np.quantile(own, Q) - np.quantile(pub, Q)).mean())
            print(f"{name:16s} curve error {errors[name]:.3f}", flush=True)
    json.dump(errors, open(OUT / "errors.json", "w"))
    print(f"\nnext: ./game2 screen 2000 {OUT}/all {' '.join(paths)}")


def objectives(s, errors):
    fit = weights_from_fit(list(errors.values()), 0.15)
    return {"fit-weighted": s @ fit, "equal": s.mean(1), "worst regime": s.min(1)}, fit


OUT_SCORES = OUT / "all.scores"


def refine(errors=None):
    errors = errors or json.load(open(OUT / "errors.json"))
    names = list(errors)
    allocs = np.fromfile(OUT / "all.alloc", dtype=np.int8).reshape(-1, 5)
    s = np.fromfile(OUT_SCORES, dtype=np.float32).reshape(len(names), -1).T
    print(f"{len(allocs):,} allocations on {len(names)} regimes")
    anchors = [(28, 0, 0, 0, 72), NEW2, OLD2]
    obj, fit = objectives(s, errors)
    print("regime weights:", ", ".join(f"{n} {w:.2f}" for n, w in zip(names, fit)))
    keep = set()
    for v in obj.values():
        keep |= set(np.argsort(-v)[:2500].tolist())
    keep = sorted(keep)
    x = np.vstack([allocs[keep].astype(np.int64), np.array(anchors)])
    del s

    rng = np.random.default_rng(909)
    makers = g2_regimes()
    table = []
    with tempfile.TemporaryDirectory() as d:
        for i, name in enumerate(names):
            f = makers[name](rng)
            table.append([float(l.split()[0]) for l in run("eval", f, x, 40000, 300 + i, d)])
        t = np.array(table).T
        obj, _ = objectives(t, errors)
        finalists = {}
        for label, v in obj.items():
            for i in np.argsort(-v)[:3]:
                finalists[tuple(int(a) for a in x[i])] = label
        finalists[(28, 0, 0, 0, 72)] = "round 1 winner"
        finalists[NEW2] = "current pick"
        finalists[OLD2] = "old pick"
        fx = np.array(list(finalists))
        # Fresh fields again, so these numbers played no part in choosing the finalists.
        results = {}
        for i, name in enumerate(names):
            f = makers[name](rng)
            lines = run("places", f, fx, 200000, 700 + i, d)
            results[name] = [[float(v) for v in l.split()[:5]] for l in lines]
    r = np.array([results[n] for n in names])  # regime, finalist, score + p1-p4
    obj, _ = objectives(r[:, :, 0].T, errors)
    print("\nallocation           found by         fit-wtd  equal  worst  worst regime")
    for i in np.argsort(-obj["fit-weighted"]):
        print(f"{str(tuple(int(v) for v in fx[i])):20s} {finalists[tuple(int(v) for v in fx[i])]:15s} "
              f"{obj['fit-weighted'][i]:7.2f} {obj['equal'][i]:6.2f} {obj['worst regime'][i]:6.2f}  "
              f"{names[int(np.argmin(r[:, i, 0]))]}")
    json.dump({"names": names, "finalists": fx.tolist(), "labels": [finalists[tuple(int(v) for v in a)] for a in fx],
               "results": r.tolist(), "errors": errors}, open(OUT / "g2_final.json", "w"))


def game1():
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_scores.json"))["g1"]])
    allocs = np.fromfile(OUT / "all.alloc", dtype=np.int8).reshape(-1, 5).astype(np.int64)
    rng = np.random.default_rng(33)
    makers = g1_regimes()
    names, errors, table, picks = list(makers), {}, [], [NEW1, OLD1]
    for name in names:
        f = makers[name](rng)
        own = scores(f, f, V5)
        errors[name] = float(np.abs(np.quantile(own, Q) - np.quantile(pub, Q)).mean())
        table.append(scores(allocs, f, V5).astype(np.float32))
    s = np.array(table).T
    obj, fit = objectives(s, errors)
    print("regime fit to the real curve:", ", ".join(f"{n} {e:.2f}" for n, e in errors.items()))
    found = {}
    for label, v in obj.items():
        for i in np.argsort(-v)[:3]:
            found[tuple(int(a) for a in allocs[i])] = label
    found[NEW1] = "current pick"
    found[OLD1] = "old pick"
    x = np.array(list(found))
    fresh = np.array([scores(x, makers[n](rng), V5) for n in names]).T
    o, _ = objectives(fresh, errors)
    print("\nallocation           found by         fit-wtd  equal  worst  worst regime")
    for i in np.argsort(-o["fit-weighted"]):
        print(f"{str(tuple(int(v) for v in x[i])):20s} {found[tuple(int(v) for v in x[i])]:15s} "
              f"{o['fit-weighted'][i]:7.2f} {o['equal'][i]:6.2f} {o['worst regime'][i]:6.2f}  "
              f"{names[int(np.argmin(fresh[i]))]}")
    json.dump({"names": names, "finalists": x.tolist(), "labels": [found[tuple(int(v) for v in a)] for a in x],
               "scores": fresh.tolist(), "errors": errors}, open(OUT / "g1_final.json", "w"))


if __name__ == "__main__":
    {"prep": prep, "refine": refine, "game1": game1}[sys.argv[1]]()
