"""Finalists against each threat separately, so one weakness can't hide in an average.

Sharp entries come from the equilibrium solutions: the field a strong applicant base
would approach. Copycat rows add jittered copies of the candidate itself, the case of
other applicants reaching the same answer from the same public data.
"""
import json
import os
import subprocess
import tempfile

import numpy as np

from field import jitter, scenario
from riddler import scores
from stress import CONCENTRATED

EXE = "./game2"
V5 = np.arange(1, 6)
ROUNDS = {int(k[1:]): v for k, v in np.load("results/riddler_merged.npz").items()}
REAL = np.vstack(list(ROUNDS.values()))

G1 = [(6, 5, 3, 42, 44), (6, 16, 1, 36, 41), (3, 16, 1, 37, 43), (7, 13, 20, 27, 33)]
G2 = [(6, 31, 0, 63, 0), (10, 30, 0, 60, 0), (0, 0, 39, 0, 61), (0, 0, 0, 40, 60)]


def sharp_g1(rng, n):
    ne = json.load(open("results/game1_ne.json"))
    s, w = np.array(ne["strategies"]), np.array(ne["weights"])
    return s[rng.choice(len(s), n, p=w / w.sum())]


def sharp_g2(rng, n):
    a = np.loadtxt("results/game2_archive.txt", dtype=np.int64)
    return a[len(a) // 2:][rng.integers(len(a) - len(a) // 2, size=n)]


def mix(rng, parts):
    """parts: list of (share, rows-maker); the remainder is real entries."""
    size = 2000
    out, used = [], 0
    for share, make in parts:
        n = int(size * share)
        used += n
        out.append(make(n))
    out.append(REAL[rng.integers(len(REAL), size=size - used)])
    return np.vstack(out)


def copies(rng, x):
    return lambda n: np.vstack([jitter(np.array(x), rng.poisson(2), rng)[None, :] for _ in range(n)])


def conc(rng):
    return lambda n: np.vstack([jitter(CONCENTRATED[rng.integers(len(CONCENTRATED))],
                                       rng.poisson(3), rng)[None, :] for _ in range(n)])


def threats(rng, sharp):
    t = {f"round {n} only": (lambda c, n=n: ROUNDS[n]) for n in ROUNDS}
    t["all real"] = lambda c: REAL
    t["+30% focal"] = lambda c: mix(rng, [(0.3, lambda n: scenario(rng, n))])
    t["+30% concentrated"] = lambda c: mix(rng, [(0.3, conc(rng))])
    t["+30% sharp"] = lambda c: mix(rng, [(0.3, lambda n: sharp(rng, n))])
    t["+60% sharp"] = lambda c: mix(rng, [(0.6, lambda n: sharp(rng, n))])
    t["+20% copies of self"] = lambda c: mix(rng, [(0.2, copies(rng, c))])
    t["+20% copy,+30% sharp"] = lambda c: mix(rng, [(0.2, copies(rng, c)),
                                                    (0.3, lambda n: sharp(rng, n))])
    return t


def g2_eval(f, x, rng, d, env=None):
    fp, cp = f"{d}/f", f"{d}/c"
    np.savetxt(fp, f, fmt="%d")
    np.savetxt(cp, np.vstack([np.array(x)[None, :], f[rng.integers(len(f), size=400)]]), fmt="%d")
    out = subprocess.run([EXE, "eval", fp, cp, "40000", "3"], capture_output=True, text=True,
                         check=True, env={**os.environ, **(env or {})}).stdout
    v = [float(line.split()[0]) for line in out.splitlines()]
    return v[0], float((np.array(v[1:]) < v[0]).mean())


def report(game, cands):
    rng = np.random.default_rng(99)
    t = threats(rng, sharp_g1 if game == 1 else sharp_g2)
    print(f"\nGAME {game}: percentile in field (score)")
    print(" " * 22 + "".join(f"{str(c).replace(' ', ''):>19s}" for c in cands))
    with tempfile.TemporaryDirectory() as d:
        for name, make in t.items():
            cells = []
            for c in cands:
                f = make(c)
                if game == 1:
                    x = np.array(c)[None, :]
                    s = scores(x, f, V5)[0]
                    p = float((scores(f, f, V5) < s).mean())
                else:
                    s, p = g2_eval(f, c, rng, d)
                cells.append(f"{p:6.3f} ({s:5.2f})")
            print(f"{name:22s}" + "".join(f"{cell:>19s}" for cell in cells), flush=True)
        if game == 2:
            cells = []
            for c in cands:
                f = t["+30% sharp"](c)
                s, p = g2_eval(f, c, rng, d, {"ZERO_UNAWARDED": "1"})
                cells.append(f"{p:6.3f} ({s:5.2f})")
            print(f"{'zero castles unaward':22s}" + "".join(f"{cell:>19s}" for cell in cells))


if __name__ == "__main__":
    report(1, G1)
    report(2, G2)
