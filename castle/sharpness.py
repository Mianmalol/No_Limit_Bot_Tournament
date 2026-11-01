"""One-parameter reading of the leaderboard: what share of near-equilibrium players
makes real entries reproduce the published score curves, and how do picks fare there?
"""
import json
import subprocess
import tempfile

import numpy as np

from riddler import scores
from stress2 import EXE, REAL, V5, sharp_g1, sharp_g2

SHARES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q = (np.arange(100) + 0.5) / 100
G1 = [(6, 5, 3, 42, 44), (6, 16, 1, 36, 41), (3, 16, 1, 37, 43), (3, 6, 3, 36, 52)]
G2 = [(6, 31, 0, 63, 0), (10, 30, 0, 60, 0), (0, 0, 39, 0, 61), (13, 37, 48, 0, 2)]
SIZE = 3000


def field(rng, share, sharp):
    n = int(SIZE * share)
    return np.vstack([sharp(rng, n), REAL[rng.integers(len(REAL), size=SIZE - n)]])


def g2_scores(f, x, d):
    np.savetxt(f"{d}/f", f, fmt="%d")
    np.savetxt(f"{d}/c", x, fmt="%d")
    out = subprocess.run([EXE, "eval", f"{d}/f", f"{d}/c", "30000", "5"], capture_output=True,
                         text=True, check=True).stdout
    return np.array([float(line.split()[0]) for line in out.splitlines()])


def main():
    pub = json.load(open("../data/leaderboard_scores.json"))
    p1 = np.array([r[1] for r in pub["g1"]])
    p2 = np.array([r[1] for r in pub["g2"]])
    rng = np.random.default_rng(3)
    print("GAME 1  published top %.3f p90 %.3f p50 %.3f" % (p1.max(), *np.quantile(p1, [.9, .5])))
    print("share  curve-err  top   p90   p50 | " + " ".join(str(c).replace(" ", "") for c in G1))
    for s in SHARES:
        f = field(rng, s, sharp_g1)
        own = scores(f, f, V5)
        err = np.abs(np.quantile(own, Q) - np.quantile(p1, Q)).mean()
        v = scores(np.array(G1), f, V5)
        cells = " ".join(f"{a:.2f}/{(p1 < a).mean():.3f}" for a in v)
        print(f"{s:4.1f} {err:8.3f} {own.max():6.2f} {np.quantile(own, .9):5.2f} "
              f"{np.quantile(own, .5):5.2f} | {cells}", flush=True)
    print("\nGAME 2  published top %.3f p90 %.3f p50 %.3f" % (p2.max(), *np.quantile(p2, [.9, .5])))
    print("share  curve-err  top   p90   p50 | " + " ".join(str(c).replace(" ", "") for c in G2))
    with tempfile.TemporaryDirectory() as d:
        for s in SHARES:
            f = field(rng, s, sharp_g2)
            sample = f[rng.integers(len(f), size=1200)]
            v = g2_scores(f, np.vstack([np.array(G2), sample]), d)
            pick, own = v[:len(G2)], v[len(G2):]
            err = np.abs(np.quantile(own, Q) - np.quantile(p2, Q)).mean()
            cells = " ".join(f"{a:.2f}/{(p2 < a).mean():.3f}" for a in pick)
            print(f"{s:4.1f} {err:8.3f} {own.max():6.2f} {np.quantile(own, .9):5.2f} "
                  f"{np.quantile(own, .5):5.2f} | {cells}", flush=True)
    for g, p in (("g1", p1), ("g2", p2)):
        hist = np.histogram(p, bins=np.arange(0, 10, 0.5))[0]
        print(g, "histogram from 0 by 0.5:", hist.tolist())


if __name__ == "__main__":
    main()
