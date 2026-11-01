"""How picks hold up as a share of the field concentrates on one or two big castles.

The base archetypes in field.py contain almost no such entries, which flatters any
concentrated pick; this measures how much of the result depends on that gap.
"""
import subprocess
import tempfile

import numpy as np

from field import scenario, jitter
from game1_robust import score_all
from game1_solve import best_response

EXE = "./game2"
CONCENTRATED = np.array([
    (0, 0, 0, 40, 60), (0, 0, 0, 45, 55), (0, 0, 0, 50, 50), (10, 0, 0, 38, 52),
    (0, 0, 40, 0, 60), (0, 0, 35, 0, 65), (0, 0, 0, 30, 70), (2, 3, 5, 40, 50),
])
SHARES = [0, 0.05, 0.1, 0.2, 0.35, 0.5]
G1 = [(3, 16, 1, 37, 43), (7, 13, 20, 27, 33), (3, 8, 22, 30, 37), (2, 3, 25, 35, 35)]
G2 = [(11, 0, 0, 37, 52), (0, 0, 0, 40, 60), (0, 0, 39, 0, 61), (3, 16, 1, 37, 43),
      (7, 13, 20, 27, 33), (3, 8, 22, 30, 37)]


def mixed(rng, share):
    base = scenario(rng)
    n = int(len(base) * share / (1 - share))
    picks = CONCENTRATED[rng.integers(len(CONCENTRATED), size=n)]
    extra = [jitter(p, rng.poisson(3), rng) for p in picks]
    return np.vstack([base, *extra]) if extra else base


def run(args):
    return subprocess.run([EXE, *map(str, args)], capture_output=True, text=True,
                          check=True).stdout


def main(per_share=30):
    rng = np.random.default_rng(77)
    print("Percentile in field, mean over scenarios (br = best response, its own field)")
    print("share  game | " + " | ".join(map(str, G1)) + " | br")
    for share in SHARES:
        fields = [mixed(rng, share) for _ in range(per_share)]
        x = np.array(G1)
        p = np.mean([(score_all(f, f)[:, None] < score_all(x, f)[None, :]).mean(0)
                     for f in fields], axis=0)
        _, br = best_response(fields[0], np.full(len(fields[0]), 1 / len(fields[0])))
        print(f"{share:4.2f}  G1 ", np.round(p, 3), tuple(int(v) for v in br))
    print("share  game | " + " | ".join(map(str, G2)) + " | br")
    with tempfile.TemporaryDirectory() as d:
        fp, cp = f"{d}/f", f"{d}/c"
        for share in SHARES:
            pct, sc = [], []
            for i in range(per_share):
                f = mixed(rng, share)
                np.savetxt(fp, f, fmt="%d")
                np.savetxt(cp, np.vstack([np.array(G2), f[rng.integers(len(f), size=300)]]),
                           fmt="%d")
                out = run(["eval", fp, cp, 20000, i + 1])
                s = np.array([float(l.split()[0]) for l in out.splitlines()])
                sc.append(s[:len(G2)])
                pct.append((s[len(G2):, None] < s[:len(G2)][None, :]).mean(0))
            np.savetxt(fp, f, fmt="%d")
            br = run(["br", fp, 4000, 24, 1, 9]).split()
            print(f"{share:4.2f}  G2  pct", np.round(np.mean(pct, 0), 3),
                  "score", np.round(np.mean(sc, 0), 2), "br", br[1:], br[0])


if __name__ == "__main__":
    main()
