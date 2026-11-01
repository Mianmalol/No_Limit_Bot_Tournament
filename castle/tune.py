"""Fine-tune around the leading families by mean raw score across every threat.

Copies are of the family centres, so an entry one soldier past them shows up as an
edge. Allocations are chosen on one set of threat fields and reported on a second.
"""
import subprocess
import tempfile

import numpy as np

from riddler import scores
from stress2 import EXE, REAL, ROUNDS, V5, conc, copies, mix, sharp_g1, sharp_g2
from field import scenario

CENTRES = {1: [(6, 16, 1, 36, 41), (7, 5, 3, 42, 43)], 2: [(10, 30, 0, 60, 0)]}
RADIUS = {1: 14, 2: 8}


def neighbours(centres, radius):
    out = set()
    for c in centres:
        c = np.array(c)
        for a in range(max(c[0] - radius, 0), c[0] + radius + 1):
            for b in range(max(c[1] - radius, 0), c[1] + radius + 1):
                for d in range(max(c[2] - radius, 0), c[2] + radius + 1):
                    for e in range(max(c[3] - radius, 0), c[3] + radius + 1):
                        last = 100 - a - b - d - e
                        if last >= 0 and abs(a - c[0]) + abs(b - c[1]) + abs(d - c[2]) \
                                + abs(e - c[3]) + abs(last - c[4]) <= radius:
                            out.add((a, b, d, e, last))
    return np.array(sorted(out))


def fields(rng, game):
    sharp = sharp_g1 if game == 1 else sharp_g2
    fam = lambda n: np.vstack([copies(rng, c)(n // len(CENTRES[game]) + 1) for c in CENTRES[game]])[:n]
    out = [ROUNDS[n] for n in ROUNDS]
    out.append(mix(rng, [(0.3, lambda n: scenario(rng, n))]))
    out.append(mix(rng, [(0.2, conc(rng))]))
    out.append(mix(rng, [(0.3, lambda n: sharp(rng, n))]))
    out.append(mix(rng, [(0.6, lambda n: sharp(rng, n))]))
    out.append(mix(rng, [(0.2, fam)]))
    out.append(mix(rng, [(0.2, fam), (0.3, lambda n: sharp(rng, n))]))
    return out


def table(game, x, group, d):
    rows = []
    for i, f in enumerate(group):
        if game == 1:
            rows.append(scores(x, f, V5))
        else:
            np.savetxt(f"{d}/f", f, fmt="%d")
            np.savetxt(f"{d}/c", x, fmt="%d")
            out = subprocess.run([EXE, "eval", f"{d}/f", f"{d}/c", "30000", str(i + 1)],
                                 capture_output=True, text=True, check=True).stdout
            rows.append(np.array([float(line.split()[0]) for line in out.splitlines()]))
    return np.array(rows)


def main():
    for game in (1, 2):
        rng = np.random.default_rng(31 + game)
        x = neighbours(CENTRES[game], RADIUS[game])
        with tempfile.TemporaryDirectory() as d:
            pick = table(game, x, fields(rng, game), d)
            top = np.argsort(-pick.mean(0))[:8]
            keep = x[top]
            refs = np.array(CENTRES[game])
            report = table(game, np.vstack([keep, refs]), fields(rng, game), d)
        print(f"\nGAME {game}: {len(x)} neighbours; columns: pick-set mean, report-set mean, report min")
        for j, i in enumerate(top):
            print(tuple(int(v) for v in x[i]), f"{pick[:, i].mean():.3f}  "
                  f"{report[:, j].mean():.3f}  {report[:, j].min():.3f}")
        for j, c in enumerate(refs):
            r = report[:, len(keep) + j]
            print("centre", tuple(int(v) for v in c), f"       {r.mean():.3f}  {r.min():.3f}")


if __name__ == "__main__":
    main()
