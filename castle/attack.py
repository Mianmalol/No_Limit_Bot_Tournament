"""Attack the search winners with regimes built specifically against them.

The 16 regimes in search.py were fixed before the winners were known, so none targets
them. These do: copies of the winner, an arms race at castle 5, and players who
best-respond to the winner being popular. Reports exact place odds.
"""
import json
import sys
import tempfile

import numpy as np

from search import OUT, blend, copies, g2_br, near, run, real, SIZE, NEW2, OLD2
from stress2 import sharp_g2

WINNER = (28, 0, 0, 0, 72)
PICKS = {"Winner 28,0,0,0,72": WINNER, "Runner-up 11,5,7,6,71": (11, 5, 7, 6, 71),
         "Current 13,37,48,0,2": NEW2, "Old 6,31,0,63,0": OLD2}


def arms_race(rng, n):
    """Players bidding past the winner on castle 5, spreading the rest."""
    bases = [(0, 0, 0, 25, 75), (20, 0, 0, 0, 80), (0, 15, 0, 0, 85), (5, 5, 5, 5, 80)]
    return np.vstack([near(rng, bases[rng.integers(len(bases))], 1) for _ in range(n)])


def countered(rng, share, sharp_share=0.2):
    """A field where the winner is popular, plus players best-responding to that."""
    popular = blend((sharp_share, sharp_g2), (share, copies(WINNER)))(rng)
    top = g2_br(popular)
    n = int(SIZE * 0.15)
    return np.vstack([popular[:SIZE - n], near(rng, top, n, 2)]), top


def main():
    rng = np.random.default_rng(404)
    fx = np.array(list(PICKS.values()))
    tests = {
        "20% sharp (baseline)": blend((0.2, sharp_g2))(rng),
        "+10% copies of winner": blend((0.2, sharp_g2), (0.1, copies(WINNER)))(rng),
        "+20% copies of winner": blend((0.2, sharp_g2), (0.2, copies(WINNER)))(rng),
        "+40% copies of winner": blend((0.2, sharp_g2), (0.4, copies(WINNER)))(rng),
        "+10% castle-5 race": blend((0.2, sharp_g2), (0.1, arms_race))(rng),
        "+25% castle-5 race": blend((0.2, sharp_g2), (0.25, arms_race))(rng),
    }
    counters = {}
    for share in (0.1, 0.2):
        tests[f"{share:.0%} copies + counters"], counters[share] = countered(rng, share)
    print("best responses to a popular winner:", {k: v.tolist() for k, v in counters.items()})
    out = {}
    print("\nscore [1st 2nd 3rd 4th]")
    print(f"{'':26s}" + "".join(f"{k.split()[0]:>28s}" for k in PICKS))
    with tempfile.TemporaryDirectory() as d:
        for i, (name, f) in enumerate(tests.items()):
            rows = [[float(v) for v in l.split()[:5]] for l in run("places", f, fx, 200000, 40 + i, d)]
            out[name] = rows
            cells = "".join(f"{r[0]:5.2f} [{r[1]:.0%} {r[2]:.0%} {r[3]:.0%} {r[4]:.0%}]".rjust(28) for r in rows)
            print(f"{name:26s}{cells}", flush=True)
    json.dump({"tests": out, "picks": list(PICKS)}, open(OUT / "attack.json", "w"))


if __name__ == "__main__":
    main()
