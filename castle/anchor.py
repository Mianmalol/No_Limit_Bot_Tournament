"""Use real allocation-and-score pairs to test Game 1 regimes.

ANCHORS holds entries whose score is known. A regime that predicts them badly can't be
the real field, which is a sharper test than matching the score curve, since very
different fields can share a curve.
"""
import json

import numpy as np

import round2  # registers the attack regimes  # noqa: F401
import search
from riddler import scores
from search import NEW1, OLD1, OUT, Q, V5

ANCHORS = {(1, 26, 29, 43, 1): 8.0, (0, 0, 37, 6, 57): 8.1}
PICK = (3, 5, 4, 43, 45)
CURVE_SCALE, ANCHOR_SCALE = 0.15, 0.15


def main():
    pub = np.array([r[1] for r in json.load(open("../data/leaderboard_latest.json"))["g1"]])
    for x, s in ANCHORS.items():
        print(f"{x} score {s}: leaderboard rank near that score {(pub > s).sum() + 1}, "
              f"percentile {(pub < s).mean():.3f}; rank 1499 scores {np.sort(pub)[::-1][1498]:.3f}")
    allocs = np.fromfile(OUT / "all.alloc", dtype=np.int8).reshape(-1, 5).astype(np.int64)
    a = np.array(list(ANCHORS))
    target = np.array(list(ANCHORS.values()))
    rng = np.random.default_rng(71)
    makers = search.g1_regimes()
    names, fields = list(makers), [makers[n](rng) for n in makers]
    curve, gap, table = [], [], []
    print("\nregime                curve err  8.0   8.1   mean gap")
    for name, f in zip(names, fields):
        own = scores(f, f, V5)
        curve.append(float(np.abs(np.quantile(own, Q) - np.quantile(pub, Q)).mean()))
        pred = scores(a, f, V5)
        gap.append(float(np.abs(pred - target).mean()))
        print(f"{name:22s}{curve[-1]:8.2f}   {pred[0]:5.2f} {pred[1]:5.2f} {gap[-1]:6.2f}", flush=True)
        table.append(scores(allocs, f, V5).astype(np.float32))
    s = np.array(table).T
    curve, gap = np.array(curve), np.array(gap)
    weight = np.exp(-curve / CURVE_SCALE - (gap / ANCHOR_SCALE) ** 2 / 2)
    weight /= weight.sum()
    print("\nweights with the anchor:", ", ".join(f"{n} {w:.2f}" for n, w in zip(names, weight) if w > 0.01))
    v = s @ weight
    found = {tuple(int(q) for q in allocs[i]): "top weighted" for i in np.argsort(-v)[:5]}
    worst = s.min(1)
    guarded = np.where(worst >= 7.0, v, -1)
    found.update({tuple(int(q) for q in allocs[i]): "top with floor 7" for i in np.argsort(-guarded)[:3]})
    for x, label in ((PICK, "current pick"), (NEW1, "earlier pick"), (OLD1, "old pick")):
        found[x] = label
    x = np.array(list(found))
    fresh = np.array([scores(x, makers[n](rng), V5) for n in names]).T
    print("\nallocation           found by            weighted  equal  worst  worst regime")
    for i in np.argsort(-(fresh @ weight)):
        key = tuple(int(q) for q in x[i])
        print(f"{str(key):20s} {found[key]:18s} {fresh[i] @ weight:8.2f} {fresh[i].mean():6.2f} "
              f"{fresh[i].min():6.2f}  {names[int(np.argmin(fresh[i]))]}")


if __name__ == "__main__":
    main()
