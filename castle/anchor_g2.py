"""Re-rank every Game 2 allocation using a real entry's known score.

Reweights the saved full-search scores by how well each regime predicts the entry, at
several levels of trust in that one number, then re-tests finalists on fresh fields.
"""
import json
import tempfile

import numpy as np

import round2  # registers the attack regimes  # noqa: F401
import search
from search import OUT, run

ENTRY, ACTUAL = (9, 36, 55, 0, 0), 6.07
NAMED = {"new 0,39,0,61,0": (0, 39, 0, 61, 0), "earlier 13,37,48,0,2": (13, 37, 48, 0, 2),
         "old 6,31,0,63,0": (6, 31, 0, 63, 0), "1&5 28,0,0,0,72": (28, 0, 0, 0, 72)}
SCALES = (0.15, 0.3, 0.6, 1.2)


def weights(curve, gap, scale):
    w = np.exp(-curve / 0.15 - (gap / scale) ** 2 / 2)
    return w / w.sum()


def main():
    errors = json.load(open(OUT / "errors2.json"))
    names = list(errors)
    curve = np.array([errors[n] for n in names])
    n = 4598126
    allocs = np.fromfile(OUT / "all.alloc", dtype=np.int8).reshape(-1, 5)
    s = np.vstack([np.fromfile(OUT / "all.scores", dtype=np.float32).reshape(-1, n),
                   np.fromfile(OUT / "round2.scores", dtype=np.float32).reshape(-1, n)])
    keys = np.ravel_multi_index(allocs[:, :4].T.astype(np.int64), (101,) * 4)
    gap = np.abs(s[:, np.flatnonzero(keys == np.ravel_multi_index(ENTRY[:4], (101,) * 4))[0]] - ACTUAL)
    found = {}
    for scale in SCALES:
        v = weights(curve, gap, scale) @ s
        for i in np.argsort(-v)[:4]:
            found.setdefault(tuple(int(q) for q in allocs[i]), f"top at trust {scale}")
    del s
    for label, x in NAMED.items():
        found[x] = label
    found[ENTRY] = "friend's entry"
    fx = np.array(list(found))
    rng = np.random.default_rng(606)
    makers = search.g2_regimes()
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for i, name in enumerate(names):
            rows.append([[float(q) for q in l.split()[:5]]
                         for l in run("places", makers[name](rng), fx, 200000, 900 + i, d)])
    r = np.array(rows)
    fresh_gap = np.abs(r[:, list(found).index(ENTRY), 0] - ACTUAL)
    print("regime predictions for the entry (actual 6.07), fresh fields:")
    print(", ".join(f"{n} {v:.2f}" for n, v in zip(names, r[:, list(found).index(ENTRY), 0])))
    print("\nFresh-field score by trust in the entry (gap scale; bigger = less trust)")
    print(f"{'allocation':22s}{'found by':22s}" + "".join(f"{q:>7}" for q in SCALES) + "   equal  worst")
    w0 = weights(curve, fresh_gap, 0.3)
    for i in np.argsort(-(w0 @ r[:, :, 0])):
        cells = "".join(f"{weights(curve, fresh_gap, q) @ r[:, i, 0]:7.2f}" for q in SCALES)
        key = tuple(int(q) for q in fx[i])
        print(f"{str(key):22s}{found[key]:22s}{cells}   {r[:, i, 0].mean():5.2f}  {r[:, i, 0].min():5.2f}")
    print("\nPlace odds at trust 0.3 (weighted): 1st 2nd 3rd last")
    for i in np.argsort(-(w0 @ r[:, :, 0]))[:6]:
        print(str(tuple(int(q) for q in fx[i])).ljust(22), np.round(w0 @ r[:, i, 1:], 2))


if __name__ == "__main__":
    main()
