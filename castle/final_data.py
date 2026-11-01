"""Compute every number the final plots use, from one fresh run. Writes results/final.json.

Every strategy in both games is scored on every regime, so figures can't disagree.
"""
import json
import tempfile

import numpy as np

import round2  # registers the attack regimes  # noqa: F401
import search
from riddler import scores
from search import OUT, Q, V5, run
from stress2 import mix, sharp_g1, sharp_g2

G1 = {
    "Solution 1": (3, 16, 1, 37, 43),
    "Solution 2": (6, 5, 3, 42, 44),
    "Solution 3": (6, 16, 1, 36, 41),
    "Solution 4 (Final)": (3, 5, 4, 43, 45),
    "Search candidate": (6, 3, 3, 43, 45),
    "Search candidate ": (7, 18, 6, 3, 66),
    "Real entry (8.0)": (1, 26, 29, 43, 1),
    "Real entry (8.1)": (0, 0, 37, 6, 57),
    "Top two": (0, 0, 0, 40, 60),
    "Top three": (0, 0, 33, 33, 34),
    "Proportional": (7, 13, 20, 27, 33),
    "Round steps": (10, 15, 20, 25, 30),
    "Even split": (20, 20, 20, 20, 20),
}
G2 = {
    "Solution 1": (0, 0, 39, 0, 61),
    "Solution 2": (6, 31, 0, 63, 0),
    "Solution 3": (13, 37, 48, 0, 2),
    "Solution 4": (0, 39, 0, 61, 0),
    "Solution 5 (Final)": (0, 36, 0, 64, 0),
    "Search candidate": (28, 0, 0, 0, 72),
    "Search candidate ": (15, 37, 48, 0, 0),
    "Real entry (6.07)": (9, 36, 55, 0, 0),
    "Top two": (0, 0, 0, 40, 60),
    "Top three": (0, 0, 33, 33, 34),
    "Proportional": (7, 13, 20, 27, 33),
    "Round steps": (10, 15, 20, 25, 30),
    "Even split": (20, 20, 20, 20, 20),
}
REGIME_LABELS = {
    "Riddler": "Riddler entries",
    "Focal": "Focal splits",
    "Heavy": "Heavy on 1–2 castles",
    "Equilibrium": "Pure equilibrium",
    "Random": "Random entries",
    "20% copy new": "20% copy Solution 3",
    "20% copy old": "20% copy Solution 2",
    "20% copy leader": "20% copy 3,5,5,42,45",
    "10% copy 28-72": "10% copy 28,0,0,0,72",
    "20% copy 28-72": "20% copy 28,0,0,0,72",
    "Copies + counters": "28,0,0,0,72 copied + countered",
    "Level 2": "Best-responders",
    "Level 3": "Counter-best-responders",
}
SHARES = [i / 10 for i in range(10)]


def curve_error(own, pub):
    return float(np.abs(np.quantile(own, Q) - np.quantile(pub, Q)).mean())


def fit_weights(curve, gap, scale):
    w = np.exp(-np.array(curve) / 0.15 - (np.array(gap) / scale) ** 2 / 2)
    return (w / w.sum()).tolist()


def game1(rng, pub):
    x = np.array(list(G1.values()))
    makers = search.g1_regimes()
    names = list(makers)
    table, curve = [], []
    for name in names:
        f = makers[name](rng)
        table.append(scores(x, f, V5).tolist())
        curve.append(curve_error(scores(f, f, V5), pub))
    t = np.array(table)
    keys = list(G1)
    gap = (np.abs(t[:, keys.index("Real entry (8.0)")] - 8.0)
           + np.abs(t[:, keys.index("Real entry (8.1)")] - 8.1)) / 2
    return {"names": names, "scores": table, "curve": curve, "weights": fit_weights(curve, gap, 0.15)}


def game2(rng, pub, d):
    x = np.array(list(G2.values()))
    makers = search.g2_regimes()
    names = list(makers)
    errors = json.load(open(OUT / "errors2.json"))
    rows = []
    for i, name in enumerate(names):
        rows.append([[float(v) for v in l.split()[:5]]
                     for l in run("places", makers[name](rng), x, 300000, 2000 + i, d)])
    r = np.array(rows)
    keys = list(G2)
    curve = [errors[n] for n in names]
    gap = np.abs(r[:, keys.index("Real entry (6.07)"), 0] - 6.07)
    return {"names": names, "places": rows, "curve": curve, "weights": fit_weights(curve, gap, 0.3)}


def sharpness(rng, d):
    out = {"shares": SHARES, "g1": [], "g2": []}
    x1, x2 = np.array(list(G1.values())), np.array(list(G2.values()))
    for s in SHARES:
        rows1, rows2 = [], []
        for _ in range(5):
            rows1.append(scores(x1, mix(rng, [(s, lambda n: sharp_g1(rng, n))]), V5))
            f = mix(rng, [(s, lambda n: sharp_g2(rng, n))])
            rows2.append([float(l.split()[0]) for l in run("places", f, x2, 100000, 7, d)])
        out["g1"].append(np.mean(rows1, 0).tolist())
        out["g2"].append(np.mean(rows2, 0).tolist())
    return out


def main():
    pub = json.load(open("../data/leaderboard_latest.json"))
    p1 = np.array([r[1] for r in pub["g1"]])
    p2 = np.array([r[1] for r in pub["g2"]])
    rng = np.random.default_rng(1115)
    with tempfile.TemporaryDirectory() as d:
        data = {"g1_strategies": {k: list(v) for k, v in G1.items()},
                "g2_strategies": {k: list(v) for k, v in G2.items()},
                "labels": REGIME_LABELS, "g1": game1(rng, p1), "g2": game2(rng, p2, d),
                "sharpness": sharpness(rng, d)}
    json.dump(data, open("results/final.json", "w"))
    w1, w2 = np.array(data["g1"]["weights"]), np.array(data["g2"]["weights"])
    s1 = np.array(data["g1"]["scores"])
    s2 = np.array(data["g2"]["places"])[:, :, 0]
    for name, s, w, keys in (("G1", s1, w1, G1), ("G2", s2, w2, G2)):
        print(name, "strategy: weighted, equal, worst")
        for i in np.argsort(-(w @ s)):
            k = list(keys)[i]
            print(f"  {k:22s}{str(keys[k]):22s}{w @ s[:, i]:6.2f} {s[:, i].mean():6.2f} {s[:, i].min():6.2f}")


if __name__ == "__main__":
    main()
