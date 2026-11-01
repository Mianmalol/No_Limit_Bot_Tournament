"""Real entries from FiveThirtyEight's Battle for Riddler Nation, mapped to five castles.

Rounds 1-4 use castles worth 1..10. Merging adjacent pairs gives value shares of
5.5/12.7/20/27.3/34.5%, close to this game's 6.7/13.3/20/26.7/33.3%. Round 5 has 13
castles with no comparable merge, so it is left out.

Run directly to validate the mapping and backtest round-to-round prediction.
"""
import csv
from pathlib import Path

import numpy as np

RAW = Path(__file__).parent.parent / "data" / "raw"
FILES = {1: "castle-solutions.csv", 2: "castle-solutions-2.csv",
         3: "castle-solutions-3.csv", 4: "castle-solutions-4.csv"}


def load_round(n):
    """Valid 10-castle entries of a round, whole numbers summing to 100."""
    rows = list(csv.reader(open(RAW / FILES[n], encoding="utf-8", errors="replace")))
    out = []
    for r in rows[1:]:
        try:
            v = [float(x) for x in r[:10]]
        except ValueError:
            continue
        if sum(v) == 100 and min(v) >= 0 and all(x == int(x) for x in v):
            out.append([int(x) for x in v])
    return np.array(out, dtype=np.int64)


def merge(entries):
    return entries.reshape(-1, 5, 2).sum(2)


def scores(x, field, values):
    """Mean castle points of each row of x against the field."""
    total = np.zeros(len(x))
    for c, v in enumerate(values):
        counts = np.bincount(field[:, c], minlength=101)[:101] / len(field)
        below = np.concatenate([[0.0], np.cumsum(counts)[:-1]])
        total += v * (below + 0.5 * counts)[x[:, c]]
    return total


def best_response(field, values):
    """Exact best allocation against a field, by knapsack over its castle marginals."""
    gains = []
    for c, v in enumerate(values):
        counts = np.bincount(field[:, c], minlength=101)[:101] / len(field)
        below = np.concatenate([[0.0], np.cumsum(counts)[:-1]])
        gains.append(v * (below + 0.5 * counts))
    dp = np.full(101, -1e18)
    dp[0] = 0
    back = []
    for g in gains:
        new, pick = np.full(101, -1e18), np.zeros(101, dtype=int)
        for used in range(101):
            if dp[used] < -1e17:
                continue
            for s in range(101 - used):
                if dp[used] + g[s] > new[used + s]:
                    new[used + s], pick[used + s] = dp[used] + g[s], s
        dp = new
        back.append(pick)
    out, used = [], 100
    for pick in reversed(back):
        out.append(int(pick[used]))
        used -= pick[used]
    return np.array(out[::-1])


def pct(x, field, values):
    return float((scores(field, field, values) < scores(x[None, :], field, values)[0]).mean())


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    v10, v5 = np.arange(1, 11), np.arange(1, 6)
    rounds = {n: load_round(n) for n in FILES}
    merged = {n: merge(r) for n, r in rounds.items()}
    print("round entries  merged mean                  top-two>=80  zero-castle-1  all-mult-5")
    for n, m in merged.items():
        print(n, len(m), m.mean(0).round(1), f"{(m[:, 3:].sum(1) >= 80).mean():.3f}",
              f"{(m[:, 0] == 0).mean():.3f}", f"{(m % 5 == 0).all(1).mean():.3f}")
    print("\nMapping check: rank correlation of 10-castle score with merged 5-castle score")
    for n in FILES:
        print(n, round(spearman(scores(rounds[n], rounds[n], v10),
                                scores(merged[n], merged[n], v5)), 3))
    print("\nBacktest: best response to earlier rounds, percentile in the next round")
    print("next  10-castle  merged-5   vs same-round best")
    for n in (2, 3, 4):
        past10 = np.vstack([rounds[k] for k in range(1, n)])
        past5 = np.vstack([merged[k] for k in range(1, n)])
        b10, b5 = best_response(past10, v10), best_response(past5, v5)
        print(n, f"{pct(b10, rounds[n], v10):8.3f}  {pct(b5, merged[n], v5):8.3f}  ",
              tuple(int(a) for a in b5), "->", tuple(int(a) for a in best_response(merged[n], v5)))
    np.savez(Path(__file__).parent / "results" / "riddler_merged.npz",
             **{f"r{n}": m for n, m in merged.items()})


if __name__ == "__main__":
    main()
