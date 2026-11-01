"""Game 1 equilibrium by double oracle with an exact best response.

Score against any field is a sum over castles of v * P(we beat that castle's
marginal), so the best response is a knapsack DP over the five marginals. The
restricted game is a symmetric zero-sum LP over the strategies found so far.

Writes results/game1_ne.json with the support and its exploitability.
"""
import json
import os
import sys
import time

import numpy as np
from numba import njit
from scipy.optimize import linprog

N, C = 100, 5
VALUES = np.array([1, 2, 3, 4, 5], dtype=np.float64)


@njit(cache=True)
def best_response(field, weights):
    """Exact best response of an allocation to a weighted field.

    field: (m, 5) int array, weights: (m,) summing to 1.
    Returns (value, allocation).
    """
    pmf = np.zeros((C, N + 1))
    for i in range(field.shape[0]):
        for c in range(C):
            pmf[c, field[i, c]] += weights[i]
    gain = np.zeros((C, N + 1))
    for c in range(C):
        below = 0.0
        for s in range(N + 1):
            gain[c, s] = (c + 1) * (below + 0.5 * pmf[c, s])
            below += pmf[c, s]
    dp = np.full((C + 1, N + 1), -1e18)
    choice = np.zeros((C + 1, N + 1), dtype=np.int64)
    dp[0, 0] = 0.0
    for c in range(C):
        for used in range(N + 1):
            if dp[c, used] < -1e17:
                continue
            for s in range(N + 1 - used):
                v = dp[c, used] + gain[c, s]
                if v > dp[c + 1, used + s]:
                    dp[c + 1, used + s] = v
                    choice[c + 1, used + s] = s
    alloc = np.zeros(C, dtype=np.int64)
    used = N
    for c in range(C, 0, -1):
        alloc[c - 1] = choice[c, used]
        used -= choice[c, used]
    return dp[C, N], alloc


@njit(cache=True)
def payoff_row(x, strategies):
    """Score of x against each strategy, minus 7.5 (the antisymmetric payoff)."""
    out = np.zeros(strategies.shape[0])
    for j in range(strategies.shape[0]):
        s = 0.0
        for c in range(C):
            if x[c] > strategies[j, c]:
                s += c + 1
            elif x[c] == strategies[j, c]:
                s += 0.5 * (c + 1)
        out[j] = s - 7.5
    return out


def solve_restricted(matrix):
    """Maximin mixture of a symmetric zero-sum game; returns weights and value."""
    k = matrix.shape[0]
    c = np.zeros(k + 1)
    c[-1] = -1.0
    a_ub = np.hstack([-matrix.T, np.ones((k, 1))])
    a_eq = np.hstack([np.ones((1, k)), np.zeros((1, 1))])
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(k), A_eq=a_eq, b_eq=[1.0],
                  bounds=[(0, None)] * k + [(None, None)], method="highs")
    w = np.clip(res.x[:-1], 0, None)
    return w / w.sum(), res.x[-1]


def main(max_iter=4000, tol=1e-3, budget_s=420):
    strategies = np.array([[7, 13, 20, 27, 33]], dtype=np.int64)
    matrix = np.zeros((1, 1))
    seen = {tuple(strategies[0])}
    best = None
    start = time.time()
    for it in range(1, max_iter + 1):
        w, _ = solve_restricted(matrix)
        value, alloc = best_response(strategies, w)
        gap = value - 7.5
        if best is None or gap < best["gap"]:
            best = {"gap": float(gap), "strategies": strategies.tolist(), "weights": w.tolist()}
        if it % 25 == 0 or gap < tol:
            print(f"iter {it} support {int((w > 1e-9).sum())}/{len(strategies)} "
                  f"exploitability {gap:.4f} t={time.time() - start:.0f}s", flush=True)
        if gap < tol or time.time() - start > budget_s:
            break
        key = tuple(int(v) for v in alloc)
        if key in seen:
            print("best response already in the set", flush=True)
            break
        seen.add(key)
        row = payoff_row(alloc, strategies)
        matrix = np.block([[matrix, -row[:, None]], [row[None, :], np.zeros((1, 1))]])
        strategies = np.vstack([strategies, alloc])
    os.makedirs("results", exist_ok=True)
    with open("results/game1_ne.json", "w") as f:
        json.dump(best, f)
    print("best exploitability", best["gap"], "of a 15-point game")
    s, w = np.array(best["strategies"]), np.array(best["weights"])
    order = np.argsort(-w)
    print("marginal mean", np.round((s * w[:, None]).sum(0), 2))
    for i in order[:20]:
        print(f"{w[i]:.4f}", s[i])


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
