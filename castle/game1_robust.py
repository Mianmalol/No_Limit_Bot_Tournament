"""Game 1 candidates scored across field scenarios.

Each scenario is split into a field and a level-up share: applicants who best-respond
to that field. Score only needs the castle marginals, so percentiles are exact.
"""
import json
import sys

import numpy as np

from field import BASES, scenario
from game1_solve import best_response


def score_all(x, field):
    """Mean Game 1 score of each row of x against the field."""
    total = np.zeros(len(x))
    for c in range(5):
        counts = np.bincount(field[:, c], minlength=101) / len(field)
        below = np.concatenate([[0.0], np.cumsum(counts)[:-1]])
        total += (c + 1) * (below + 0.5 * counts)[x[:, c]]
    return total


def build(rng):
    base = scenario(rng)
    weights = np.full(len(base), 1 / len(base))
    _, top = best_response(base, weights)
    n = int(len(base) * rng.uniform(0, 0.25))
    return np.vstack([base, np.tile(top, (n, 1))])


def main(scenarios=400):
    rng = np.random.default_rng(2026)
    fields = [build(rng) for _ in range(scenarios)]
    cand = {tuple(int(v) for v in b) for b in BASES}
    for f in fields:
        _, a = best_response(f, np.full(len(f), 1 / len(f)))
        cand.add(tuple(int(v) for v in a))
    ne = json.load(open("results/game1_ne.json"))
    support = np.array(ne["strategies"])[np.array(ne["weights"]) > 1e-6]
    cand |= {tuple(int(v) for v in s) for s in support}
    x = np.array(sorted(cand))
    # Half the scenarios pick the leaders, the other half report them.
    pick, report = fields[: scenarios // 2], fields[scenarios // 2:]

    def stats(group):
        scores = np.array([score_all(x, f) for f in group])
        pct = np.array([(score_all(f, f)[:, None] < s[None, :]).mean(0)
                        for f, s in zip(group, scores)])
        return scores, pct

    ps, pp = stats(pick)
    rs, rp = stats(report)
    order = np.argsort(-pp.mean(0))[:12]
    print(f"{len(x)} candidates, {len(pick)} pick / {len(report)} report scenarios")
    print("allocation            hold-out: score  mean pct  worst pct  top-1% rate")
    for i in order:
        print(f"{str(tuple(int(v) for v in x[i])):22s} {rs[:, i].mean():6.2f}  "
              f"{rp[:, i].mean():7.3f}  {rp[:, i].min():7.3f}  {(rp[:, i] >= .99).mean():6.2f}")
    ne_mask = np.array([tuple(r) in {tuple(s) for s in support.tolist()} for r in x.tolist()])
    print(f"NE support entries, hold-out mean pct: {rp[:, ne_mask].mean():.3f} "
          f"(best single {rp[:, ne_mask].mean(0).max():.3f})")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
