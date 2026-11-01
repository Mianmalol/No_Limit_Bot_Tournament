"""Pick allocations on a field built from real entries, holding out one real round.

Each field mixes mapped Riddler rounds with focal archetypes, concentrated entries and
solver-driven entries that best-respond to the rest. Candidates come from fields of
three rounds; the fold's numbers come from fields of the fourth.

    python3 robust2.py 1|2 [pick_scenarios] [report_scenarios]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

from field import jitter, scenario
from riddler import best_response, scores
from stress import CONCENTRATED

EXE = "./game2"
V5 = np.arange(1, 6)
SIZE = 1200
CONC_MAX = {1: 0.15, 2: 0.35}
ROUNDS = {n: a for n, a in ((int(k[1:]), v) for k, v in
                            np.load("results/riddler_merged.npz").items())}


def g2_best(path, seed, top=1):
    out = subprocess.run([EXE, "br", path, "4000", "24", str(top), str(seed)],
                         capture_output=True, text=True, check=True).stdout
    return [tuple(int(v) for v in line.split()[1:]) for line in out.splitlines()]


def best_for(game, field, path, seed):
    if game == 1:
        return [tuple(int(v) for v in best_response(field, V5))]
    np.savetxt(path, field, fmt="%d")
    return g2_best(path, seed)


def build(rng, rounds, game, path):
    keys = sorted(rounds)
    weights = rng.dirichlet(np.ones(len(keys)))
    picks = rng.choice(len(keys), SIZE, p=weights)
    real = np.vstack([rounds[keys[k]][rng.integers(len(rounds[keys[k]]))] for k in picks])
    parts = [real]
    n_focal = int(SIZE * rng.uniform(0, 0.3) / 0.7)
    if n_focal:
        parts.append(scenario(rng, n_focal))
    n_conc = int(SIZE * rng.uniform(0, CONC_MAX[game]) / 0.65)
    parts += [jitter(CONCENTRATED[rng.integers(len(CONCENTRATED))], rng.poisson(3), rng)[None, :]
              for _ in range(n_conc)]
    base = np.vstack(parts)
    top = np.array(best_for(game, base, path, int(rng.integers(1 << 30)))[0])
    n_level = int(len(base) * rng.uniform(0, 0.3))
    level = [jitter(top, rng.poisson(3), rng)[None, :] for _ in range(n_level)]
    return np.vstack([base, *level]) if level else base


def rank_on(game, x, fields, rng, d):
    """(scenarios, candidates) arrays of score and percentile."""
    all_s, all_p = [], []
    for i, f in enumerate(fields):
        if game == 1:
            s, others = scores(x, f, V5), scores(f, f, V5)
        else:
            fp, cp = f"{d}/f", f"{d}/c"
            np.savetxt(fp, f, fmt="%d")
            np.savetxt(cp, np.vstack([x, f[rng.integers(len(f), size=300)]]), fmt="%d")
            out = subprocess.run([EXE, "eval", fp, cp, "20000", str(i + 1)],
                                 capture_output=True, text=True, check=True).stdout
            v = np.array([float(line.split()[0]) for line in out.splitlines()])
            s, others = v[: len(x)], v[len(x):]
        all_s.append(s)
        all_p.append((others[:, None] < s[None, :]).mean(0))
    return np.array(all_s), np.array(all_p)


def main(game, n_pick=25, n_report=20):
    rng = np.random.default_rng(4242 + game)
    union, report_sets = set(), []
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/b"
        print(f"GAME {game}: each fold's winner, chosen without the held-out round")
        for held in sorted(ROUNDS):
            train = {k: v for k, v in ROUNDS.items() if k != held}
            fields = [build(rng, train, game, path) for _ in range(n_pick)]
            cand = set()
            for i, f in enumerate(fields):
                cand.update(best_for(game, f, path, 100 + i))
            x = np.array(sorted(cand))
            _, tp = rank_on(game, x, fields, rng, d)
            winner = x[np.argmax(tp.mean(0))]
            hold = [build(rng, {held: ROUNDS[held]}, game, path) for _ in range(n_report)]
            report_sets += hold
            union |= cand
            s, p = rank_on(game, winner[None, :], hold, rng, d)
            print(f"held round {held}: {tuple(int(v) for v in winner)} score {s.mean():.2f} "
                  f"mean pct {p.mean():.3f} worst {p.min():.3f}", flush=True)
        x = np.array(sorted(union))
        s, p = rank_on(game, x, report_sets, rng, d)
    order = np.argsort(-p.mean(0))[:10]
    print(f"\nAll {len(x)} candidates on all {len(report_sets)} held-out fields "
          "(slightly optimistic: chosen on the same fields)")
    print("allocation            score  mean pct  worst pct  top-1% rate")
    for i in order:
        print(f"{str(tuple(int(v) for v in x[i])):22s} {s[:, i].mean():5.2f}  {p[:, i].mean():7.3f}"
              f"  {p[:, i].min():7.3f}  {(p[:, i] >= .99).mean():6.2f}")
    np.savetxt(f"results/finalists_g{game}.txt", x[order], fmt="%d")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
