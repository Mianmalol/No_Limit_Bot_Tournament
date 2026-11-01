"""Game 2 candidates scored across field scenarios, on the same pick/report split idea
as game1_robust: best responses come from pick scenarios, numbers from the others.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from field import BASES, scenario

EXE = str(Path(__file__).parent / "game2")


def write(path, rows):
    np.savetxt(path, rows, fmt="%d")


def best_responses(field_path, seed, top=3):
    out = subprocess.run([EXE, "br", field_path, "4000", "24", str(top), str(seed)],
                         capture_output=True, text=True, check=True).stdout
    return [tuple(int(v) for v in line.split()[1:]) for line in out.splitlines()]


def evaluate(field_path, cand_path, seed):
    out = subprocess.run([EXE, "eval", field_path, cand_path, "20000", str(seed)],
                         capture_output=True, text=True, check=True).stdout
    return np.array([float(line.split()[0]) for line in out.splitlines()])


def main(scenarios=120):
    rng = np.random.default_rng(1103)
    archive = np.loadtxt("results/game2_archive.txt", dtype=np.int64)
    tail = archive[len(archive) // 2:]
    fields = []
    for _ in range(scenarios):
        f = scenario(rng)
        extra = tail[rng.integers(len(tail), size=int(len(f) * rng.uniform(0, 0.25)))]
        fields.append(np.vstack([f, extra]))

    with tempfile.TemporaryDirectory() as d:
        field_path, cand_path = f"{d}/field", f"{d}/cand"
        cand = {tuple(int(v) for v in b) for b in BASES}
        for i, f in enumerate(fields[: scenarios // 2]):
            write(field_path, f)
            cand.update(best_responses(field_path, i + 1))
        cand |= {tuple(int(v) for v in s) for s in tail[::10]}
        x = np.array(sorted(cand))
        scores, pcts = [], []
        for i, f in enumerate(fields[scenarios // 2:]):
            sample = f[rng.integers(len(f), size=300)]
            write(field_path, f)
            write(cand_path, np.vstack([x, sample]))
            s = evaluate(field_path, cand_path, 5000 + i)
            scores.append(s[: len(x)])
            pcts.append((s[len(x):, None] < s[: len(x)][None, :]).mean(0))
    scores, pcts = np.array(scores), np.array(pcts)
    order = np.argsort(-pcts.mean(0))[:12]
    print(f"{len(x)} candidates, {scenarios // 2} report scenarios; fair score 3.75")
    print("allocation            score  mean pct  worst pct  top-1% rate")
    for i in order:
        print(f"{str(tuple(int(v) for v in x[i])):22s} {scores[:, i].mean():5.2f}  "
              f"{pcts[:, i].mean():7.3f}  {pcts[:, i].min():7.3f}  {(pcts[:, i] >= .99).mean():6.2f}")
    keys = {tuple(int(v) for v in s) for s in tail[::10]}
    mask = np.array([tuple(r) in keys for r in x.tolist()])
    print(f"Equilibrium-archive entries: mean pct {pcts[:, mask].mean():.3f}, "
          f"best {pcts[:, mask].mean(0).max():.3f}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
