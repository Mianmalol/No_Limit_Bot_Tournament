"""Cross-check the C scorer against the exact Fraction scorer in scoring.py."""
import random
import subprocess
import tempfile
from itertools import product
from pathlib import Path

from scoring import game2

HERE = Path(__file__).parent


def alloc(rng):
    # Few distinct values, so that castle and rank ties are common.
    while True:
        parts = [rng.choice([0, 0, 10, 20, 25, 30, 40, 50]) for _ in range(4)]
        if sum(parts) <= 100:
            return tuple(parts + [100 - sum(parts)])


def test_c_matches_exact_scorer_on_a_one_strategy_field():
    subprocess.run(["cc", "-O2", "-o", str(HERE / "game2"), str(HERE / "game2.c"), "-lpthread"],
                   check=True)
    rng = random.Random(7)
    with tempfile.TemporaryDirectory() as d:
        field, cand = Path(d) / "field", Path(d) / "cand"
        for _ in range(300):
            x, opp = alloc(rng), alloc(rng)
            field.write_text(" ".join(map(str, opp)) + "\n")
            cand.write_text(" ".join(map(str, x)) + "\n")
            out = subprocess.run([str(HERE / "game2"), "eval", str(field), str(cand), "1"],
                                 capture_output=True, text=True, check=True).stdout
            got = float(out.split()[0])
            assert abs(got - float(game2([x, opp, opp, opp])[0])) < 1e-3, (x, opp)


def test_c_matches_exact_scorer_with_distinct_opponents():
    rng = random.Random(11)
    with tempfile.TemporaryDirectory() as d:
        field, cand = Path(d) / "field", Path(d) / "cand"
        for _ in range(20):
            x, opps = alloc(rng), [alloc(rng) for _ in range(3)]
            exact = sum(float(game2([x, a, b, c])[0]) for a, b, c in product(opps, repeat=3)) / 27
            field.write_text("".join(" ".join(map(str, o)) + "\n" for o in opps))
            cand.write_text(" ".join(map(str, x)) + "\n")
            out = subprocess.run([str(HERE / "game2"), "eval", str(field), str(cand), "400000"],
                                 capture_output=True, text=True, check=True).stdout
            assert abs(float(out.split()[0]) - exact) < 0.03, (x, opps)
