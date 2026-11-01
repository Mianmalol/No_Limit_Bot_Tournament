from fractions import Fraction as F

import scoring
from scoring import game1, game2, castle_points


def test_game1_hand_cases():
    a = (0, 0, 0, 50, 50)
    b = (20, 20, 20, 20, 20)
    assert game1(a, b) == 9
    assert game1(b, a) == 6
    assert game1(a, a) == F(15, 2)


def test_game1_constant_sum():
    a, b = (10, 20, 30, 40, 0), (10, 25, 25, 20, 20)
    assert game1(a, b) + game1(b, a) == 15


def test_game2_clear_ranking():
    # Castle 5 + 4 beats everyone; the rest fall in order of the castles they take.
    players = [(0, 0, 0, 50, 50), (0, 0, 100, 0, 0), (0, 100, 0, 0, 0), (100, 0, 0, 0, 0)]
    assert castle_points(players) == [9, 3, 2, 1]
    assert game2(players) == [9, 4, 2, 0]


def test_game2_all_identical_splits_everything():
    p = (20, 20, 20, 20, 20)
    assert game2([p] * 4) == [F(15, 4)] * 4


def test_game2_tied_for_first():
    # Two players share castles 4 and 5; uncontested castle 1 gives everyone a quarter.
    a, b, c = (0, 0, 0, 50, 50), (0, 0, 100, 0, 0), (0, 100, 0, 0, 0)
    assert castle_points([a, a, b, c]) == [F(19, 4), F(19, 4), F(13, 4), F(9, 4)]
    assert game2([a, a, b, c]) == [F(13, 2), F(13, 2), 2, 0]


def test_game2_payouts_always_sum_to_15():
    import random
    rng = random.Random(1)
    for _ in range(2000):
        group = []
        for _ in range(4):
            cuts = sorted(rng.randint(0, 100) for _ in range(4))
            parts = [b - a for a, b in zip([0] + cuts, cuts + [100])]
            group.append(tuple(parts))
        assert sum(game2(group)) == 15


def test_unawarded_zero_castles_lose_points():
    scoring.ZERO_SPLITS = False
    try:
        assert game1((0, 25, 25, 25, 25), (0, 25, 25, 25, 25)) == 7
    finally:
        scoring.ZERO_SPLITS = True
