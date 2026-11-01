"""Exact scoring for the two Castle Challenge games, as stated on the entry form.

Assumption the form leaves open: a castle nobody contests (all 0) is treated as a
tie and split, matching "a tie splits them". ZERO_SPLITS = False scores it as
unawarded so the sensitivity can be measured.
"""
from fractions import Fraction

VALUES = (1, 2, 3, 4, 5)
PAYOUTS = (9, 4, 2, 0)
ZERO_SPLITS = True


def game1(a, b):
    """Points a scores against b, head to head."""
    total = Fraction(0)
    for x, y, v in zip(a, b, VALUES):
        if x > y:
            total += v
        elif x == y and (x > 0 or ZERO_SPLITS):
            total += Fraction(v, 2)
    return total


def castle_points(players):
    """Castle points for each of the players in one group."""
    points = [Fraction(0)] * len(players)
    for c, v in enumerate(VALUES):
        sent = [p[c] for p in players]
        top = max(sent)
        if top == 0 and not ZERO_SPLITS:
            continue
        winners = [i for i, s in enumerate(sent) if s == top]
        for i in winners:
            points[i] += Fraction(v, len(winners))
    return points


def game2(players):
    """Payout for each of 4 players; tied ranks split the payouts of their places."""
    points = castle_points(players)
    payouts = []
    for p in points:
        above = sum(q > p for q in points)
        tied = sum(q == p for q in points)
        places = PAYOUTS[above:above + tied]
        payouts.append(Fraction(sum(places), tied))
    return payouts
