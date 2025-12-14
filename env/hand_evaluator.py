"""
Hand evaluation functions for poker showdowns.
"""

from itertools import combinations
from typing import List, Tuple, Optional, Dict

from env.cards import Card


# ----------------------------
# Hand evaluation (showdown)
# ----------------------------

def _is_straight(values_desc: List[int]) -> Optional[int]:
    """
    values_desc: sorted unique ranks, descending (e.g. [14,13,12,11,10,...])
    Returns the high card of the straight if present, else None.
    Handles wheel A-2-3-4-5 as high=5.
    """
    vals = sorted(set(values_desc), reverse=True)
    if 14 in vals:  # Ace can be low
        vals.append(1)

    run = 1
    for i in range(len(vals) - 1):
        if vals[i] - 1 == vals[i + 1]:
            run += 1
            if run >= 5:
                # high card is the start of this 5-run
                high = vals[i - (run - 2)]
                return 5 if high == 1 else high
        else:
            run = 1
    return None


def rank_5card_hand(cards5: List[Card]) -> Tuple[int, List[int]]:
    """
    Returns a comparable rank:
      (category, kickers...)
    category: 8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight,
              3 trips, 2 two pair, 1 pair, 0 high card
    Kickers list breaks ties.
    """
    vals = sorted([c.rank_value for c in cards5], reverse=True)
    suits = [c.suit for c in cards5]

    # counts
    counts: Dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    # sort by (count desc, value desc)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    is_flush = len(set(suits)) == 1
    straight_high = _is_straight(vals)

    if is_flush and straight_high is not None:
        return (8, [straight_high])

    if groups[0][1] == 4:
        quad = groups[0][0]
        kicker = max(v for v in vals if v != quad)
        return (7, [quad, kicker])

    if groups[0][1] == 3 and groups[1][1] == 2:
        trips = groups[0][0]
        pair = groups[1][0]
        return (6, [trips, pair])

    if is_flush:
        return (5, vals)

    if straight_high is not None:
        return (4, [straight_high])

    if groups[0][1] == 3:
        trips = groups[0][0]
        kickers = [v for v in vals if v != trips]
        return (3, [trips] + kickers)

    if groups[0][1] == 2 and groups[1][1] == 2:
        high_pair = max(groups[0][0], groups[1][0])
        low_pair = min(groups[0][0], groups[1][0])
        kicker = max(v for v in vals if v != high_pair and v != low_pair)
        return (2, [high_pair, low_pair, kicker])

    if groups[0][1] == 2:
        pair = groups[0][0]
        kickers = [v for v in vals if v != pair]
        return (1, [pair] + kickers)

    return (0, vals)


def best_hand_rank(seven_cards: List[Card]) -> Tuple[int, List[int]]:
    best = (-1, [])
    for combo in combinations(seven_cards, 5):
        r = rank_5card_hand(list(combo))
        if r > best:
            best = r
    return best
