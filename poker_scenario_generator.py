from treys import Card, Deck, Evaluator
import random

# CATEGORY MAPS
HAND_CATEGORY_MAP = {
    "premium_pairs":  ["AA", "KK", "QQ", "JJ", "TT"],
    "medium_pairs":   ["99", "88", "77", "66"],
    "small_pairs":    ["55", "44", "33", "22"],
    "premium_aces":   ["AK", "AQ", "AJ"],
    "medium_aces":    ["AT", "A9", "A8"],
    "connectors":     ["98", "87", "76", "65"],
    "trash":          ["72", "83", "94", "T3"]
}

PAIR_TYPE_MAP = {
    "overpair":        lambda hand, board: is_overpair(hand, board),
    "top_pair_strong": lambda hand, board: is_top_pair(hand, board, strong=True),
    "top_pair_weak":   lambda hand, board: is_top_pair(hand, board, strong=False),
    "middle_pair":     lambda hand, board: is_middle_pair(hand, board),
    "bottom_pair":     lambda hand, board: is_bottom_pair(hand, board),
    "no_pair":         lambda hand, board: not has_pair(hand, board)
}

DRAW_TYPE_MAP = {
    "nut_flush_draw":  lambda hand, board: is_nut_flush_draw(hand, board),
    "flush_draw":      lambda hand, board: is_flush_draw(hand, board),
    "open_ended":      lambda hand, board: is_open_ended_straight_draw(hand, board),
    "gutshot":         lambda hand, board: is_gutshot_straight_draw(hand, board),
    "combo_draw":      lambda hand, board: is_combo_draw(hand, board),
    "no_draw":         lambda hand, board: not (is_flush_draw(hand, board) or is_open_ended_straight_draw(hand, board) or is_gutshot_straight_draw(hand, board))
}

def random_suit():
    return random.choice(['s','h','d','c'])

def generate_hole_cards(hand_category):
    rank_str = random.choice(HAND_CATEGORY_MAP[hand_category])
    card1 = Card.new(rank_str[0] + random_suit())
    card2 = Card.new(rank_str[1] + random_suit())
    while card1 == card2:
        card2 = Card.new(rank_str[1] + random_suit())
    return [card1, card2]

def generate_valid_board(hole_cards, pair_type, draw_category, max_tries=100):
    tries = 0
    while tries < max_tries:
        deck = Deck()
        # Remove hero's hole cards from deck
        for c in hole_cards:
            deck.cards.remove(c)
        board = deck.draw(5)
        # Validate that board + hand matches desired pair/draw
        pair_ok = PAIR_TYPE_MAP[pair_type](hole_cards, board)
        draw_ok = DRAW_TYPE_MAP[draw_category](hole_cards, board)
        if pair_ok and draw_ok:
            return board
        tries += 1
    return None

def scenario_to_labels(hole_cards, board):
    evaluator = Evaluator()
    score = evaluator.evaluate(hole_cards, board)
    rank_class = evaluator.get_rank_class(score)
    # Map rank_class to relative_strength
    strength_map = {1: "nuts", 2: "near_nuts", 3: "strong", 4: "medium", 5: "weak", 6: "air"}
    rel_strength = strength_map.get(rank_class, "medium")
    # For demo, assign other categories randomly
    board_texture = random.choice(['dry_low','dry_high','wet_coordinated','paired','monotone','connected','rainbow'])
    board_danger = random.choice(['safe','moderate','dangerous','extremely_wet'])
    position = random.choice(['BTN','BB'])
    preflop_action = random.choice(['limped','raised','3bet','4bet'])
    postflop_action = random.choice(['check_check','bet_call','bet_raise','check_raise','all_in'])
    pot_size = random.choice(['small_pot','medium_pot','large_pot','huge_pot'])
    stack_depth = random.choice(['short','medium','deep'])
    return rel_strength, board_texture, board_danger, position, preflop_action, postflop_action, pot_size, stack_depth

def cards_to_str(cards):
    return " ".join([Card.int_to_pretty_str(c) for c in cards])

def generate_scenarios(n_samples=100):
    scenarios = []
    for _ in range(n_samples):
        hand_category = random.choice(list(HAND_CATEGORY_MAP.keys()))
        pair_type = random.choice(list(PAIR_TYPE_MAP.keys()))
        draw_category = random.choice(list(DRAW_TYPE_MAP.keys()))
        hole_cards = generate_hole_cards(hand_category)
        board = generate_valid_board(hole_cards, pair_type, draw_category)
        if board is None:
            continue # skip impossible combos
        rel_strength, board_texture, board_danger, position, preflop_action, postflop_action, pot_size, stack_depth = scenario_to_labels(hole_cards, board)
        scenarios.append({
            "hand_category": hand_category,
            "hole_cards": cards_to_str(hole_cards),
            "board": cards_to_str(board),
            "pair_type": pair_type,
            "draw_category": draw_category,
            "relative_strength": rel_strength,
            "board_texture": board_texture,
            "board_danger": board_danger,
            "position": position,
            "preflop_action": preflop_action,
            "postflop_action": postflop_action,
            "pot_size": pot_size,
            "stack_depth": stack_depth
        })
    return scenarios

# --------- HAND EVALUATION HELPERS ---------
def has_pair(hole, board):
    evaluator = Evaluator()
    score = evaluator.evaluate(hole, board)
    rank_class = evaluator.get_rank_class(score)
    return rank_class in [3,4,5] # pair, two pair, trips

def is_overpair(hole, board):
    # Overpair: both hole cards paired, and both higher than board cards
    ranks = [Card.get_rank_int(c) for c in hole]
    if ranks[0] == ranks[1]:
        board_ranks = [Card.get_rank_int(c) for c in board]
        return all(ranks[0] > r for r in board_ranks)
    return False

def is_top_pair(hole, board, strong=True):
    board_ranks = [Card.get_rank_int(c) for c in board]
    top_board = max(board_ranks)
    for c in hole:
        if Card.get_rank_int(c) == top_board:
            if strong:
                return top_board >= 11 # J or higher
            else:
                return top_board < 11
    return False

def is_middle_pair(hole, board):
    board_ranks = sorted([Card.get_rank_int(c) for c in board])
    median = board_ranks[len(board_ranks)//2]
    for c in hole:
        if Card.get_rank_int(c) == median:
            return True
    return False

def is_bottom_pair(hole, board):
    board_ranks = [Card.get_rank_int(c) for c in board]
    bottom_board = min(board_ranks)
    for c in hole:
        if Card.get_rank_int(c) == bottom_board:
            return True
    return False

def is_flush_draw(hole, board):
    # At least 4 of same suit between hand and board, but not made flush
    suits = [Card.get_suit_int(c) for c in hole+board]
    for s in range(4):
        if suits.count(s) == 4:
            return True
    return False

def is_nut_flush_draw(hole, board):
    # Ace high flush draw (simplified)
    if is_flush_draw(hole, board):
        for c in hole+board:
            if Card.get_rank_int(c) == 14: # Ace
                return True
    return False

def is_open_ended_straight_draw(hole, board):
    # Simplified: checks for 4 consecutive ranks in hand+board
    ranks = set([Card.get_rank_int(c) for c in hole+board])
    for low in range(2, 11):
        if all([(low+i) in ranks for i in range(4)]):
            return True
    return False

def is_gutshot_straight_draw(hole, board):
    # Simplified: checks for 4 out of 5 needed for straight (with a gap)
    ranks = set([Card.get_rank_int(c) for c in hole+board])
    for low in range(2, 11):
        missing = [low+i for i in range(5) if (low+i) not in ranks]
        if len(missing) == 1:
            return True
    return False

def is_combo_draw(hole, board):
    # Both flush and straight draw
    return is_flush_draw(hole, board) and (is_open_ended_straight_draw(hole, board) or is_gutshot_straight_draw(hole, board))

# --------- USAGE EXAMPLE ---------
if __name__ == "__main__":
    scenarios = generate_scenarios(50)
    for s in scenarios[:5]:
        print(s)
