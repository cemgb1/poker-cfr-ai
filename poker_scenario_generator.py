# Enhanced poker_scenario_generator.py - More realistic scenario generation

from treys import Card, Deck, Evaluator
import random
import itertools

# EXPANDED CATEGORY MAPS
HAND_CATEGORY_MAP = {
    "premium_pairs":  ["AA", "KK", "QQ", "JJ", "TT"],
    "medium_pairs":   ["99", "88", "77", "66"],
    "small_pairs":    ["55", "44", "33", "22"],
    "premium_aces":   ["AK", "AQ", "AJ"],
    "medium_aces":    ["AT", "A9", "A8", "A7", "A6", "A5"],
    "suited_connectors": ["98s", "87s", "76s", "65s", "54s"],
    "offsuit_connectors": ["98o", "87o", "76o", "65o"],
    "broadway":       ["KQ", "KJ", "QJ", "QT", "JT"],
    "suited_gappers": ["T8s", "97s", "86s", "75s"],
    "weak_aces":      ["A4", "A3", "A2"],
    "trash":          ["72", "83", "94", "T3", "J4", "Q2"]
}

# More realistic combinations - each pair type works with specific scenarios
VALID_COMBINATIONS = {
    "overpair": {
        "compatible_hands": ["premium_pairs", "medium_pairs"],
        "compatible_draws": ["no_draw", "flush_draw", "gutshot"],
        "board_types": ["dry_low", "dry_medium", "rainbow"]
    },
    "top_pair_strong": {
        "compatible_hands": ["premium_aces", "broadway"],
        "compatible_draws": ["no_draw", "flush_draw", "gutshot", "open_ended"],
        "board_types": ["dry_high", "wet_coordinated", "paired"]
    },
    "top_pair_weak": {
        "compatible_hands": ["medium_aces", "weak_aces"],
        "compatible_draws": ["no_draw", "flush_draw"],
        "board_types": ["dry_low", "dry_medium"]
    },
    "middle_pair": {
        "compatible_hands": ["medium_pairs", "small_pairs"],
        "compatible_draws": ["no_draw", "flush_draw", "gutshot"],
        "board_types": ["dry_medium", "wet_coordinated"]
    },
    "bottom_pair": {
        "compatible_hands": ["small_pairs", "weak_aces"],
        "compatible_draws": ["no_draw", "flush_draw"],
        "board_types": ["dry_high", "rainbow"]
    },
    "no_pair": {
        "compatible_hands": ["suited_connectors", "broadway", "trash"],
        "compatible_draws": ["flush_draw", "open_ended", "gutshot", "combo_draw", "no_draw"],
        "board_types": ["dry_low", "wet_coordinated", "monotone", "rainbow"]
    }
}

def generate_realistic_scenario():
    """Generate a single realistic poker scenario"""
    # Pick a pair type first
    pair_type = random.choice(list(VALID_COMBINATIONS.keys()))
    combo_info = VALID_COMBINATIONS[pair_type]
    
    # Pick compatible categories
    hand_category = random.choice(combo_info["compatible_hands"])
    draw_category = random.choice(combo_info["compatible_draws"])
    board_texture = random.choice(combo_info["board_types"])
    
    # Generate hole cards
    hole_cards = generate_hole_cards_enhanced(hand_category)
    if hole_cards is None:
        return None
    
    # Generate compatible board
    board = generate_compatible_board(hole_cards, pair_type, draw_category, board_texture)
    if board is None:
        return None
    
    # Generate other realistic attributes
    rel_strength = calculate_relative_strength(hole_cards, board)
    board_danger = assess_board_danger(board, draw_category)
    
    # Contextual factors
    position = random.choice(['BTN', 'BB', 'SB', 'EP', 'MP', 'LP'])
    preflop_action = random.choice(['limped', 'raised', '3bet', '4bet', 'called'])
    postflop_action = random.choice(['check_check', 'bet_call', 'bet_raise', 'check_raise', 'all_in', 'bet_fold'])
    pot_size = random.choice(['small_pot', 'medium_pot', 'large_pot', 'huge_pot'])
    stack_depth = random.choice(['short', 'medium', 'deep', 'very_deep'])
    
    return {
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
    }

def generate_hole_cards_enhanced(hand_category):
    """Enhanced hole card generation with suited/offsuit handling"""
    try:
        if hand_category not in HAND_CATEGORY_MAP:
            return None
            
        rank_str = random.choice(HAND_CATEGORY_MAP[hand_category])
        
        # Handle suited (s) and offsuit (o) designations
        if rank_str.endswith('s'):
            # Suited - same suit
            ranks = rank_str[:-1]
            suit = random_suit()
            card1 = Card.new(ranks[0] + suit)
            card2 = Card.new(ranks[1] + suit)
        elif rank_str.endswith('o'):
            # Offsuit - different suits
            ranks = rank_str[:-1]
            suit1 = random_suit()
            suit2 = random_suit()
            while suit2 == suit1:
                suit2 = random_suit()
            card1 = Card.new(ranks[0] + suit1)
            card2 = Card.new(ranks[1] + suit2)
        else:
            # No designation - random suits (but avoid pairs having same suit)
            if ranks[0] == ranks[1]:
                # Pocket pair - must be different suits
                suit1 = random_suit()
                suit2 = random_suit()
                while suit2 == suit1:
                    suit2 = random_suit()
                card1 = Card.new(ranks[0] + suit1)
                card2 = Card.new(ranks[1] + suit2)
            else:
                # Random for non-pairs
                if random.random() < 0.3:  # 30% chance suited
                    suit = random_suit()
                    card1 = Card.new(ranks[0] + suit)
                    card2 = Card.new(ranks[1] + suit)
                else:  # 70% chance offsuit
                    suit1 = random_suit()
                    suit2 = random_suit()
                    while suit2 == suit1:
                        suit2 = random_suit()
                    card1 = Card.new(ranks[0] + suit1)
                    card2 = Card.new(ranks[1] + suit2)
        
        return [card1, card2]
    except:
        return None

def generate_compatible_board(hole_cards, pair_type, draw_category, board_texture, max_tries=50):
    """Generate board that's compatible with desired categories"""
    for _ in range(max_tries):
        try:
            deck = Deck()
            # Remove hole cards
            for c in hole_cards:
                deck.cards.remove(c)
            
            # Generate board based on requirements
            board = create_targeted_board(deck, hole_cards, pair_type, draw_category, board_texture)
            
            if board and len(board) == 5:
                # Validate the board meets requirements
                if validate_scenario(hole_cards, board, pair_type, draw_category):
                    return board
        except:
            continue
    return None

def create_targeted_board(deck, hole_cards, pair_type, draw_category, board_texture):
    """Create a board targeted to specific requirements"""
    try:
        if pair_type == "overpair":
            # Create board with cards lower than pocket pair
            hole_ranks = [Card.get_rank_int(c) for c in hole_cards]
            if hole_ranks[0] == hole_ranks[1]:  # Ensure it's a pair
                pair_rank = hole_ranks[0]
                board = []
                available_ranks = [r for r in range(2, pair_rank) if len(board) < 5]
                if len(available_ranks) >= 3:
                    for _ in range(5):
                        rank = random.choice(available_ranks)
                        suit = random.choice([0, 1, 2, 3])
                        card = Card.new(Card.STR_RANKS[rank] + Card.STR_SUITS[suit])
                        if card not in hole_cards and card not in board:
                            board.append(card)
                    if len(board) == 5:
                        return board
        
        elif pair_type == "top_pair_strong" or pair_type == "top_pair_weak":
            # Create board where one hole card pairs with highest board card
            hole_ranks = [Card.get_rank_int(c) for c in hole_cards]
            target_rank = max(hole_ranks)  # Use higher hole card as top pair
            
            board = []
            # Add the pairing card
            for suit in range(4):
                card = Card.new(Card.STR_RANKS[target_rank] + Card.STR_SUITS[suit])
                if card not in hole_cards:
                    board.append(card)
                    break
            
            # Add 4 more cards of lower rank
            lower_ranks = [r for r in range(2, target_rank)]
            for _ in range(4):
                if lower_ranks:
                    rank = random.choice(lower_ranks)
                    suit = random.choice([0, 1, 2, 3])
                    card = Card.new(Card.STR_RANKS[rank] + Card.STR_SUITS[suit])
                    if card not in hole_cards and card not in board:
                        board.append(card)
            
            if len(board) == 5:
                return board
        
        # Fallback: generate random board
        return deck.draw(5)
        
    except:
        return deck.draw(5) if len(deck.cards) >= 5 else None

def validate_scenario(hole_cards, board, expected_pair_type, expected_draw_category):
    """Validate that the generated scenario matches expectations"""
    try:
        # Check pair type
        actual_pair_type = determine_pair_type(hole_cards, board)
        if actual_pair_type != expected_pair_type:
            return False
        
        # Check draw category (more lenient)
        actual_draw_category = determine_draw_category(hole_cards, board)
        if expected_draw_category != "no_draw" and actual_draw_category == "no_draw":
            return False
        
        return True
    except:
        return False

def determine_pair_type(hole_cards, board):
    """Determine the actual pair type for validation"""
    try:
        if is_overpair(hole_cards, board):
            return "overpair"
        elif is_top_pair(hole_cards, board, strong=True):
            return "top_pair_strong"
        elif is_top_pair(hole_cards, board, strong=False):
            return "top_pair_weak"
        elif is_middle_pair(hole_cards, board):
            return "middle_pair"
        elif is_bottom_pair(hole_cards, board):
            return "bottom_pair"
        else:
            return "no_pair"
    except:
        return "no_pair"

def determine_draw_category(hole_cards, board):
    """Determine the actual draw category"""
    try:
        if is_combo_draw(hole_cards, board):
            return "combo_draw"
        elif is_nut_flush_draw(hole_cards, board):
            return "nut_flush_draw"
        elif is_flush_draw(hole_cards, board):
            return "flush_draw"
        elif is_open_ended_straight_draw(hole_cards, board):
            return "open_ended"
        elif is_gutshot_straight_draw(hole_cards, board):
            return "gutshot"
        else:
            return "no_draw"
    except:
        return "no_draw"

def calculate_relative_strength(hole_cards, board):
    """Calculate relative hand strength"""
    try:
        evaluator = Evaluator()
        score = evaluator.evaluate(hole_cards, board)
        rank_class = evaluator.get_rank_class(score)
        
        strength_map = {
            1: "nuts",
            2: "near_nuts", 
            3: "strong",
            4: "medium",
            5: "weak",
            6: "air",
            7: "air"
        }
        return strength_map.get(rank_class, "medium")
    except:
        return "medium"

def assess_board_danger(board, draw_category):
    """Assess how dangerous the board is"""
    try:
        if draw_category in ["combo_draw", "nut_flush_draw"]:
            return "extremely_dangerous"
        elif draw_category in ["flush_draw", "open_ended"]:
            return "dangerous"
        elif draw_category == "gutshot":
            return "moderate"
        else:
            return "safe"
    except:
        return "moderate"

def generate_scenarios(n_samples=1000):
    """Generate many realistic scenarios with progress tracking"""
    scenarios = []
    attempts = 0
    max_attempts = n_samples * 5  # Allow 5x attempts for better success rate
    
    print(f"🎯 Generating {n_samples} realistic poker scenarios...")
    
    while len(scenarios) < n_samples and attempts < max_attempts:
        attempts += 1
        scenario = generate_realistic_scenario()
        
        if scenario:
            scenarios.append(scenario)
            
            # Progress update
            if len(scenarios) % 100 == 0:
                print(f"  ✅ Generated {len(scenarios)}/{n_samples} scenarios (success rate: {len(scenarios)/attempts:.1%})")
    
    print(f"🏆 Final: {len(scenarios)} scenarios generated from {attempts} attempts (success rate: {len(scenarios)/attempts:.1%})")
    return scenarios

# Keep all the existing helper functions from your original file
def random_suit():
    return random.choice(['s','h','d','c'])

def cards_to_str(cards):
    return " ".join([Card.int_to_pretty_str(c) for c in cards])

# ... (keep all your existing helper functions: has_pair, is_overpair, etc.)
# [Include all the helper functions from your original paste-2.txt file here]

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

if __name__ == "__main__":
    # Test the enhanced generator
    scenarios = generate_scenarios(100)
    print(f"\n📊 SCENARIO BREAKDOWN:")
    
    # Show category distribution
    categories = {}
    for s in scenarios:
        cat = f"{s['pair_type']}|{s['draw_category']}"
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"Generated {len(categories)} unique category combinations:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:30s}: {count:3d} scenarios")
    
    # Show sample scenarios
    print(f"\n🎯 SAMPLE SCENARIOS:")
    for i, s in enumerate(scenarios[:5]):
        print(f"{i+1}. {s['hand_category']} | {s['pair_type']} | {s['draw_category']} | {s['relative_strength']}")
        print(f"   Cards: {s['hole_cards']} on {s['board']}")
