# poker_scenario_generator.py - Fixed version with guaranteed scenario counts

from treys import Card, Deck, Evaluator
import random
from collections import Counter

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

PAIR_TYPES = ["overpair", "top_pair_strong", "top_pair_weak", "middle_pair", "bottom_pair", "no_pair"]
DRAW_CATEGORIES = ["nut_flush_draw", "flush_draw", "open_ended", "gutshot", "combo_draw", "no_draw"]

def generate_scenarios_guaranteed(target_count=1000):
    """
    Generate exactly target_count scenarios with forced category distribution
    Ensures all hand categories get coverage
    """
    scenarios = []
    category_counts = Counter()
    hand_categories = list(HAND_CATEGORY_MAP.keys())
    
    print(f"🎯 Generating exactly {target_count} poker scenarios...")
    print(f"📋 Will ensure coverage across {len(hand_categories)} hand categories")
    
    # Target scenarios per hand category (roughly equal distribution)
    scenarios_per_category = target_count // len(hand_categories)
    remainder = target_count % len(hand_categories)
    
    for i, hand_category in enumerate(hand_categories):
        # Some categories get +1 scenario to handle remainder
        target_for_this_category = scenarios_per_category + (1 if i < remainder else 0)
        
        print(f"🎲 Generating {target_for_this_category} scenarios for {hand_category}...")
        
        category_scenarios = generate_for_specific_category(hand_category, target_for_this_category)
        scenarios.extend(category_scenarios)
        category_counts[hand_category] = len(category_scenarios)
        
        print(f"   ✅ Generated {len(category_scenarios)}/{target_for_this_category} for {hand_category}")
    
    print(f"\n🏆 Final: {len(scenarios)} scenarios generated (target: {target_count})")
    print(f"📊 Category distribution:")
    for cat, count in category_counts.items():
        print(f"   {cat:20s}: {count:3d} scenarios")
    
    return scenarios

def generate_for_specific_category(hand_category, target_count, max_attempts_per_scenario=100):
    """Generate scenarios for a specific hand category until we reach target"""
    scenarios = []
    attempts = 0
    max_total_attempts = target_count * max_attempts_per_scenario
    
    while len(scenarios) < target_count and attempts < max_total_attempts:
        attempts += 1
        
        # Force different pair types and draw categories for variety
        pair_type = random.choice(PAIR_TYPES)
        draw_category = random.choice(DRAW_CATEGORIES)
        
        scenario = generate_specific_scenario(hand_category, pair_type, draw_category)
        
        if scenario:
            scenarios.append(scenario)
            
        # Progress update every 50 attempts
        if attempts % 50 == 0 and len(scenarios) < target_count:
            success_rate = len(scenarios) / attempts * 100
            print(f"      Progress: {len(scenarios)}/{target_count} scenarios ({success_rate:.1f}% success)")
    
    return scenarios

def generate_specific_scenario(hand_category, pair_type, draw_category):
    """Generate a specific scenario with relaxed validation"""
    try:
        # Generate hole cards for this category
        hole_cards = generate_hole_cards_enhanced(hand_category)
        if not hole_cards:
            return None
        
        # Generate a compatible board (more lenient)
        board = generate_flexible_board(hole_cards, pair_type, draw_category)
        if not board or len(board) != 5:
            return None
        
        # Calculate actual characteristics (don't force strict matching)
        actual_pair_type = determine_pair_type(hole_cards, board)
        actual_draw_category = determine_draw_category(hole_cards, board)
        rel_strength = calculate_relative_strength(hole_cards, board)
        
        # Generate other attributes
        board_texture = random.choice(['dry_low', 'dry_high', 'wet_coordinated', 'paired', 'monotone', 'connected', 'rainbow'])
        board_danger = assess_board_danger(board, actual_draw_category)
        position = random.choice(['BTN', 'BB', 'SB', 'EP', 'MP', 'LP'])
        preflop_action = random.choice(['limped', 'raised', '3bet', '4bet', 'called'])
        postflop_action = random.choice(['check_check', 'bet_call', 'bet_raise', 'check_raise', 'all_in', 'bet_fold'])
        pot_size = random.choice(['small_pot', 'medium_pot', 'large_pot', 'huge_pot'])
        stack_depth = random.choice(['short', 'medium', 'deep', 'very_deep'])
        
        return {
            "hand_category": hand_category,
            "hole_cards": cards_to_str(hole_cards),
            "board": cards_to_str(board),
            "pair_type": actual_pair_type,  # Use actual, not forced
            "draw_category": actual_draw_category,  # Use actual, not forced
            "relative_strength": rel_strength,
            "board_texture": board_texture,
            "board_danger": board_danger,
            "position": position,
            "preflop_action": preflop_action,
            "postflop_action": postflop_action,
            "pot_size": pot_size,
            "stack_depth": stack_depth
        }
    except Exception as e:
        return None

def generate_flexible_board(hole_cards, target_pair_type, target_draw_category):
    """Generate board with more flexible validation"""
    try:
        deck = Deck()
        # Remove hole cards
        for c in hole_cards:
            if c in deck.cards:
                deck.cards.remove(c)
        
        # Simple approach: generate random board and see if it's reasonable
        if len(deck.cards) >= 5:
            board = deck.draw(5)
            return board
        else:
            return None
    except:
        # Fallback: create a basic board
        try:
            basic_ranks = [2, 7, 10, 13, 6]  # Mixed board
            basic_suits = [0, 1, 2, 3, 0]   # Mostly rainbow
            board = []
            
            for rank, suit in zip(basic_ranks, basic_suits):
                card = Card.new(Card.STR_RANKS[rank] + Card.STR_SUITS[suit])
                if card not in hole_cards:
                    board.append(card)
                else:
                    # Replace with a different card
                    for alt_rank in range(2, 15):
                        alt_card = Card.new(Card.STR_RANKS[alt_rank] + Card.STR_SUITS[suit])
                        if alt_card not in hole_cards and alt_card not in board:
                            board.append(alt_card)
                            break
            
            return board if len(board) == 5 else None
        except:
            return None

def generate_hole_cards_enhanced(hand_category):
    """Enhanced hole card generation with better error handling"""
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
            # No designation - handle pairs vs non-pairs
            ranks = rank_str
            if len(ranks) == 2 and ranks[0] == ranks[1]:
                # Pocket pair - must be different suits
                suit1 = random_suit()
                suit2 = random_suit()
                while suit2 == suit1:
                    suit2 = random_suit()
                card1 = Card.new(ranks[0] + suit1)
                card2 = Card.new(ranks[1] + suit2)
            else:
                # Non-pair - random suited/offsuit
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
    except Exception as e:
        # Fallback: create random suited connector
        try:
            ranks = random.choice(["98", "87", "76", "65"])
            suit = random_suit()
            card1 = Card.new(ranks[0] + suit)
            card2 = Card.new(ranks[1] + suit)
            return [card1, card2]
        except:
            return None

# Keep all the existing helper functions
def random_suit():
    return random.choice(['s','h','d','c'])

def cards_to_str(cards):
    return " ".join([Card.int_to_pretty_str(c) for c in cards])

def determine_pair_type(hole_cards, board):
    """Determine the actual pair type"""
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

# Legacy function for backward compatibility
def generate_scenarios(n_samples=1000):
    """Legacy function - calls the new guaranteed generation"""
    return generate_scenarios_guaranteed(n_samples)

# All the poker hand evaluation functions (keep from original)
def has_pair(hole, board):
    try:
        evaluator = Evaluator()
        score = evaluator.evaluate(hole, board)
        rank_class = evaluator.get_rank_class(score)
        return rank_class in [3,4,5] # pair, two pair, trips
    except:
        return False

def is_overpair(hole, board):
    try:
        ranks = [Card.get_rank_int(c) for c in hole]
        if len(set(ranks)) == 1:  # Pocket pair
            pair_rank = ranks[0]
            board_ranks = [Card.get_rank_int(c) for c in board]
            return all(pair_rank > r for r in board_ranks)
        return False
    except:
        return False

def is_top_pair(hole, board, strong=True):
    try:
        board_ranks = [Card.get_rank_int(c) for c in board]
        top_board = max(board_ranks)
        for c in hole:
            if Card.get_rank_int(c) == top_board:
                if strong:
                    return top_board >= 11 # J or higher
                else:
                    return top_board < 11
        return False
    except:
        return False

def is_middle_pair(hole, board):
    try:
        board_ranks = sorted([Card.get_rank_int(c) for c in board])
        if len(board_ranks) >= 3:
            median = board_ranks[len(board_ranks)//2]
            for c in hole:
                if Card.get_rank_int(c) == median:
                    return True
        return False
    except:
        return False

def is_bottom_pair(hole, board):
    try:
        board_ranks = [Card.get_rank_int(c) for c in board]
        bottom_board = min(board_ranks)
        for c in hole:
            if Card.get_rank_int(c) == bottom_board:
                return True
        return False
    except:
        return False

def is_flush_draw(hole, board):
    try:
        suits = [Card.get_suit_int(c) for c in hole+board]
        for s in range(4):
            if suits.count(s) == 4:
                return True
        return False
    except:
        return False

def is_nut_flush_draw(hole, board):
    try:
        if is_flush_draw(hole, board):
            for c in hole+board:
                if Card.get_rank_int(c) == 14: # Ace
                    return True
        return False
    except:
        return False

def is_open_ended_straight_draw(hole, board):
    try:
        ranks = set([Card.get_rank_int(c) for c in hole+board])
        for low in range(2, 11):
            if sum((low+i) in ranks for i in range(4)) >= 4:
                return True
        return False
    except:
        return False

def is_gutshot_straight_draw(hole, board):
    try:
        ranks = set([Card.get_rank_int(c) for c in hole+board])
        for low in range(2, 11):
            needed = [(low+i) for i in range(5)]
            missing = [r for r in needed if r not in ranks]
            if len(missing) == 1:
                return True
        return False
    except:
        return False

def is_combo_draw(hole, board):
    try:
        return is_flush_draw(hole, board) and (is_open_ended_straight_draw(hole, board) or is_gutshot_straight_draw(hole, board))
    except:
        return False

if __name__ == "__main__":
    # Test the guaranteed generator
    scenarios = generate_scenarios_guaranteed(100)
    print(f"\n📊 GUARANTEED SCENARIO TEST:")
    
    # Show category distribution
    categories = Counter(s['hand_category'] for s in scenarios)
    print(f"Generated {len(categories)} different hand categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:20s}: {count:3d} scenarios")
    
    # Show variety in other dimensions
    pair_types = Counter(s['pair_type'] for s in scenarios)
    draw_cats = Counter(s['draw_category'] for s in scenarios)
    
    print(f"\nPair type variety: {len(pair_types)} types")
    print(f"Draw category variety: {len(draw_cats)} types")
    
    # Show sample scenarios
    print(f"\n🎯 SAMPLE SCENARIOS:")
    for i, s in enumerate(scenarios[:5]):
        print(f"{i+1}. {s['hand_category']} | {s['pair_type']} | {s['draw_category']} | {s['relative_strength']}")
        print(f"   Cards: {s['hole_cards']} on {s['board']}")
