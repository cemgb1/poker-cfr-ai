# enhanced_cfr_preflop_generator_v2.py - CFR scenarios with dynamic opponent betting

"""
Enhanced CFR Preflop Scenario Generator

This module generates all possible preflop poker scenarios for CFR training.
Key refactoring: bet_size_category is NO LONGER a scenario variable.

Scenario Space:
- hand_category: 11 categories from PREFLOP_HAND_RANGES  
- hero_position: 2 positions (BTN, BB)
- stack_category: 5 categories from STACK_CATEGORIES
- blinds_level: 3 levels (low, medium, high)
- Total: 11 × 2 × 5 × 3 = 330 scenarios

Dynamic Elements (generated during simulation):
- opponent_action: The opponent's specific betting action
- bet_to_call_bb: Amount hero needs to call (varies per training iteration)
- bet_size_category: Used only for action mapping, not scenario keys
- available_actions: Actions mapped based on actual bet size vs stack

Actions: fold, call_small, call_mid, call_high, raise_small, raise_mid, raise_high
Action mapping depends on bet size as % of stack:
- call_small: ≤15% of stack
- call_mid: 15-30% of stack  
- call_high: >30% of stack
"""

from treys.card import Card
from treys.deck import Deck
from treys.evaluator import Evaluator
import random
from collections import Counter
import numpy as np

# PREFLOP HAND CATEGORIES
PREFLOP_HAND_RANGES = {
    "premium_pairs": ["AA", "KK", "QQ", "JJ", "TT"],
    "medium_pairs": ["99", "88", "77", "66"],
    "small_pairs": ["55", "44", "33", "22"],
    "premium_aces": ["AKs", "AKo", "AQs", "AQo", "AJs", "AJo"],
    "medium_aces": ["ATs", "ATo", "A9s", "A8s", "A7s", "A6s", "A5s"],
    "suited_broadway": ["KQs", "KJs", "QJs", "KTs", "QTs", "JTs"],
    "offsuit_broadway": ["KQo", "KJo", "QJo", "KTo", "QTo", "JTo"],
    "suited_connectors": ["98s", "87s", "76s", "65s", "54s", "T9s"],
    "suited_gappers": ["T8s", "97s", "86s", "75s", "64s", "J9s"],
    "weak_aces": ["A4s", "A3s", "A2s", "A4o", "A3o", "A2o"],
    "trash": ["72o", "83o", "94o", "T2o", "J3o", "Q4o", "K2o"]
}

# STACK SIZE CATEGORIES (in big blinds)
STACK_CATEGORIES = {
    "ultra_short": (8, 12),    # Desperate territory
    "short": (13, 20),         # Push/fold mode
    "medium": (21, 50),        # Standard play
    "deep": (51, 100),         # Post-flop play
    "very_deep": (101, 200)    # Deep stack strategy
}

# TOURNAMENT STAGES
TOURNAMENT_STAGES = ["early", "middle", "late", "bubble"]

# ENHANCED ACTION SET - Updated to match specification
ACTIONS = {
    "fold": 0,
    "call_small": 1,    # Call ≤15% of stack  
    "call_mid": 2,      # Call 15-30% of stack
    "call_high": 3,     # Call >30% of stack
    "raise_small": 4,   # Raise 2-2.5x
    "raise_mid": 5,     # Raise 2.5-3x  
    "raise_high": 6     # Raise 3x+ or all-in
}

def generate_enhanced_scenarios():
    """
    Generate all possible CFR scenario combinations across:
    - hand_category (from PREFLOP_HAND_RANGES)
    - hero_position (BTN, BB) 
    - stack_category (from STACK_CATEGORIES)
    - blinds_level (low, medium, high)
    
    Does NOT include bet_size_category as a scenario variable.
    Opponent bet sizes are randomized during simulation/training.
    """
    scenarios = []
    hand_categories = list(PREFLOP_HAND_RANGES.keys())
    positions = ["BTN", "BB"]
    stack_categories = list(STACK_CATEGORIES.keys())
    blinds_levels = ["low", "medium", "high"]
    
    total_combinations = len(hand_categories) * len(positions) * len(stack_categories) * len(blinds_levels)
    
    print(f"🎯 Generating all possible CFR scenario combinations...")
    print(f"🏆 Features: Hand categories, positions, stack depths, blinds levels")
    print(f"🎲 Total combinations: {total_combinations}")
    print(f"   • Hand categories: {len(hand_categories)}")
    print(f"   • Positions: {len(positions)}")  
    print(f"   • Stack categories: {len(stack_categories)}")
    print(f"   • Blinds levels: {len(blinds_levels)}")
    
    scenario_id = 0
    
    for hand_category in hand_categories:
        for hero_position in positions:
            for stack_category in stack_categories:
                for blinds_level in blinds_levels:
                    scenario = create_enhanced_scenario(
                        scenario_id, hand_category, hero_position, 
                        stack_category, blinds_level
                    )
                    if scenario:
                        scenarios.append(scenario)
                        scenario_id += 1
    
    print(f"🏆 Generated {len(scenarios)} enhanced scenarios")
    
    # Show enhanced distribution
    analyze_scenario_distribution(scenarios)
    
    # Display scenario space analysis
    analyze_scenario_space()
    
    return scenarios

def analyze_scenario_space():
    """Analyze and display total possible scenario combinations"""
    print(f"\n🧠 SCENARIO SPACE ANALYSIS:")
    
    # Calculate theoretical maximum scenarios (bet_size_category removed)
    hand_categories = len(PREFLOP_HAND_RANGES)
    positions = 2  # BTN, BB
    stack_categories = len(STACK_CATEGORIES)  
    blinds_levels = 3  # low, medium, high
    
    total_possible = hand_categories * positions * stack_categories * blinds_levels
    
    print(f"Hand categories: {hand_categories}")
    print(f"Positions: {positions} (BTN, BB)")
    print(f"Stack categories: {stack_categories} ({list(STACK_CATEGORIES.keys())})")
    print(f"Blinds levels: {blinds_levels} (low, medium, high)")
    print(f"")
    print(f"🎯 TOTAL SCENARIO COMBINATIONS: {total_possible:,}")
    print(f"   This represents the complete scenario space that CFR explores")
    print(f"   Opponent bet sizes are randomized during training for each scenario")
    print(f"   Actions are mapped to buckets based on actual bet size vs stack")

def create_enhanced_scenario(scenario_id, hand_category, hero_position, stack_category, blinds_level):
    """
    Create enhanced scenario with deterministic combination of:
    - hand_category: The hand strength category
    - hero_position: BTN or BB
    - stack_category: Stack depth category
    - blinds_level: Tournament blinds level
    
    Opponent bet size is NOT part of scenario key - it's randomized during simulation.
    """
    try:
        # Generate hero's cards
        hero_cards = generate_hole_cards_for_category(hand_category)
        if not hero_cards:
            return None
        
        # Use provided position and stack category
        min_stack, max_stack = STACK_CATEGORIES[stack_category]
        hero_stack_bb = random.randint(min_stack, max_stack)
        
        # Create enhanced scenario WITHOUT bet_size_category in the key
        scenario = {
            "scenario_id": scenario_id,
            "hand_category": hand_category,
            "hero_cards": cards_to_str(hero_cards),
            "hero_cards_int": hero_cards,
            "hero_position": hero_position,
            "villain_position": "BB" if hero_position == "BTN" else "BTN",
            
            # Enhanced stack context
            "hero_stack_bb": hero_stack_bb,
            "stack_category": stack_category,
            "villain_stack_bb": random.randint(min_stack, max_stack),  # Similar stack
            
            # Tournament context
            "blinds_level": blinds_level,
            
            # NOTE: opponent_action, bet_to_call_bb, bet_size_category, and pot_odds
            # are NOT included in the base scenario since they vary during training.
            # They will be generated dynamically during simulation.
        }
        
        return scenario
        
    except Exception as e:
        print(f"Error creating enhanced scenario: {e}")
        return None

def generate_dynamic_betting_context(scenario):
    """
    Generate dynamic betting context for a scenario during simulation.
    This replaces the static bet_size_category with randomized opponent betting.
    
    Returns dict with:
    - opponent_action: The opponent's betting action
    - bet_to_call_bb: Amount hero needs to call
    - bet_size_category: Categorization of the bet size (for action mapping)
    - pot_odds: Current pot odds
    - available_actions: Actions available to hero given this bet
    """
    hero_stack_bb = scenario['hero_stack_bb']
    hero_position = scenario['hero_position']
    
    # Generate opponent's action and bet size (same logic as before)
    opponent_action, bet_to_call_bb = generate_opponent_betting(hero_stack_bb, hero_position)
    
    # Calculate bet sizing context for action mapping
    bet_size_category = get_bet_size_category(bet_to_call_bb, hero_stack_bb)
    pot_odds = calculate_pot_odds(bet_to_call_bb, hero_position)
    
    # Get available actions based on this specific bet
    available_actions = get_available_actions(hero_stack_bb, bet_to_call_bb)
    
    return {
        "opponent_action": opponent_action,
        "bet_to_call_bb": bet_to_call_bb,
        "bet_size_category": bet_size_category,
        "pot_odds": round(pot_odds, 2),
        "available_actions": available_actions
    }

def generate_opponent_betting(hero_stack_bb, hero_position):
    """Generate realistic opponent betting based on stack size"""
    
    if hero_position == "BTN":
        # Hero is on button, BB can check or raise
        if random.random() < 0.6:  # 60% check
            return "check", 0
        else:  # 40% raise
            if hero_stack_bb <= 15:  # Short stack opponent
                raise_size = random.choice([hero_stack_bb, hero_stack_bb // 2])  # All-in or big raise
            else:
                raise_size = random.choice([2, 3, 4])  # Normal raises
            return "raise", raise_size
    else:
        # Hero is BB, BTN can limp or raise
        if random.random() < 0.3:  # 30% limp
            return "limp", 1  # Just call the big blind
        else:  # 70% raise
            if hero_stack_bb <= 15:
                raise_size = random.choice([hero_stack_bb, hero_stack_bb // 2])
            else:
                raise_size = random.choice([2, 3, 4, 5])
            return "raise", raise_size

def get_bet_size_category(bet_to_call, stack_size):
    """Categorize bet size relative to stack"""
    if bet_to_call == 0:
        return "no_bet"
    
    bet_ratio = bet_to_call / stack_size
    
    if bet_ratio <= 0.1:
        return "tiny"
    elif bet_ratio <= 0.3:
        return "small"
    elif bet_ratio <= 0.6:
        return "large"
    else:
        return "huge"

def calculate_pot_odds(bet_to_call, position):
    """Calculate pot odds for the call"""
    if bet_to_call == 0:
        return 0
    
    # Starting pot (blinds)
    pot_size = 1.5  # SB + BB
    
    # Add opponent's bet
    pot_size += bet_to_call
    
    # Pot odds = pot size / bet to call
    return pot_size / bet_to_call if bet_to_call > 0 else 0

def get_available_actions(stack_bb, bet_to_call):
    """Get available actions based on stack size and bet"""
    actions = ["fold"]
    
    if bet_to_call == 0:
        # No bet to call - can only raise (no checking in this preflop model)
        actions.extend(["raise_small", "raise_mid", "raise_high"])
    else:
        # Bet to call - categorize call size based on stack percentage
        bet_ratio = bet_to_call / stack_bb
        
        # Add appropriate call actions based on bet size
        if bet_ratio <= 0.15:
            actions.append("call_small")
        elif bet_ratio <= 0.30:
            actions.append("call_mid") 
        else:
            actions.append("call_high")
        
        # Add raise options if stack allows
        if stack_bb > bet_to_call + 3:  # Must have chips left for meaningful raise
            actions.extend(["raise_small", "raise_mid", "raise_high"])
    
    return actions

def generate_hole_cards_for_category(hand_category):
    """Generate hole cards for a specific category"""
    try:
        if hand_category not in PREFLOP_HAND_RANGES:
            return None
        
        # Pick random hand from category
        hand_combo = random.choice(PREFLOP_HAND_RANGES[hand_category])
        
        # Parse hand notation
        if len(hand_combo) == 2 and hand_combo[0] == hand_combo[1]:
            # Pocket pair
            rank = hand_combo[0]
            suits = ['s', 'h', 'd', 'c']
            suit1, suit2 = random.sample(suits, 2)
            card1 = Card.new(rank + suit1)
            card2 = Card.new(rank + suit2)
            
        elif hand_combo.endswith('s'):
            # Suited
            ranks = hand_combo[:-1]
            suit = random.choice(['s', 'h', 'd', 'c'])
            card1 = Card.new(ranks[0] + suit)
            card2 = Card.new(ranks[1] + suit)
            
        elif hand_combo.endswith('o'):
            # Offsuit
            ranks = hand_combo[:-1]
            suits = ['s', 'h', 'd', 'c']
            suit1, suit2 = random.sample(suits, 2)
            card1 = Card.new(ranks[0] + suit1)
            card2 = Card.new(ranks[1] + suit2)
            
        else:
            # Default handling
            ranks = hand_combo
            if random.random() < 0.3:
                suit = random.choice(['s', 'h', 'd', 'c'])
                card1 = Card.new(ranks[0] + suit)
                card2 = Card.new(ranks[1] + suit)
            else:
                suits = ['s', 'h', 'd', 'c']
                suit1, suit2 = random.sample(suits, 2)
                card1 = Card.new(ranks[0] + suit1)
                card2 = Card.new(ranks[1] + suit2)
        
        return [card1, card2]
        
    except Exception as e:
        print(f"Error generating cards: {e}")
        return None

def simulate_enhanced_showdown(hero_cards, villain_cards, hero_action, villain_action, 
                             hero_stack, villain_stack, bet_amount):
    """Enhanced showdown simulation with stack considerations"""
    try:
        # Handle folds
        if hero_action == "fold":
            return {"result": "villain_wins", "hero_stack_change": -min(bet_amount, hero_stack)}
        if villain_action == "fold":
            return {"result": "hero_wins", "hero_stack_change": bet_amount}
        
        # Both players in - simulate to showdown
        deck = Deck()
        all_cards = hero_cards + villain_cards
        for card in all_cards:
            if card in deck.cards:
                deck.cards.remove(card)
        
        board = deck.draw(5)
        
        # Evaluate hands
        evaluator = Evaluator()
        hero_score = evaluator.evaluate(hero_cards, board)
        villain_score = evaluator.evaluate(villain_cards, board)
        
        # Determine winner and stack changes
        pot_size = calculate_final_pot_size(hero_action, villain_action, bet_amount)
        
        if hero_score < villain_score:  # Hero wins
            return {"result": "hero_wins", "hero_stack_change": pot_size}
        elif villain_score < hero_score:  # Villain wins
            return {"result": "villain_wins", "hero_stack_change": -bet_amount}
        else:  # Tie
            return {"result": "tie", "hero_stack_change": 0}
            
    except Exception as e:
        return {"result": "tie", "hero_stack_change": 0}

def calculate_final_pot_size(hero_action, villain_action, base_bet):
    """Calculate final pot size based on actions"""
    pot = 1.5  # Blinds
    
    if "raise" in hero_action:
        pot += base_bet * 2
    elif "call" in hero_action:
        pot += base_bet
    elif "all_in" in hero_action:
        pot += base_bet * 3
        
    return pot

def analyze_scenario_distribution(scenarios):
    """Analyze enhanced scenario distribution"""
    print(f"\n📊 ENHANCED SCENARIO DISTRIBUTION:")
    
    # Hand categories
    hand_counts = Counter(s['hand_category'] for s in scenarios)
    print(f"\nHand Categories:")
    for hand_cat, count in hand_counts.most_common():
        print(f"  {hand_cat:18s}: {count:3d} scenarios")
    
    # Stack categories
    stack_counts = Counter(s['stack_category'] for s in scenarios)
    print(f"\nStack Categories:")
    for stack_cat, count in stack_counts.most_common():
        avg_stack = np.mean([s['hero_stack_bb'] for s in scenarios if s['stack_category'] == stack_cat])
        print(f"  {stack_cat:12s}: {count:3d} scenarios (avg: {avg_stack:.1f}bb)")
    
    # Positions
    position_counts = Counter(s['hero_position'] for s in scenarios)
    print(f"\nPositions:")
    for position, count in position_counts.most_common():
        print(f"  {position:8s}: {count:3d} scenarios")
    
    # Blinds levels
    blinds_counts = Counter(s['blinds_level'] for s in scenarios)
    print(f"\nBlinds Levels:")
    for blinds, count in blinds_counts.most_common():
        print(f"  {blinds:8s}: {count:3d} scenarios")
    
    print(f"\n💡 Note: Opponent bet sizes are randomized during training")
    print(f"   Each scenario will experience various bet sizes and action mappings")

def cards_to_str(cards):
    """Convert card integers to readable strings"""
    return " ".join([Card.int_to_pretty_str(c) for c in cards])

if __name__ == "__main__":
    # Test enhanced scenario generation
    scenarios = generate_enhanced_scenarios()
    
    print(f"\n🎯 SAMPLE ENHANCED SCENARIOS:")
    for i, scenario in enumerate(scenarios[:8]):
        print(f"{i+1}. {scenario['hand_category']:15s} | {scenario['hero_cards']:8s} | "
              f"{scenario['hero_position']:3s} | {scenario['stack_category']:12s} ({scenario['hero_stack_bb']:2d}bb) | "
              f"{scenario['blinds_level']:6s}")
        
        # Show example of dynamic betting context
        betting_ctx = generate_dynamic_betting_context(scenario)
        print(f"   Example bet: {betting_ctx['opponent_action']} {betting_ctx['bet_to_call_bb']}bb -> "
              f"{', '.join(betting_ctx['available_actions'])}")
    
    print(f"\n💡 Enhanced CFR Ready!")
    print(f"   • Deterministic scenario combinations (no bet_size_category)")
    print(f"   • Dynamic opponent bet sizing during simulation")
    print(f"   • Action mapping based on actual bet vs stack ratios")
    print(f"   • Robust to opponent bet size distributions")
