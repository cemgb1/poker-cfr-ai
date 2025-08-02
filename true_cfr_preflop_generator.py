# true_cfr_preflop_generator.py - Generates preflop scenarios for true CFR learning

from treys import Card, Deck, Evaluator
import random
from collections import Counter
import itertools

# PREFLOP HAND CATEGORIES for CFR training
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

# POSITIONS in heads-up
POSITIONS = ["BTN", "BB"]  # Button and Big Blind

# PREFLOP ACTIONS
PREFLOP_ACTIONS = ["fold", "call", "raise"]

def generate_preflop_scenarios(n_scenarios=1000):
    """
    Generate preflop CFR training scenarios
    Each scenario represents a decision point for the hero
    """
    scenarios = []
    hand_categories = list(PREFLOP_HAND_RANGES.keys())
    
    print(f"🎯 Generating {n_scenarios} preflop CFR scenarios...")
    
    # Ensure coverage across all hand categories
    scenarios_per_category = n_scenarios // len(hand_categories)
    remainder = n_scenarios % len(hand_categories)
    
    scenario_id = 0
    
    for i, hand_category in enumerate(hand_categories):
        target_for_category = scenarios_per_category + (1 if i < remainder else 0)
        
        print(f"🎲 Generating {target_for_category} scenarios for {hand_category}...")
        
        for _ in range(target_for_category):
            scenario = create_preflop_scenario(scenario_id, hand_category)
            if scenario:
                scenarios.append(scenario)
                scenario_id += 1
    
    print(f"🏆 Generated {len(scenarios)} preflop scenarios")
    
    # Show distribution
    category_counts = Counter(s['hand_category'] for s in scenarios)
    print(f"\n📊 Hand Category Distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat:20s}: {count:3d} scenarios")
    
    return scenarios

def create_preflop_scenario(scenario_id, hand_category):
    """Create a single preflop scenario"""
    try:
        # Generate hero's hole cards
        hero_cards = generate_hole_cards_for_category(hand_category)
        if not hero_cards:
            return None
        
        # Random position (heads-up)
        hero_position = random.choice(POSITIONS)
        
        # Generate game context
        stack_depth = random.choice(["short", "medium", "deep"])
        blind_level = random.choice(["low", "medium", "high"])
        
        # Create scenario dictionary
        scenario = {
            "scenario_id": scenario_id,
            "hand_category": hand_category,
            "hero_cards": cards_to_str(hero_cards),
            "hero_cards_int": hero_cards,  # For treys evaluation
            "hero_position": hero_position,
            "villain_position": "BB" if hero_position == "BTN" else "BTN",
            "stack_depth": stack_depth,
            "blind_level": blind_level,
            "game_stage": "preflop"
        }
        
        return scenario
        
    except Exception as e:
        print(f"Error creating scenario: {e}")
        return None

def generate_hole_cards_for_category(hand_category):
    """Generate hole cards for a specific category"""
    try:
        if hand_category not in PREFLOP_HAND_RANGES:
            return None
        
        # Pick random hand from category
        hand_combo = random.choice(PREFLOP_HAND_RANGES[hand_category])
        
        # Parse hand notation (e.g., "AKs", "72o", "AA")
        if len(hand_combo) == 2 and hand_combo[0] == hand_combo[1]:
            # Pocket pair (e.g., "AA")
            rank = hand_combo[0]
            suits = ['s', 'h', 'd', 'c']
            suit1, suit2 = random.sample(suits, 2)  # Different suits for pair
            card1 = Card.new(rank + suit1)
            card2 = Card.new(rank + suit2)
            
        elif hand_combo.endswith('s'):
            # Suited (e.g., "AKs")
            ranks = hand_combo[:-1]
            suit = random.choice(['s', 'h', 'd', 'c'])
            card1 = Card.new(ranks[0] + suit)
            card2 = Card.new(ranks[1] + suit)
            
        elif hand_combo.endswith('o'):
            # Offsuit (e.g., "AKo")
            ranks = hand_combo[:-1]
            suits = ['s', 'h', 'd', 'c']
            suit1, suit2 = random.sample(suits, 2)  # Different suits
            card1 = Card.new(ranks[0] + suit1)
            card2 = Card.new(ranks[1] + suit2)
            
        else:
            # No designation - random (shouldn't happen with our ranges)
            ranks = hand_combo
            if random.random() < 0.3:  # 30% suited
                suit = random.choice(['s', 'h', 'd', 'c'])
                card1 = Card.new(ranks[0] + suit)
                card2 = Card.new(ranks[1] + suit)
            else:  # 70% offsuit
                suits = ['s', 'h', 'd', 'c']
                suit1, suit2 = random.sample(suits, 2)
                card1 = Card.new(ranks[0] + suit1)
                card2 = Card.new(ranks[1] + suit2)
        
        return [card1, card2]
        
    except Exception as e:
        print(f"Error generating cards for {hand_category}: {e}")
        return None

def simulate_preflop_showdown(hero_cards, villain_cards):
    """
    Simulate preflop to showdown using treys
    Returns the result from hero's perspective
    """
    try:
        # Create deck and remove hero/villain cards
        deck = Deck()
        all_cards = hero_cards + villain_cards
        for card in all_cards:
            if card in deck.cards:
                deck.cards.remove(card)
        
        # Deal 5 community cards
        board = deck.draw(5)
        
        # Evaluate both hands
        evaluator = Evaluator()
        hero_score = evaluator.evaluate(hero_cards, board)
        villain_score = evaluator.evaluate(villain_cards, board)
        
        # Lower score wins in treys
        if hero_score < villain_score:
            return "hero_wins"
        elif villain_score < hero_score:
            return "villain_wins"
        else:
            return "tie"
            
    except Exception as e:
        print(f"Error in showdown simulation: {e}")
        return "tie"

def calculate_hand_equity(hero_cards, num_simulations=1000):
    """
    Calculate hero's equity against random villain hands
    Used for validating CFR learning
    """
    wins = 0
    ties = 0
    
    for _ in range(num_simulations):
        # Generate random villain hand
        deck = Deck()
        for card in hero_cards:
            if card in deck.cards:
                deck.cards.remove(card)
        
        villain_cards = deck.draw(2)
        result = simulate_preflop_showdown(hero_cards, villain_cards)
        
        if result == "hero_wins":
            wins += 1
        elif result == "tie":
            ties += 1
    
    equity = (wins + 0.5 * ties) / num_simulations
    return equity

def cards_to_str(cards):
    """Convert card integers to readable strings"""
    return " ".join([Card.int_to_pretty_str(c) for c in cards])

def analyze_preflop_ranges():
    """Analyze the equity of different preflop ranges"""
    print("\n🔍 PREFLOP EQUITY ANALYSIS:")
    print("=" * 50)
    
    for category, hands in PREFLOP_HAND_RANGES.items():
        # Sample a few hands from each category
        sample_hands = hands[:3] if len(hands) >= 3 else hands
        category_equities = []
        
        for hand_str in sample_hands:
            # Generate cards for this hand
            scenario = {"hand_category": category}
            # Temporarily modify to generate specific hand
            original_ranges = PREFLOP_HAND_RANGES[category]
            PREFLOP_HAND_RANGES[category] = [hand_str]
            
            cards = generate_hole_cards_for_category(category)
            if cards:
                equity = calculate_hand_equity(cards, 100)  # Quick calculation
                category_equities.append(equity)
                print(f"  {hand_str:4s}: {equity:.1%} equity")
            
            # Restore original ranges
            PREFLOP_HAND_RANGES[category] = original_ranges
        
        if category_equities:
            avg_equity = sum(category_equities) / len(category_equities)
            print(f"  {category:20s} avg equity: {avg_equity:.1%}")
        print()

if __name__ == "__main__":
    # Test preflop scenario generation
    scenarios = generate_preflop_scenarios(200)
    
    # Show sample scenarios
    print(f"\n🎯 SAMPLE PREFLOP SCENARIOS:")
    for i, scenario in enumerate(scenarios[:8]):
        print(f"{i+1}. {scenario['hand_category']:15s} | {scenario['hero_cards']:8s} | "
              f"{scenario['hero_position']:3s} | {scenario['stack_depth']:6s}")
    
    # Analyze ranges
    analyze_preflop_ranges()
    
    print(f"\n💡 Ready for True CFR Training!")
    print(f"   • {len(scenarios)} diverse preflop scenarios")
    print(f"   • Real game simulation with treys")
    print(f"   • Self-play with regret minimization")
    print(f"   • Learning from actual win/loss outcomes")
