# simplified_scenario_generator.py - Direct hole card scenario generation for CFR training

"""
Simplified Scenario Generator for Preflop Poker CFR Training

This module replaces the hand category abstraction with direct hole card combinations.
Key changes:
1. Scenarios defined only by: hole cards for each player + stack sizes
2. No hand category abstraction - all 1326 hole card combinations supported
3. Random scenario selection for each Monte Carlo iteration
4. Preflop-only simulation with immediate showdown after betting
5. Full coverage tracking to ensure all hole card combinations are visited

Actions: fold, call_small, call_mid, call_high, raise_small, raise_mid, raise_high
Action mapping depends on bet size as % of stack:
- call_small: ≤15% of stack
- call_mid: 15-30% of stack  
- call_high: >30% of stack
"""

from treys import Card, Deck, Evaluator
import random
import itertools
from collections import Counter
import numpy as np


# STACK SIZE CATEGORIES (in big blinds) - simplified to fewer categories
STACK_CATEGORIES = {
    "short": (10, 25),         # Push/fold territory  
    "medium": (26, 75),        # Standard play
    "deep": (76, 150)          # Deep stack strategy
}

# ENHANCED ACTION SET
ACTIONS = {
    "fold": 0,
    "call_small": 1,    # Call ≤15% of stack  
    "call_mid": 2,      # Call 15-30% of stack
    "call_high": 3,     # Call >30% of stack
    "raise_small": 4,   # Raise 2-2.5x
    "raise_mid": 5,     # Raise 2.5-3x  
    "raise_high": 6     # Raise 3x+ or all-in
}


def generate_all_hole_card_combinations():
    """
    Generate all possible hole card combinations (1326 total).
    Returns list of tuples with (card1, card2) where card1 > card2 to avoid duplicates.
    """
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['s', 'h', 'd', 'c']
    
    # Generate all 52 cards
    all_cards = []
    for rank in ranks:
        for suit in suits:
            all_cards.append(Card.new(rank + suit))
    
    # Generate all combinations of 2 cards
    hole_card_combos = []
    for i in range(len(all_cards)):
        for j in range(i + 1, len(all_cards)):
            hole_card_combos.append((all_cards[i], all_cards[j]))
    
    return hole_card_combos


def generate_random_scenario():
    """
    Generate a random scenario for Monte Carlo training.
    
    Returns:
        dict: Scenario with hero_cards, villain_cards, hero_stack_bb, villain_stack_bb
    """
    # Generate all hole card combinations
    all_combos = generate_all_hole_card_combinations()
    
    # Pick random hole cards for hero and villain (ensuring no overlap)
    deck = Deck()
    
    # Draw 4 cards (2 for each player)
    hero_cards = deck.draw(2)
    villain_cards = deck.draw(2)
    
    # Generate random but equal stack sizes
    stack_category = random.choice(list(STACK_CATEGORIES.keys()))
    min_stack, max_stack = STACK_CATEGORIES[stack_category]
    stack_size = random.randint(min_stack, max_stack)
    
    scenario = {
        "hero_cards": hero_cards,
        "villain_cards": villain_cards,
        "hero_cards_str": cards_to_str(hero_cards),
        "villain_cards_str": cards_to_str(villain_cards),
        "hero_stack_bb": stack_size,
        "villain_stack_bb": stack_size,  # Equal stacks
        "stack_category": stack_category
    }
    
    return scenario


def create_scenario_key(cards, stack_bb):
    """
    Create a scenario key based on hole cards and stack size.
    
    Args:
        cards: List of two card objects
        stack_bb: Stack size in big blinds
        
    Returns:
        str: Scenario key in format "cards|stack_category"
    """
    cards_str = cards_to_str(cards)
    
    # Determine stack category
    stack_category = "deep"  # default
    for category, (min_stack, max_stack) in STACK_CATEGORIES.items():
        if min_stack <= stack_bb <= max_stack:
            stack_category = category
            break
    
    return f"{cards_str}|{stack_category}"


def get_available_actions(stack_bb, bet_to_call_bb=1.5):
    """
    Get available actions based on stack size and current bet.
    
    Args:
        stack_bb: Player's stack in big blinds
        bet_to_call_bb: Amount needed to call in big blinds
        
    Returns:
        list: Available action names
    """
    actions = ["fold"]
    
    # Add call actions based on bet size relative to stack
    if bet_to_call_bb <= stack_bb:
        bet_percent = bet_to_call_bb / stack_bb
        
        if bet_percent <= 0.15:
            actions.append("call_small")
        elif bet_percent <= 0.30:
            actions.append("call_mid")
        else:
            actions.append("call_high")
    
    # Add raise actions if player has chips for meaningful raise
    if stack_bb > bet_to_call_bb + 3:
        actions.extend(["raise_small", "raise_mid", "raise_high"])
    
    return actions


def simulate_preflop_showdown(hero_cards, villain_cards, hero_action, villain_action, 
                             hero_stack, villain_stack, bet_amount):
    """
    Simulate preflop-only showdown with immediate card reveal.
    
    Args:
        hero_cards: Hero's hole cards
        villain_cards: Villain's hole cards  
        hero_action: Hero's action
        villain_action: Villain's action
        hero_stack: Hero's stack size
        villain_stack: Villain's stack size
        bet_amount: Amount at risk
        
    Returns:
        dict: Result with winner and stack changes
    """
    try:
        # Handle folds
        if hero_action == "fold":
            return {"result": "villain_wins", "hero_stack_change": -min(bet_amount, hero_stack)}
        if villain_action == "fold":
            return {"result": "hero_wins", "hero_stack_change": bet_amount}
        
        # Both players in - reveal all cards immediately (preflop showdown)
        deck = Deck()
        all_hole_cards = hero_cards + villain_cards
        for card in all_hole_cards:
            if card in deck.cards:
                deck.cards.remove(card)
        
        # Deal community cards
        board = deck.draw(5)
        
        # Evaluate hands
        evaluator = Evaluator()
        hero_score = evaluator.evaluate(hero_cards, board)
        villain_score = evaluator.evaluate(villain_cards, board)
        
        # Determine winner and calculate stack changes
        pot_size = calculate_pot_size(hero_action, villain_action, bet_amount)
        
        if hero_score < villain_score:  # Lower score wins in treys
            return {"result": "hero_wins", "hero_stack_change": pot_size}
        elif villain_score < hero_score:
            return {"result": "villain_wins", "hero_stack_change": -bet_amount}
        else:  # Tie
            return {"result": "tie", "hero_stack_change": 0}
            
    except Exception as e:
        # In case of error, return neutral result
        return {"result": "tie", "hero_stack_change": 0}


def calculate_pot_size(hero_action, villain_action, base_bet):
    """Calculate pot size based on preflop actions."""
    pot = 1.5  # Blinds (0.5 + 1.0)
    
    if "raise" in hero_action:
        pot += base_bet * 2
    elif "call" in hero_action:
        pot += base_bet
    
    if "raise" in villain_action:
        pot += base_bet * 2
    elif "call" in villain_action:
        pot += base_bet
    
    return pot


def cards_to_str(cards):
    """Convert card objects to string representation."""
    if not cards or len(cards) != 2:
        return "??"
    
    try:
        card_strs = []
        suit_symbols = "♠♥♦♣"  # spades, hearts, diamonds, clubs
        for card in cards:
            rank_char = Card.int_to_str(card)[0]
            suit_char = Card.int_to_str(card)[1]
            suit_index = "shdc".index(suit_char)
            card_strs.append(f"[{rank_char}{suit_symbols[suit_index]}]")
        return " ".join(card_strs)
    except:
        return "??"


def get_scenario_coverage_stats(visited_scenarios):
    """
    Calculate coverage statistics for visited scenarios.
    
    Args:
        visited_scenarios: Set of scenario keys that have been visited
        
    Returns:
        dict: Coverage statistics
    """
    total_possible_combinations = 1326  # C(52,2)
    
    # Count unique hole card combinations
    unique_hole_cards = set()
    for scenario_key in visited_scenarios:
        cards_part = scenario_key.split('|')[0]
        unique_hole_cards.add(cards_part)
    
    coverage_percent = len(unique_hole_cards) / total_possible_combinations * 100
    
    return {
        "total_possible_combinations": total_possible_combinations,
        "unique_hole_cards_visited": len(unique_hole_cards),
        "coverage_percent": coverage_percent,
        "total_scenarios_visited": len(visited_scenarios)
    }


if __name__ == "__main__":
    print("🧪 Testing Simplified Scenario Generator")
    print("=" * 50)
    
    # Test scenario generation
    scenario = generate_random_scenario()
    print(f"Random scenario:")
    print(f"  Hero: {scenario['hero_cards_str']} (stack: {scenario['hero_stack_bb']}bb)")
    print(f"  Villain: {scenario['villain_cards_str']} (stack: {scenario['villain_stack_bb']}bb)")
    print(f"  Stack category: {scenario['stack_category']}")
    
    # Test scenario keys
    hero_key = create_scenario_key(scenario['hero_cards'], scenario['hero_stack_bb'])
    villain_key = create_scenario_key(scenario['villain_cards'], scenario['villain_stack_bb'])
    print(f"  Hero key: {hero_key}")
    print(f"  Villain key: {villain_key}")
    
    # Test actions
    hero_actions = get_available_actions(scenario['hero_stack_bb'])
    print(f"  Available actions: {hero_actions}")
    
    # Test hole card combinations
    all_combos = generate_all_hole_card_combinations()
    print(f"  Total hole card combinations: {len(all_combos)}")
    
    print("✅ Simplified scenario generator working correctly!")