#!/usr/bin/env python3
"""
Test script to validate Natural Game CFR Trainer functionality.

This script tests various components of the natural CFR training system
to ensure everything is working correctly.
"""

import random
import time
from natural_game_cfr_trainer import NaturalGameCFRTrainer


def test_basic_functionality():
    """Test basic trainer functionality."""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    # Initialize trainer
    trainer = NaturalGameCFRTrainer(epsilon_exploration=0.2)
    
    # Test game state generation
    game_state = trainer.generate_random_game_state()
    print(f"✅ Generated random game state:")
    print(f"   Hero: {game_state['hero_cards_str']} ({game_state['hero_hand_category']})")
    print(f"   Villain: {game_state['villain_cards_str']} ({game_state['villain_hand_category']})")
    print(f"   Position: Hero={game_state['hero_position']}, Villain={game_state['villain_position']}")
    print(f"   Stacks: Hero={game_state['hero_stack_bb']}bb, Villain={game_state['villain_stack_bb']}bb")
    print(f"   Blinds: {game_state['blinds_level']}")
    
    # Test action availability
    hero_actions = trainer.get_available_actions_for_game_state(game_state, is_hero=True)
    villain_actions = trainer.get_available_actions_for_game_state(game_state, is_hero=False)
    print(f"✅ Available actions:")
    print(f"   Hero: {hero_actions}")
    print(f"   Villain: {villain_actions}")
    
    # Test scenario classification
    hero_scenario_key = trainer.get_scenario_key_from_game_state(game_state, is_hero=True)
    villain_scenario_key = trainer.get_scenario_key_from_game_state(game_state, is_hero=False)
    print(f"✅ Scenario keys:")
    print(f"   Hero: {hero_scenario_key}")
    print(f"   Villain: {villain_scenario_key}")
    
    return trainer


def test_game_simulation():
    """Test complete game simulation."""
    print("\n🎲 Testing Game Simulation")
    print("=" * 40)
    
    trainer = NaturalGameCFRTrainer(epsilon_exploration=0.3)
    
    # Run a few simulations
    for i in range(3):
        print(f"\n🎮 Game {i+1}:")
        result = trainer.monte_carlo_game_simulation()
        
        game_state = result['game_state']
        natural_scenario = result['natural_scenario']
        payoff_result = result['payoff_result']
        
        if natural_scenario is not None:
            print(f"   Cards: {natural_scenario['hero_cards']} vs {natural_scenario['villain_cards']}")
            print(f"   Scenario: {natural_scenario['scenario_key']}")
            print(f"   Actions: {len(natural_scenario['action_history'])} moves")
            print(f"   Result: Hero {'WON' if payoff_result['hero_won'] else 'LOST'} (+{payoff_result['hero_payoff']:.2f})")
            print(f"   3-bet: {'Yes' if natural_scenario['is_3bet'] else 'No'}")
            print(f"   Showdown: {'Yes' if natural_scenario['showdown'] else 'No'}")
        else:
            print(f"   ❌ Scenario filtered (unrealistic combination)")
            print(f"   Cards: {game_state['hero_cards_str']} vs {game_state['villain_cards_str']}")
            print(f"   Actions: {len(game_state['action_history'])} moves")
            print(f"   Result: Hero {'WON' if payoff_result['hero_won'] else 'LOST'} (+{payoff_result['hero_payoff']:.2f})")
    
    return trainer


def test_strategy_evolution():
    """Test strategy evolution over time."""
    print("\n🧠 Testing Strategy Evolution")
    print("=" * 40)
    
    trainer = NaturalGameCFRTrainer(epsilon_exploration=0.1, min_visit_threshold=2)
    
    # Train for a short period
    print("🚀 Training for 100 games...")
    results = trainer.train(n_games=100, log_interval=25, save_interval=50)
    
    print(f"✅ Training completed:")
    print(f"   Games played: {results['games_played']}")
    print(f"   Unique scenarios: {results['unique_scenarios']}")
    print(f"   Hero strategies: {results['hero_strategy_scenarios']}")
    print(f"   Villain strategies: {results['villain_strategy_scenarios']}")
    
    # Show some learned strategies
    print(f"\n📊 Sample Hero Strategies:")
    sample_scenarios = list(trainer.strategy_sum.keys())[:3]
    for scenario_key in sample_scenarios:
        strategy_counts = trainer.strategy_sum[scenario_key]
        total = sum(strategy_counts.values())
        if total > 0:
            print(f"   {scenario_key}:")
            for action, count in strategy_counts.items():
                prob = count / total
                if prob > 0.1:  # Only show significant probabilities
                    print(f"     {action}: {prob:.1%}")
    
    print(f"\n📊 Sample Villain Strategies:")
    sample_villain_scenarios = list(trainer.villain_strategy_sum.keys())[:3]
    for scenario_key in sample_villain_scenarios:
        strategy_counts = trainer.villain_strategy_sum[scenario_key]
        total = sum(strategy_counts.values())
        if total > 0:
            print(f"   {scenario_key}:")
            for action, count in strategy_counts.items():
                prob = count / total
                if prob > 0.1:  # Only show significant probabilities
                    print(f"     {action}: {prob:.1%}")
    
    return trainer


def test_save_load():
    """Test save and load functionality."""
    print("\n💾 Testing Save/Load Functionality")
    print("=" * 40)
    
    # Create and train a trainer
    trainer1 = NaturalGameCFRTrainer()
    trainer1.train(n_games=20, log_interval=10, save_interval=20)
    
    # Save state
    save_file = "test_save.pkl"
    trainer1.save_training_state(save_file)
    print(f"✅ Saved state to {save_file}")
    
    # Create new trainer and load state
    trainer2 = NaturalGameCFRTrainer()
    success = trainer2.load_training_state(save_file)
    
    if success:
        print(f"✅ Successfully loaded state")
        print(f"   Games played: {trainer2.natural_metrics['games_played']}")
        print(f"   Scenarios: {len(trainer2.natural_scenarios)}")
    else:
        print(f"❌ Failed to load state")
    
    # Clean up
    import os
    if os.path.exists(save_file):
        os.remove(save_file)
        print(f"✅ Cleaned up {save_file}")
    
    return trainer2


def test_hand_categories():
    """Test that different hand categories are properly discovered."""
    print("\n🃏 Testing Hand Category Discovery")
    print("=" * 40)
    
    trainer = NaturalGameCFRTrainer(epsilon_exploration=0.3)
    
    # Run training and track hand categories
    trainer.train(n_games=200, log_interval=50, save_interval=100)
    
    # Analyze discovered hand categories
    hand_categories = {}
    for scenario in trainer.natural_scenarios:
        cat = scenario['hand_category']
        hand_categories[cat] = hand_categories.get(cat, 0) + 1
    
    print(f"✅ Discovered hand categories:")
    for cat, count in sorted(hand_categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(trainer.natural_scenarios)) * 100
        print(f"   {cat:15s}: {count:3d} games ({percentage:5.1f}%)")
    
    print(f"\n📊 Total categories discovered: {len(hand_categories)}/11")
    
    return trainer


def run_all_tests():
    """Run all tests."""
    print("🧪 Natural Game CFR Trainer Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run tests
        test_basic_functionality()
        test_game_simulation()
        test_strategy_evolution()
        test_save_load()
        test_hand_categories()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"⏱️  Total test time: {duration:.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(f"🔍 Traceback:\n{traceback.format_exc()}")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ Natural Game CFR Trainer is working correctly!")
    else:
        print("\n❌ Some tests failed!")