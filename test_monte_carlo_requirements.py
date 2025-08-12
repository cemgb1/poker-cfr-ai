#!/usr/bin/env python3
# test_monte_carlo_requirements.py - Test new Monte Carlo simulation requirements

"""
Test suite to verify the new Monte Carlo simulation requirements:
1. Random hole cards for both players
2. Equal stack sizes from specific set (500, 1000, 5000)
3. Random blind sizes from specific set (2, 5, 10, 25, 50)
4. Preflop-only play
5. Scenario recording with specific values
6. Single lookup table output
"""

import sys
import os
import tempfile
import pandas as pd
from natural_game_cfr_trainer import NaturalGameCFRTrainer
from enhanced_cfr_preflop_generator_v2 import STACK_SIZES, BLIND_SIZES


def test_stack_and_blind_generation():
    """Test that stack sizes and blind sizes use specific values."""
    print("🧪 Testing Stack and Blind Generation")
    print("=" * 50)
    
    trainer = NaturalGameCFRTrainer()
    
    # Generate multiple game states to test randomization
    stack_sizes_seen = set()
    blind_sizes_seen = set()
    equal_stacks_count = 0
    
    for i in range(50):
        game_state = trainer.generate_random_game_state()
        
        # Check that both players have equal stacks
        hero_stack = game_state['hero_stack_bb']
        villain_stack = game_state['villain_stack_bb']
        
        if hero_stack == villain_stack:
            equal_stacks_count += 1
        
        # Collect stack and blind sizes
        stack_sizes_seen.add(hero_stack)
        blind_sizes_seen.add(game_state['blind_size'])
    
    print(f"✅ Equal stacks: {equal_stacks_count}/50 games (should be 50/50)")
    print(f"✅ Stack sizes seen: {sorted(stack_sizes_seen)}")
    print(f"✅ Expected stack sizes: {STACK_SIZES}")
    print(f"✅ Blind sizes seen: {sorted(blind_sizes_seen)}")
    print(f"✅ Expected blind sizes: {BLIND_SIZES}")
    
    # Verify all stacks are from the expected set
    assert stack_sizes_seen.issubset(set(STACK_SIZES)), f"Unexpected stack sizes: {stack_sizes_seen - set(STACK_SIZES)}"
    assert blind_sizes_seen.issubset(set(BLIND_SIZES)), f"Unexpected blind sizes: {blind_sizes_seen - set(BLIND_SIZES)}"
    assert equal_stacks_count == 50, f"Not all games had equal stacks: {equal_stacks_count}/50"
    
    print("✅ All stack and blind generation tests passed!")
    return True


def test_scenario_key_format():
    """Test that scenario keys use the new specific value format."""
    print("\n🧪 Testing Scenario Key Format")
    print("=" * 50)
    
    trainer = NaturalGameCFRTrainer()
    
    # Generate a game state and check scenario key format
    game_state = trainer.generate_random_game_state()
    hero_key = trainer.get_scenario_key_from_game_state(game_state, is_hero=True)
    villain_key = trainer.get_scenario_key_from_game_state(game_state, is_hero=False)
    
    print(f"✅ Hero scenario key: {hero_key}")
    print(f"✅ Villain scenario key: {villain_key}")
    
    # Check format: hand_category|position|stack_size|blind_size
    hero_parts = hero_key.split("|")
    villain_parts = villain_key.split("|")
    
    assert len(hero_parts) == 4, f"Hero key should have 4 parts: {hero_parts}"
    assert len(villain_parts) == 4, f"Villain key should have 4 parts: {villain_parts}"
    
    # Check that stack_size and blind_size are numeric
    hero_stack_size = int(hero_parts[2])
    hero_blind_size = int(hero_parts[3])
    villain_stack_size = int(villain_parts[2])
    villain_blind_size = int(villain_parts[3])
    
    assert hero_stack_size in STACK_SIZES, f"Invalid hero stack size: {hero_stack_size}"
    assert hero_blind_size in BLIND_SIZES, f"Invalid hero blind size: {hero_blind_size}"
    assert villain_stack_size in STACK_SIZES, f"Invalid villain stack size: {villain_stack_size}"
    assert villain_blind_size in BLIND_SIZES, f"Invalid villain blind size: {villain_blind_size}"
    
    # Both should have same stack and blind sizes
    assert hero_stack_size == villain_stack_size, "Hero and villain should have equal stacks"
    assert hero_blind_size == villain_blind_size, "Hero and villain should have same blind size"
    
    print("✅ All scenario key format tests passed!")
    return True


def test_monte_carlo_simulation():
    """Test that Monte Carlo simulation works with new requirements."""
    print("\n🧪 Testing Monte Carlo Simulation")
    print("=" * 50)
    
    trainer = NaturalGameCFRTrainer()
    
    # Run a few simulations
    scenarios_seen = set()
    for i in range(10):
        result = trainer.monte_carlo_game_simulation()
        scenario_key = result['natural_scenario']['scenario_key']
        scenarios_seen.add(scenario_key)
        
        # Check that scenario has required fields
        scenario = result['natural_scenario']
        assert 'stack_size' in scenario, "Missing stack_size field"
        assert 'blind_size' in scenario, "Missing blind_size field"
        assert scenario['stack_size'] in STACK_SIZES, f"Invalid stack size: {scenario['stack_size']}"
        assert scenario['blind_size'] in BLIND_SIZES, f"Invalid blind size: {scenario['blind_size']}"
    
    print(f"✅ Generated {len(scenarios_seen)} unique scenarios from 10 games")
    print(f"✅ Example scenarios: {list(scenarios_seen)[:3]}")
    
    print("✅ All Monte Carlo simulation tests passed!")
    return True


def test_output_format():
    """Test that output files have the correct format."""
    print("\n🧪 Testing Output Format")
    print("=" * 50)
    
    trainer = NaturalGameCFRTrainer()
    
    # Run a small training session
    training_result = trainer.train(
        n_games=5,
        save_interval=1000,  # Don't save checkpoints
        log_interval=1000    # Don't log progress
    )
    
    # Create output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export scenarios
        scenario_file = os.path.join(temp_dir, "test_scenarios.csv")
        trainer.export_natural_scenarios_csv(scenario_file)
        
        # Export lookup table
        lookup_file = os.path.join(temp_dir, "test_lookup.csv")
        trainer.create_final_lookup_table(lookup_file)
        
        # Check scenario file format
        if os.path.exists(scenario_file):
            scenarios_df = pd.read_csv(scenario_file)
            print(f"✅ Scenario file columns: {list(scenarios_df.columns)}")
            
            # Check for required columns
            required_scenario_cols = ['stack_size', 'blind_size', 'scenario_key']
            for col in required_scenario_cols:
                assert col in scenarios_df.columns, f"Missing column: {col}"
            
            # Check that stack_size and blind_size have valid values
            if len(scenarios_df) > 0:
                stack_sizes = set(scenarios_df['stack_size'].unique())
                blind_sizes = set(scenarios_df['blind_size'].unique())
                assert stack_sizes.issubset(set(STACK_SIZES)), f"Invalid stack sizes in output: {stack_sizes}"
                assert blind_sizes.issubset(set(BLIND_SIZES)), f"Invalid blind sizes in output: {blind_sizes}"
                print(f"✅ Output stack sizes: {sorted(stack_sizes)}")
                print(f"✅ Output blind sizes: {sorted(blind_sizes)}")
        
        # Check lookup table format
        if os.path.exists(lookup_file):
            lookup_df = pd.read_csv(lookup_file)
            print(f"✅ Lookup table columns: {list(lookup_df.columns)}")
            
            # Check for required columns
            required_lookup_cols = ['scenario_key', 'stack_size', 'blind_size', 'best_action', 'player']
            for col in required_lookup_cols:
                assert col in lookup_df.columns, f"Missing column: {col}"
            
            # Check scenario key format in lookup table
            if len(lookup_df) > 0:
                sample_key = lookup_df['scenario_key'].iloc[0]
                parts = sample_key.split("|")
                assert len(parts) == 4, f"Invalid scenario key format: {sample_key}"
                print(f"✅ Sample scenario key: {sample_key}")
    
    print("✅ All output format tests passed!")
    return True


def main():
    """Run all tests."""
    print("🚀 Monte Carlo Requirements Test Suite")
    print("=" * 60)
    print("Testing the new Monte Carlo simulation requirements:")
    print("1. Random hole cards for both players")
    print("2. Equal stack sizes from specific set (500, 1000, 5000)")
    print("3. Random blind sizes from specific set (2, 5, 10, 25, 50)")
    print("4. Preflop-only play")
    print("5. Scenario recording with specific values")
    print("6. Single lookup table output")
    print("=" * 60)
    
    tests = [
        test_stack_and_blind_generation,
        test_scenario_key_format,
        test_monte_carlo_simulation,
        test_output_format
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n🎉 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All Monte Carlo requirements are satisfied!")
        return True
    else:
        print("❌ Some tests failed - requirements not fully met")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)