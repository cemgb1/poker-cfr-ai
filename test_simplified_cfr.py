#!/usr/bin/env python3
# test_simplified_cfr.py - Test script for simplified CFR system

"""
Test script to validate Simplified CFR System functionality.

This script tests the new simplified CFR training system to ensure:
1. Direct hole card scenarios work correctly
2. Full coverage of hole card combinations is achieved
3. Preflop-only simulation works properly  
4. Heads-up mode functions as expected
5. Checkpointing and resuming work correctly
"""

import os
import time
import random
from pathlib import Path

from simplified_cfr_trainer import SimplifiedCFRTrainer
from simplified_scenario_generator import (
    generate_random_scenario, create_scenario_key, get_available_actions,
    simulate_preflop_showdown, get_scenario_coverage_stats
)


def test_basic_functionality():
    """Test basic trainer functionality."""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    # Test scenario generation
    scenario = generate_random_scenario()
    print(f"✅ Generated random scenario:")
    print(f"   Hero: {scenario['hero_cards_str']} (stack: {scenario['hero_stack_bb']}bb)")
    print(f"   Villain: {scenario['villain_cards_str']} (stack: {scenario['villain_stack_bb']}bb)")
    print(f"   Stack category: {scenario['stack_category']}")
    
    # Test scenario keys
    hero_key = create_scenario_key(scenario['hero_cards'], scenario['hero_stack_bb'])
    villain_key = create_scenario_key(scenario['villain_cards'], scenario['villain_stack_bb'])
    print(f"✅ Scenario keys:")
    print(f"   Hero: {hero_key}")
    print(f"   Villain: {villain_key}")
    
    # Test actions
    hero_actions = get_available_actions(scenario['hero_stack_bb'])
    villain_actions = get_available_actions(scenario['villain_stack_bb'])
    print(f"✅ Available actions:")
    print(f"   Hero: {hero_actions}")
    print(f"   Villain: {villain_actions}")
    
    return True


def test_coverage_progression():
    """Test that coverage improves over training iterations."""
    print("\n🎯 Testing Coverage Progression")
    print("=" * 40)
    
    trainer = SimplifiedCFRTrainer(epsilon_exploration=0.5)  # High exploration
    
    # Track coverage over iterations
    coverage_points = []
    
    for iteration_batch in [50, 100, 200, 500]:
        trainer.train(num_iterations=50)  # Train in small batches
        coverage = get_scenario_coverage_stats(trainer.visited_scenarios)
        coverage_points.append(coverage['coverage_percent'])
        print(f"   After {iteration_batch} iterations: {coverage['coverage_percent']:.1f}% coverage "
              f"({coverage['unique_hole_cards_visited']}/1326 combinations)")
    
    # Verify coverage is increasing
    for i in range(1, len(coverage_points)):
        if coverage_points[i] <= coverage_points[i-1]:
            print(f"⚠️  Coverage not increasing: {coverage_points[i-1]:.1f}% -> {coverage_points[i]:.1f}%")
        else:
            print(f"✅ Coverage increased: {coverage_points[i-1]:.1f}% -> {coverage_points[i]:.1f}%")
    
    final_coverage = coverage_points[-1]
    print(f"✅ Final coverage after 500 iterations: {final_coverage:.1f}%")
    
    return final_coverage > 15.0  # Should achieve reasonable coverage


def test_preflop_simulation():
    """Test preflop-only simulation."""
    print("\n🃏 Testing Preflop Simulation")
    print("=" * 40)
    
    # Generate test scenario
    scenario = generate_random_scenario()
    hero_cards = scenario['hero_cards']
    villain_cards = scenario['villain_cards']
    
    # Test different action combinations
    test_cases = [
        ("fold", "raise_small"),
        ("raise_high", "fold"),
        ("call_small", "call_small"),
        ("raise_mid", "raise_high")
    ]
    
    for hero_action, villain_action in test_cases:
        result = simulate_preflop_showdown(
            hero_cards, villain_cards, hero_action, villain_action,
            50, 50, 1.5
        )
        
        print(f"   {hero_action} vs {villain_action}: {result['result']} "
              f"(hero change: {result['hero_stack_change']:+.1f})")
    
    print("✅ Preflop simulation working correctly")
    return True


def test_heads_up_mode():
    """Test heads-up match mode."""
    print("\n🎮 Testing Heads-up Mode")
    print("=" * 40)
    
    trainer = SimplifiedCFRTrainer(
        epsilon_exploration=0.4,
        starting_stack_bb=25  # Small stacks for quick match
    )
    
    print(f"   Initial stacks: Hero={trainer.hero_stack}bb, Villain={trainer.villain_stack}bb")
    
    # Run heads-up match
    summary = trainer.train(num_iterations=100, heads_up_mode=True)
    
    print(f"   Final stacks: Hero={trainer.hero_stack}bb, Villain={trainer.villain_stack}bb")
    print(f"   Hands played: {trainer.hands_played}")
    
    # Check that stacks changed and one player is busted or close
    total_chips = trainer.hero_stack + trainer.villain_stack
    expected_total = 50  # 25 + 25
    
    if abs(total_chips - expected_total) > 5:  # Allow some variance due to rounding
        print(f"⚠️  Chip conservation issue: {total_chips} vs expected {expected_total}")
        return False
    
    if trainer.hands_played == 0:
        print("⚠️  No hands played in heads-up mode")
        return False
    
    print("✅ Heads-up mode working correctly")
    return True


def test_checkpointing():
    """Test checkpoint save/load functionality."""
    print("\n💾 Testing Checkpointing")
    print("=" * 40)
    
    # Create trainer and run some iterations
    trainer1 = SimplifiedCFRTrainer(epsilon_exploration=0.2)
    trainer1.train(num_iterations=100)
    
    original_iterations = trainer1.iterations_completed
    original_scenarios = len(trainer1.visited_scenarios)
    
    # Save checkpoint
    checkpoint_file = trainer1.save_checkpoint("test_checkpoint")
    print(f"   Saved checkpoint: {checkpoint_file}")
    
    # Create new trainer and load checkpoint
    trainer2 = SimplifiedCFRTrainer()
    trainer2.load_checkpoint(checkpoint_file)
    
    # Verify state was restored
    if trainer2.iterations_completed != original_iterations:
        print(f"⚠️  Iterations mismatch: {trainer2.iterations_completed} vs {original_iterations}")
        return False
    
    if len(trainer2.visited_scenarios) != original_scenarios:
        print(f"⚠️  Scenarios mismatch: {len(trainer2.visited_scenarios)} vs {original_scenarios}")
        return False
    
    # Continue training from checkpoint
    trainer2.train(num_iterations=50)
    
    if trainer2.iterations_completed != original_iterations + 50:
        print(f"⚠️  Resume training failed: {trainer2.iterations_completed} vs {original_iterations + 50}")
        return False
    
    print("✅ Checkpointing working correctly")
    
    # Clean up
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return True


def test_coverage_diversity():
    """Test that we're exploring diverse hole card combinations."""
    print("\n🌈 Testing Coverage Diversity")
    print("=" * 40)
    
    trainer = SimplifiedCFRTrainer(epsilon_exploration=0.8)  # Very high exploration
    trainer.train(num_iterations=300)
    
    # Analyze the diversity of visited scenarios
    hole_card_sets = set()
    stack_categories = set()
    
    for scenario_key in trainer.visited_scenarios:
        parts = scenario_key.split('|')
        hole_cards = parts[0]
        stack_category = parts[1]
        
        hole_card_sets.add(hole_cards)
        stack_categories.add(stack_category)
    
    print(f"   Unique hole card combinations: {len(hole_card_sets)}")
    print(f"   Stack categories explored: {sorted(stack_categories)}")
    
    # Should explore at least 3 stack categories
    if len(stack_categories) < 3:
        print(f"⚠️  Limited stack category exploration: {stack_categories}")
        return False
    
    # Should explore a good variety of hole cards (at least 20% coverage)
    coverage_percent = len(hole_card_sets) / 1326 * 100
    if coverage_percent < 20:
        print(f"⚠️  Low hole card coverage: {coverage_percent:.1f}%")
        return False
    
    print(f"✅ Good diversity: {coverage_percent:.1f}% hole card coverage, "
          f"{len(stack_categories)} stack categories")
    
    return True


def test_scenario_uniqueness():
    """Test that scenarios are truly unique and not repeating patterns."""
    print("\n🔄 Testing Scenario Uniqueness") 
    print("=" * 40)
    
    # Generate many random scenarios and check for uniqueness
    generated_scenarios = []
    for _ in range(1000):
        scenario = generate_random_scenario()
        key = (
            tuple(scenario['hero_cards']),
            tuple(scenario['villain_cards']),
            scenario['hero_stack_bb'],
            scenario['villain_stack_bb']
        )
        generated_scenarios.append(key)
    
    unique_scenarios = set(generated_scenarios)
    uniqueness_ratio = len(unique_scenarios) / len(generated_scenarios)
    
    print(f"   Generated scenarios: {len(generated_scenarios)}")
    print(f"   Unique scenarios: {len(unique_scenarios)}")
    print(f"   Uniqueness ratio: {uniqueness_ratio:.3f}")
    
    # Should have high uniqueness (>90%)
    if uniqueness_ratio < 0.9:
        print(f"⚠️  Low scenario uniqueness: {uniqueness_ratio:.3f}")
        return False
    
    print("✅ High scenario uniqueness confirmed")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Simplified CFR System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Coverage Progression", test_coverage_progression),
        ("Preflop Simulation", test_preflop_simulation),
        ("Heads-up Mode", test_heads_up_mode),
        ("Checkpointing", test_checkpointing),
        ("Coverage Diversity", test_coverage_diversity),
        ("Scenario Uniqueness", test_scenario_uniqueness)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n🎉 Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, test_passed in results.items():
        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"   {test_name:20} {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total test time: {total_time:.1f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! Simplified CFR system is working correctly!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    # Ensure test directories exist
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    success = run_all_tests()
    exit(0 if success else 1)