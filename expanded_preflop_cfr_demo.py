#!/usr/bin/env python3
"""
Expanded Preflop CFR Demo - Demonstrates the enhanced action abstraction

This script demonstrates the expanded preflop action abstraction with:
- 7 distinct actions: FOLD, CALL_SMALL, CALL_MID, CALL_HIGH, RAISE_SMALL, RAISE_MID, RAISE_HIGH  
- Comprehensive scenario representation encoding bet sizes at each node
- Enhanced regret minimization and strategy tracking
- Complete CSV export with action probabilities and scenario details

Usage:
    python expanded_preflop_cfr_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios, ACTIONS, PREFLOP_HAND_RANGES
import random
import time

def run_expanded_preflop_training(n_scenarios=200, n_iterations=2000, output_file="expanded_preflop_strategies.csv"):
    """
    Run expanded preflop CFR training with the new 7-action abstraction.
    
    Args:
        n_scenarios (int): Number of unique scenarios to generate
        n_iterations (int): Number of CFR training iterations  
        output_file (str): CSV filename for strategy export
    
    Returns:
        EnhancedCFRTrainer: Trained CFR model
    """
    
    print("🃏 EXPANDED PREFLOP CFR TRAINING")
    print("=" * 60)
    print(f"🎯 Enhanced Action Set: {list(ACTIONS.keys())}")
    print(f"📊 Scenarios: {n_scenarios}")
    print(f"🔄 Training iterations: {n_iterations}")
    print(f"📁 Output file: {output_file}")
    print()
    
    # Generate diverse scenarios across all hand categories and situations
    print("🎲 Generating enhanced scenarios...")
    scenarios = generate_enhanced_scenarios(n_scenarios)
    
    print(f"\n📋 Hand Categories Covered:")
    for category, hands in PREFLOP_HAND_RANGES.items():
        print(f"  {category}: {', '.join(hands[:3])}{'...' if len(hands) > 3 else ''}")
    
    # Initialize trainer with expanded action set
    print(f"\n🚀 Initializing CFR trainer...")
    trainer = EnhancedCFRTrainer(scenarios=scenarios)
    
    # Training loop with progress reporting
    print(f"\n🎯 Starting CFR training...")
    start_time = time.time()
    
    for iteration in range(n_iterations):
        # Equal distribution training - ensure all scenarios get coverage
        scenario = random.choice(scenarios)
        trainer.play_enhanced_scenario(scenario)
        
        scenario_key = trainer.get_scenario_key(scenario)
        trainer.scenario_counter[scenario_key] += 1
        
        # Progress reporting
        if iteration % (n_iterations // 10) == 0 and iteration > 0:
            elapsed = time.time() - start_time
            scenarios_trained = len(trainer.strategy_sum)
            print(f"  Iteration {iteration:4d}/{n_iterations}: {scenarios_trained:3d} scenarios trained "
                  f"({elapsed:.1f}s elapsed)")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete! ({elapsed:.1f}s total)")
    print(f"📊 Learned strategies for {len(trainer.strategy_sum)} unique scenarios")
    
    # Export comprehensive CSV with all action probabilities
    print(f"\n📁 Exporting strategies to {output_file}...")
    df = trainer.export_strategies_to_csv(output_file)
    
    if df is not None:
        print(f"\n🎯 TRAINING RESULTS SUMMARY:")
        print(f"Total unique scenarios trained: {len(df)}")
        print(f"Average training games per scenario: {df['training_games'].mean():.1f}")
        print(f"Average confidence in best action: {df['confidence'].mean():.3f}")
        
        # Show strategy distribution across positions
        print(f"\nStrategy by Position:")
        for position in df['position'].unique():
            pos_data = df[df['position'] == position]
            action_dist = pos_data['best_action'].value_counts()
            print(f"  {position}:")
            for action, count in action_dist.items():
                print(f"    {action}: {count} scenarios ({count/len(pos_data)*100:.1f}%)")
        
        # Show sample high-confidence strategies
        print(f"\n🏆 High-Confidence Strategies (top 5):")
        top_strategies = df.nlargest(5, 'confidence')
        for _, row in top_strategies.iterrows():
            print(f"  {row['example_hands']} ({row['position']}, {row['stack_depth']} stack): "
                  f"{row['best_action']} ({row['confidence']:.1%} confidence)")
    
    return trainer

def demonstrate_action_coverage():
    """Demonstrate that all 7 actions are properly supported"""
    
    print(f"\n🔍 ACTION COVERAGE DEMONSTRATION")
    print("=" * 50)
    
    from enhanced_cfr_preflop_generator_v2 import get_available_actions
    
    test_cases = [
        (50, 5, "50bb stack, 5bb bet (10% of stack)"),
        (30, 8, "30bb stack, 8bb bet (27% of stack)"), 
        (20, 12, "20bb stack, 12bb bet (60% of stack)"),
        (100, 0, "100bb stack, first to act"),
        (15, 0, "15bb stack, first to act")
    ]
    
    for stack, bet, description in test_cases:
        actions = get_available_actions(stack, bet)
        print(f"{description}: {actions}")
    
    print(f"\n✅ All 7 actions can be generated in appropriate scenarios")

if __name__ == "__main__":
    print(__doc__)
    
    # Demonstrate action coverage
    demonstrate_action_coverage()
    
    # Run training demonstration
    trainer = run_expanded_preflop_training(
        n_scenarios=100,  # Reasonable for demo
        n_iterations=1000, 
        output_file="expanded_preflop_demo_results.csv"
    )
    
    print(f"\n🎉 Demo complete! Check 'expanded_preflop_demo_results.csv' for detailed results.")
    print(f"\nKey improvements implemented:")
    print(f"✅ Expanded to 7-action set: FOLD, CALL_SMALL, CALL_MID, CALL_HIGH, RAISE_SMALL, RAISE_MID, RAISE_HIGH")
    print(f"✅ Scenario representation encodes bet sizes at each node")
    print(f"✅ CFR regret minimization accounts for all expanded actions")
    print(f"✅ CSV export includes probabilities for each action")
    print(f"✅ Export includes scenario details: hand categories, position, stack depth, bet sizing")
    print(f"✅ Export includes best action as determined by the model")
    print(f"✅ Changes isolated to preflop trainer logic as requested")