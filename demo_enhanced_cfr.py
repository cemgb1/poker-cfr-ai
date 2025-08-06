#!/usr/bin/env python3
"""
Demo script showing the enhanced CFR training improvements:
1. Balanced hand category sampling vs random
2. Enhanced performance metrics and scenario space analysis
3. Deeper training (200k iterations vs 50k)
"""

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
import random
from collections import defaultdict

def demo_balanced_vs_random_sampling():
    """Demonstrate balanced sampling vs random sampling"""
    print("🎯 DEMO: Balanced vs Random Sampling")
    print("=" * 50)
    
    scenarios = generate_enhanced_scenarios()
    
    # Test random sampling (old way)
    print("\n📊 OLD METHOD - Random sampling (500 iterations):")
    random_visits = defaultdict(int)
    for _ in range(500):
        scenario = random.choice(scenarios)
        random_visits[scenario['hand_category']] += 1
    
    for category, visits in sorted(random_visits.items()):
        percentage = (visits / 500) * 100
        print(f"   {category:18s}: {visits:3d} visits ({percentage:5.1f}%)")
    
    # Calculate balance quality for random
    random_percentages = [(visits / 500) * 100 for visits in random_visits.values()]
    random_std = __import__('statistics').stdev(random_percentages)
    
    # Test balanced sampling (new way)
    print(f"\n🎯 NEW METHOD - Balanced sampling (500 iterations):")
    trainer = EnhancedCFRTrainer(scenarios=scenarios)
    for _ in range(500):
        trainer.select_balanced_scenario()
    
    for category, visits in sorted(trainer.hand_category_visits.items()):
        percentage = (visits / 500) * 100
        print(f"   {category:18s}: {visits:3d} visits ({percentage:5.1f}%)")
    
    # Calculate balance quality for balanced
    balanced_percentages = [(visits / 500) * 100 for visits in trainer.hand_category_visits.values()]
    balanced_std = __import__('statistics').stdev(balanced_percentages)
    
    print(f"\n📊 BALANCE QUALITY COMPARISON:")
    print(f"   Random sampling std dev:   {random_std:.2f}% (higher = more uneven)")
    print(f"   Balanced sampling std dev: {balanced_std:.2f}% (lower = more even)")
    print(f"   Improvement factor:        {random_std/balanced_std:.1f}x more balanced")

def demo_enhanced_metrics():
    """Demonstrate enhanced performance metrics"""
    print(f"\n\n🧠 DEMO: Enhanced Performance Metrics")
    print("=" * 50)
    
    scenarios = generate_enhanced_scenarios()
    trainer = EnhancedCFRTrainer(scenarios=scenarios)
    trainer.start_performance_tracking()
    
    # Run a small training
    for i in range(100):
        scenario = trainer.select_balanced_scenario()
        trainer.play_enhanced_scenario(scenario)
        trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1
    
    # Get enhanced metrics
    metrics = trainer.record_iteration_metrics(99)
    
    print(f"\n📊 ENHANCED METRICS AVAILABLE:")
    print(f"   🎯 Total possible scenarios: {metrics['total_possible_scenarios']:,}")
    print(f"   📈 Scenario coverage: {metrics['scenario_coverage_percentage']}%")
    print(f"   🔍 Unique scenarios visited: {metrics['unique_scenarios_visited']}")
    print(f"   ⏱️  Training efficiency: {metrics['average_regret']:.6f} avg regret")
    
    print(f"\n🎲 HAND CATEGORY COVERAGE TRACKING:")
    for category, visits in sorted(trainer.hand_category_visits.items()):
        if visits > 0:
            print(f"   {category:18s}: {visits:2d} visits")
    
    print(f"\n📈 SCENARIO VISIT DISTRIBUTION:")
    coverage_histogram = trainer.get_scenario_coverage_histogram()
    for bracket, count in coverage_histogram.items():
        print(f"   {bracket:8s} visits: {count:2d} scenarios")

def demo_training_improvements():
    """Show training configuration improvements"""
    print(f"\n\n⚡ DEMO: Training Improvements")
    print("=" * 50)
    
    print(f"📊 TRAINING CONFIGURATION CHANGES:")
    print(f"   Old approach:       Manual n_scenarios setting (e.g. 1000)")
    print(f"   New approach:       All possible combinations (330 scenarios)")
    print(f"   Scenario variables: Removed bet_size_category from keys")
    print(f"   Opponent betting:   Static → Dynamic during simulation")
    print(f"   Action mapping:     Fixed → Based on actual bet vs stack ratio")
    print(f"   Scenario space:     11×2×5×5×3 = 1,650 → 11×2×5×3 = 330")
    print(f"   Benefits:           Smaller space, more realistic, better generalization")

if __name__ == "__main__":
    print("🚀 ENHANCED CFR TRAINING SYSTEM DEMO")
    print("=" * 60)
    print("Demonstrating improvements for balanced hand category coverage")
    print("and deeper learning with comprehensive performance tracking.")
    
    demo_balanced_vs_random_sampling()
    demo_enhanced_metrics()
    demo_training_improvements()
    
    print(f"\n\n✅ DEMO COMPLETE")
    print(f"🎯 Key Benefits of Refactoring:")
    print(f"   • Removed bet_size_category from scenario keys (more realistic)")
    print(f"   • Generate all 330 possible scenario combinations automatically")
    print(f"   • Dynamic opponent betting during simulation (robust training)")
    print(f"   • Action mapping based on actual bet sizes vs stack ratios")
    print(f"   • Smaller, more focused scenario space (330 vs 1,650)")
    print(f"   • Better generalization to various opponent bet distributions")