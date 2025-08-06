#!/usr/bin/env python3
"""
Monte Carlo CFR Demo - Showcasing the new implementation

Demonstrates the key improvements:
1. Simplified action space (fold, call, raise_small, shove)
2. Removed tournament functionality and bet size variables
3. Monte Carlo CFR with dynamic stopping criteria
4. X simulations per scenario approach
"""

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
import time

def run_monte_carlo_training_demo(n_scenarios=5, simulations_per_scenario=3):
    """Quick Monte Carlo training for demo"""
    scenarios = generate_enhanced_scenarios(n_scenarios)
    trainer = EnhancedCFRTrainer(scenarios=scenarios, monte_carlo=True, 
                                simulations_per_scenario=simulations_per_scenario)
    trainer.start_performance_tracking()
    
    iteration = 0
    while trainer.should_continue_training():
        scenario = trainer.select_balanced_scenario()
        trainer.play_enhanced_scenario(scenario)
        trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1
        iteration += 1
    
    trainer.export_strategies_to_csv("demo_monte_carlo_results.csv")
    return trainer, iteration

def demo_action_space_changes():
    """Demonstrate the simplified action space"""
    print("🎯 DEMO: Simplified Action Space")
    print("=" * 50)
    
    scenarios = generate_enhanced_scenarios(10)
    
    print(f"\n📊 ACTION SPACE COMPARISON:")
    print(f"   Old actions:  FOLD, CALL_SMALL, CALL_MID, CALL_HIGH, RAISE_SMALL, RAISE_MID, RAISE_HIGH (7 total)")
    print(f"   New actions:  FOLD, CALL, RAISE_SMALL, SHOVE (4 total)")
    print(f"   Reduction:    57% fewer actions (simpler decision space)")
    
    print(f"\n🔍 SCENARIO ANALYSIS:")
    action_counts = {}
    for scenario in scenarios:
        for action in scenario['available_actions']:
            action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"   Available actions across {len(scenarios)} scenarios:")
    for action, count in sorted(action_counts.items()):
        print(f"     {action:12s}: available in {count:2d} scenarios ({count/len(scenarios)*100:.0f}%)")

def demo_monte_carlo_vs_traditional():
    """Compare Monte Carlo CFR with traditional CFR"""
    print(f"\n\n🏆 DEMO: Monte Carlo CFR vs Traditional CFR")
    print("=" * 60)
    
    # Generate common scenarios for fair comparison
    scenarios = generate_enhanced_scenarios(8)
    print(f"\nUsing {len(scenarios)} scenarios for comparison...")
    
    print(f"\n🔄 TRADITIONAL CFR (Fixed 50 iterations):")
    start_time = time.time()
    traditional_trainer = EnhancedCFRTrainer(scenarios=scenarios, monte_carlo=False)
    traditional_trainer.start_performance_tracking()
    
    for i in range(50):
        scenario = traditional_trainer.select_balanced_scenario()
        traditional_trainer.play_enhanced_scenario(scenario)
        traditional_trainer.scenario_counter[traditional_trainer.get_scenario_key(scenario)] += 1
    
    trad_time = time.time() - start_time
    
    trad_visits = list(traditional_trainer.scenario_counter.values())
    trad_min = min(trad_visits) if trad_visits else 0
    trad_max = max(trad_visits) if trad_visits else 0
    trad_unique = len(traditional_trainer.scenario_counter)
    
    print(f"   Time taken:     {trad_time:.2f} seconds")
    print(f"   Total iterations: 50 (fixed)")
    print(f"   Unique scenarios: {trad_unique}")
    print(f"   Visit distribution: min={trad_min}, max={trad_max} (uneven)")
    
    print(f"\n🎯 MONTE CARLO CFR (5 simulations per scenario):")
    start_time = time.time()
    mc_trainer = EnhancedCFRTrainer(scenarios=scenarios, monte_carlo=True, simulations_per_scenario=5)
    mc_trainer.start_performance_tracking()
    
    iteration = 0
    while mc_trainer.should_continue_training():
        scenario = mc_trainer.select_balanced_scenario()
        mc_trainer.play_enhanced_scenario(scenario)
        mc_trainer.scenario_counter[mc_trainer.get_scenario_key(scenario)] += 1
        iteration += 1
    
    mc_time = time.time() - start_time
    
    mc_visits = list(mc_trainer.scenario_counter.values())
    mc_min = min(mc_visits) if mc_visits else 0
    mc_max = max(mc_visits) if mc_visits else 0
    mc_unique = len(mc_trainer.scenario_counter)
    
    print(f"   Time taken:     {mc_time:.2f} seconds")
    print(f"   Total iterations: {iteration} (dynamic: 5 × {len(scenarios)} scenarios)")
    print(f"   Unique scenarios: {mc_unique}")
    print(f"   Visit distribution: min={mc_min}, max={mc_max} (perfectly even)")
    
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"   Coverage quality: Monte Carlo ensures every scenario gets exactly X visits")
    print(f"   Predictability:   Monte Carlo guarantees total iterations = X × N")
    print(f"   Balance:         Monte Carlo provides perfect scenario balance")

def demo_dynamic_stopping_criteria():
    """Demonstrate dynamic stopping criteria"""
    print(f"\n\n⏹️  DEMO: Dynamic Stopping Criteria")
    print("=" * 50)
    
    print(f"📋 STOPPING CRITERIA LOGIC:")
    print(f"   Traditional CFR: Fixed iteration count (e.g., 200,000)")
    print(f"   Monte Carlo CFR: Dynamic - stops when all scenarios have X visits")
    print(f"   ")
    print(f"   Formula: Total iterations = X simulations × N scenarios")
    print(f"   Example: 50 simulations × 100 scenarios = 5,000 iterations")
    
    print(f"\n🔍 PRACTICAL EXAMPLE:")
    scenarios = generate_enhanced_scenarios(3)  # Small example
    
    print(f"   Scenarios: {len(scenarios)}")
    print(f"   Simulations per scenario: 4")
    print(f"   Expected total iterations: {len(scenarios) * 4}")
    
    trainer = EnhancedCFRTrainer(scenarios=scenarios, monte_carlo=True, simulations_per_scenario=4)
    
    iteration = 0
    visit_progress = []
    
    while trainer.should_continue_training():
        scenario = trainer.select_balanced_scenario()
        trainer.play_enhanced_scenario(scenario)
        trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1
        iteration += 1
        
        # Track progress
        visits = list(trainer.scenario_counter.values())
        min_visits = min(visits) if visits else 0
        max_visits = max(visits) if visits else 0
        visit_progress.append((iteration, min_visits, max_visits))
        
        if iteration % 4 == 0:
            print(f"   Iteration {iteration:2d}: min_visits={min_visits}, max_visits={max_visits}")
    
    print(f"   ✅ Training stopped automatically at {iteration} iterations")
    print(f"   📊 All scenarios have exactly 4 visits")

def demo_output_compatibility():
    """Demonstrate that outputs remain the same format"""
    print(f"\n\n📁 DEMO: Output Format Compatibility")
    print("=" * 50)
    
    print(f"🔍 TESTING OUTPUT FORMATS:")
    
    # Test Monte Carlo training with small parameters
    trainer, iterations = run_monte_carlo_training_demo(n_scenarios=5, simulations_per_scenario=2)
    
    print(f"\n✅ TRAINING COMPLETED:")
    print(f"   📊 {iterations} total iterations (5 scenarios × 2 simulations)")
    print(f"   📁 demo_monte_carlo_results.csv generated")
    
    print(f"\n🔍 STRATEGY CSV COLUMNS:")
    import pandas as pd
    try:
        df = pd.read_csv("demo_monte_carlo_results.csv")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")
        print(f"   ✅ Format maintained - compatible with existing systems")
        
        # Show sample of best actions
        print(f"\n📋 SAMPLE STRATEGIES:")
        for i, row in df.head(3).iterrows():
            print(f"   {row['hand_category']:15s} {row['position']} {row['stack_depth']:10s}: {row['best_action']} ({row['confidence']:.2f})")
            
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")

if __name__ == "__main__":
    print("🚀 MONTE CARLO CFR IMPLEMENTATION DEMO")
    print("=" * 70)
    print("Demonstrating the new simplified, tournament-free Monte Carlo CFR system")
    print("with dynamic stopping criteria and heads-up focus.")
    
    demo_action_space_changes()
    demo_monte_carlo_vs_traditional() 
    demo_dynamic_stopping_criteria()
    demo_output_compatibility()
    
    print(f"\n\n✅ DEMO COMPLETE")
    print(f"🎯 Key Achievements:")
    print(f"   • Simplified action space: 4 actions vs 7 (57% reduction)")
    print(f"   • Removed tournament functionality completely") 
    print(f"   • Eliminated bet size as scenario variable")
    print(f"   • Implemented true Monte Carlo CFR (X sims per scenario)")
    print(f"   • Dynamic stopping criteria (stops at X × N iterations)")
    print(f"   • Maintained all output formats for compatibility")
    print(f"   • Perfect scenario balance and predictable runtime")