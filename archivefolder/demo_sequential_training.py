#!/usr/bin/env python3
"""
Demo script showcasing the Sequential Scenario Training approach

This demo compares the new sequential approach with the original dynamic approach
and shows the enhanced logging capabilities with time estimates.
"""

from enhanced_cfr_trainer_v2 import SequentialScenarioTrainer, EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
import time


def demo_sequential_vs_dynamic_training():
    """Compare Sequential vs Dynamic training approaches"""
    print("🚀 DEMO: Sequential vs Dynamic Training Comparison")
    print("=" * 60)
    
    # Generate scenario set for comparison  
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:8]  # Use 8 scenarios for meaningful comparison
    
    print(f"📊 Using {len(demo_scenarios)} scenarios for comparison")
    
    # 1. Run original dynamic approach
    print(f"\n🔄 RUNNING ORIGINAL DYNAMIC APPROACH...")
    dynamic_start = time.time()
    
    dynamic_trainer = EnhancedCFRTrainer(scenarios=demo_scenarios)
    dynamic_trainer.start_performance_tracking()
    
    total_iterations = 400  # Total iterations to distribute across scenarios
    for i in range(total_iterations):
        scenario = dynamic_trainer.select_balanced_scenario()
        dynamic_trainer.play_enhanced_scenario(scenario)
        dynamic_trainer.scenario_counter[dynamic_trainer.get_scenario_key(scenario)] += 1
    
    dynamic_end = time.time()
    dynamic_time = dynamic_end - dynamic_start
    
    print(f"✅ Dynamic training completed in {dynamic_time:.2f}s")
    print(f"📊 Scenarios visited: {len(dynamic_trainer.scenario_counter)}")
    print(f"🎯 Total strategy entries: {len(dynamic_trainer.strategy_sum)}")
    
    # Show scenario visit distribution
    visits = list(dynamic_trainer.scenario_counter.values())
    print(f"📈 Visit distribution: min={min(visits)}, max={max(visits)}, avg={sum(visits)/len(visits):.1f}")
    
    # 2. Run new sequential approach
    print(f"\n🎯 RUNNING NEW SEQUENTIAL APPROACH...")
    sequential_start = time.time()
    
    sequential_trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        iterations_per_scenario=50,  # Average of total_iterations / num_scenarios
        stopping_condition_window=15,
        regret_stability_threshold=0.1
    )
    
    results = sequential_trainer.run_sequential_training()
    
    sequential_end = time.time()
    sequential_time = sequential_end - sequential_start
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'Approach':<15} {'Time (s)':<10} {'Scenarios':<12} {'Strategies':<12} {'Total Iter':<12}")
    print("-" * 60)
    print(f"{'Dynamic':<15} {dynamic_time:<10.2f} {len(dynamic_trainer.scenario_counter):<12} {len(dynamic_trainer.strategy_sum):<12} {total_iterations:<12}")
    
    sequential_total_iter = sum(r['iterations_completed'] for r in results)
    print(f"{'Sequential':<15} {sequential_time:<10.2f} {len(results):<12} {len(sequential_trainer.strategy_sum):<12} {sequential_total_iter:<12}")
    
    # 3. Analysis
    print(f"\n📈 ANALYSIS:")
    print(f"Sequential approach ensures:")
    print(f"  ✅ All scenarios processed completely")
    print(f"  ✅ Deterministic execution order") 
    print(f"  ✅ Stopping condition based convergence")
    print(f"  ✅ Time estimation and progress tracking")
    
    return dynamic_trainer, sequential_trainer, results


def demo_enhanced_logging_features():
    """Demonstrate enhanced logging and time estimation"""
    print(f"\n\n🚀 DEMO: Enhanced Logging and Time Estimation")
    print("=" * 60)
    
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:5]  # Small set for demo
    
    trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        iterations_per_scenario=30,
        stopping_condition_window=10,
        regret_stability_threshold=0.2
    )
    
    print("🎯 Features demonstrated:")
    print("  📊 Real-time progress tracking")
    print("  ⏱️  Time estimation for remaining scenarios") 
    print("  🔢 Total iterations remaining calculation")
    print("  📈 Stopping condition monitoring")
    print("  📋 Detailed completion reports")
    
    # Run training with enhanced logging
    results = trainer.run_sequential_training()
    
    print(f"\n📊 Enhanced Logging Summary:")
    print(f"  📁 Exported files:")
    print(f"    - sequential_cfr_strategies.csv (learned strategies)")
    print(f"    - scenario_completion_report.csv (detailed completion data)")
    
    return trainer, results


def demo_stopping_condition_analysis():
    """Demonstrate stopping condition behavior"""
    print(f"\n\n🚀 DEMO: Stopping Condition Analysis") 
    print("=" * 60)
    
    all_scenarios = generate_enhanced_scenarios()
    # Use scenarios from different hand categories to show variation
    demo_scenarios = [
        all_scenarios[0],   # Premium pairs
        all_scenarios[30],  # Medium pairs
        all_scenarios[60],  # Small pairs 
        all_scenarios[90],  # Premium aces
        all_scenarios[300]  # Trash hands
    ]
    
    trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        iterations_per_scenario=80,
        stopping_condition_window=20,
        regret_stability_threshold=0.05  # Stricter threshold
    )
    
    print("🎯 Stopping condition parameters:")
    print(f"  📊 Window size: {trainer.stopping_condition_window} iterations")
    print(f"  📈 Stability threshold: {trainer.regret_stability_threshold}")
    print(f"  🎲 Different hand categories for varied convergence")
    
    results = trainer.run_sequential_training()
    
    print(f"\n📈 Stopping Condition Results:")
    print(f"{'Scenario':<30} {'Iterations':<12} {'Stop Reason':<20} {'Final Regret':<15}")
    print("-" * 80)
    
    for result in results:
        scenario_parts = result['scenario_key'].split('|')
        hand_cat = scenario_parts[0][:15] if len(scenario_parts) > 0 else 'unknown'
        print(f"{hand_cat:<30} {result['iterations_completed']:<12} {result['stop_reason']:<20} {result['final_regret']:<15.6f}")
    
    return trainer, results


if __name__ == "__main__":
    print("🚀 Sequential CFR Training Demo Suite")
    print("Demonstrating the new sequential training approach with stopping conditions")
    print("=" * 70)
    
    # Run all demos
    dynamic_trainer, sequential_trainer, seq_results = demo_sequential_vs_dynamic_training()
    
    enhanced_trainer, enhanced_results = demo_enhanced_logging_features()
    
    stopping_trainer, stopping_results = demo_stopping_condition_analysis()
    
    print(f"\n🎉 ALL DEMOS COMPLETED!")
    print(f"📁 Check the generated CSV files for detailed results:")
    print(f"   - sequential_cfr_strategies.csv")
    print(f"   - scenario_completion_report.csv") 
    print(f"")
    print(f"🚀 The sequential training approach is ready for production use!")