#!/usr/bin/env python3
"""
Demo script showing the performance metrics tracking feature.

This script demonstrates how to use the new performance metrics tracking
functionality alongside the existing CFR training system.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
import random

def demo_performance_tracking(n_scenarios=50, n_iterations=5000, metrics_interval=100):
    """
    Demonstrate performance metrics tracking during CFR training.
    
    This shows how the new functionality works:
    1. Creates model_performance.csv with iteration metrics
    2. Preserves existing strategy CSV output
    3. Tracks convergence and learning dynamics
    """
    
    print("🃏 CFR TRAINING WITH PERFORMANCE METRICS DEMO")
    print("=" * 70)
    print(f"📊 Training parameters:")
    print(f"   Scenarios: {n_scenarios}")
    print(f"   Iterations: {n_iterations}")
    print(f"   Metrics tracked every: {metrics_interval} iterations")
    print(f"   Outputs: model_performance.csv + enhanced_cfr_results.csv")
    print()
    
    # Generate scenarios
    print("🎲 Generating scenarios...")
    scenarios = generate_enhanced_scenarios(n_scenarios)
    
    # Initialize trainer
    print("🚀 Initializing CFR trainer...")
    trainer = EnhancedCFRTrainer(scenarios=scenarios)
    
    # Start performance tracking
    print("📊 Starting performance tracking...")
    trainer.start_performance_tracking()
    
    # Training loop with integrated performance tracking
    print(f"\n🎯 Training with performance metrics...")
    print("Iter | Scenarios | Avg Regret | Max Regret | Time/Iter")
    print("-" * 55)
    
    for iteration in range(n_iterations):
        # Standard CFR training step
        scenario = random.choice(scenarios)
        trainer.play_enhanced_scenario(scenario)
        trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1
        
        # Record and display metrics at intervals
        if iteration % metrics_interval == 0:
            metrics = trainer.record_iteration_metrics(iteration)
            if iteration > 0:  # Skip first iteration for timing
                print(f"{iteration:4d} | {metrics['unique_scenarios_visited']:9d} | "
                      f"{metrics['average_regret']:10.6f} | "
                      f"{metrics['max_regret']:10.6f} | "
                      f"{metrics['time_per_iteration']:8.4f}s")
    
    # Record final metrics
    final_metrics = trainer.record_iteration_metrics(n_iterations - 1)
    print(f"{n_iterations-1:4d} | {final_metrics['unique_scenarios_visited']:9d} | "
          f"{final_metrics['average_regret']:10.6f} | "
          f"{final_metrics['max_regret']:10.6f} | "
          f"{final_metrics['time_per_iteration']:8.4f}s")
    
    print(f"\n✅ Training complete!")
    
    # Export both types of CSV files
    print(f"\n📁 Exporting results...")
    
    # Export performance metrics (NEW)
    perf_df = trainer.export_performance_metrics("model_performance.csv")
    
    # Export strategies (EXISTING - unchanged)
    strategy_df = trainer.export_strategies_to_csv("enhanced_cfr_results.csv")
    
    # Show what we've created
    print(f"\n🎯 OUTPUT FILES CREATED:")
    print(f"✅ model_performance.csv - Performance metrics for analyzing convergence")
    print(f"   - Iteration-by-iteration regret, timing, and scenario coverage data")
    print(f"   - Use for convergence analysis and training dynamics")
    print(f"✅ enhanced_cfr_results.csv - Strategy results (existing functionality)")
    print(f"   - Best actions and probabilities for each scenario")
    print(f"   - Use for actual poker strategy lookup")
    
    # Show sample performance data
    if perf_df is not None and len(perf_df) > 1:
        print(f"\n📊 SAMPLE PERFORMANCE METRICS:")
        print(f"Initial avg regret: {perf_df['average_regret'].iloc[0]:.6f}")
        print(f"Final avg regret: {perf_df['average_regret'].iloc[-1]:.6f}")
        print(f"Total training time: {perf_df['total_elapsed_time'].iloc[-1]:.2f}s")
        print(f"Scenarios visited: {perf_df['unique_scenarios_visited'].iloc[-1]}")
        
        # Show convergence
        if len(perf_df) >= 3:
            early_regret = perf_df['average_regret'].iloc[1]  # Skip iteration 0
            final_regret = perf_df['average_regret'].iloc[-1]
            if early_regret > 0:
                change_pct = (final_regret - early_regret) / early_regret * 100
                print(f"Regret change: {change_pct:+.1f}% (negative = improvement)")
    
    return trainer, perf_df, strategy_df

if __name__ == "__main__":
    print("Run with: python demo_with_performance_metrics.py")
    print("This will create model_performance.csv and enhanced_cfr_results.csv")
    print()
    
    # Run the demo
    trainer, perf_df, strategy_df = demo_performance_tracking()
    
    print(f"\n🎉 Demo complete! Check the CSV files to see the results.")