#!/usr/bin/env python3
"""
Example usage of SequentialScenarioTrainer with multiple rollouts per visit

This script demonstrates the enhanced functionality of the SequentialScenarioTrainer
including configurable multiple rollouts per scenario visit, minimum rollout requirements,
and comprehensive rollout statistics tracking.

Key Features Demonstrated:
1. Multiple rollouts per visit with different random contexts
2. Averaged returns from multiple rollouts for regret updates 
3. Configurable minimum rollouts before convergence checking
4. Enhanced logging with rollout distribution statistics
5. Comparison between single and multiple rollout training
"""

from enhanced_cfr_trainer_v2 import SequentialScenarioTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
import time
import numpy as np


def demo_single_vs_multiple_rollouts():
    """
    Demonstrate the difference between single rollout and multiple rollout training
    """
    print("🚀 DEMO: Single vs Multiple Rollouts Training")
    print("=" * 60)
    
    # Generate small scenario set for comparison
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:3]  # Use 3 scenarios for demo
    
    # Training parameters
    iterations_per_scenario = 100
    stopping_window = 20
    regret_threshold = 0.1
    min_rollouts = 50
    
    print(f"📊 Training Configuration:")
    print(f"   Scenarios: {len(demo_scenarios)}")
    print(f"   Iterations per scenario: {iterations_per_scenario}")
    print(f"   Stopping window: {stopping_window}")
    print(f"   Regret threshold: {regret_threshold}")
    print(f"   Min rollouts before convergence: {min_rollouts}")
    
    # 1. Train with single rollout per visit (traditional approach)
    print(f"\n🎯 TRAINING WITH SINGLE ROLLOUT PER VISIT:")
    single_start = time.time()
    
    single_trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        rollouts_per_visit=1,  # Single rollout
        iterations_per_scenario=iterations_per_scenario,
        stopping_condition_window=stopping_window,
        regret_stability_threshold=regret_threshold,
        min_rollouts_before_convergence=min_rollouts
    )
    
    single_results = single_trainer.run_sequential_training()
    single_time = time.time() - single_start
    
    # 2. Train with multiple rollouts per visit (enhanced approach)
    print(f"\n🎲 TRAINING WITH MULTIPLE ROLLOUTS PER VISIT:")
    multiple_start = time.time()
    
    multiple_trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        rollouts_per_visit=3,  # Multiple rollouts
        iterations_per_scenario=iterations_per_scenario,
        stopping_condition_window=stopping_window,
        regret_stability_threshold=regret_threshold,
        min_rollouts_before_convergence=min_rollouts
    )
    
    multiple_results = multiple_trainer.run_sequential_training()
    multiple_time = time.time() - multiple_start
    
    # 3. Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'Approach':<20} {'Time (s)':<10} {'Total Rollouts':<15} {'Avg Variance':<15}")
    print("-" * 65)
    
    single_total_rollouts = sum(r.get('total_rollouts', r['iterations_completed']) for r in single_results)
    multiple_total_rollouts = sum(r.get('total_rollouts', r['iterations_completed']) for r in multiple_results)
    
    # Calculate average payoff variance
    single_variances = [r.get('payoff_variance', 0.0) for r in single_results]
    multiple_variances = [r.get('payoff_variance', 0.0) for r in multiple_results]
    
    single_avg_var = np.mean(single_variances) if single_variances else 0.0
    multiple_avg_var = np.mean(multiple_variances) if multiple_variances else 0.0
    
    print(f"{'Single Rollout':<20} {single_time:<10.2f} {single_total_rollouts:<15} {single_avg_var:<15.4f}")
    print(f"{'Multiple Rollout':<20} {multiple_time:<10.2f} {multiple_total_rollouts:<15} {multiple_avg_var:<15.4f}")
    
    print(f"\n📈 ANALYSIS:")
    print(f"   Multiple rollouts provide more robust training through:")
    print(f"   ✅ Averaged payoffs reduce variance in regret updates")
    print(f"   ✅ Multiple random contexts per scenario visit")
    print(f"   ✅ Better exploration of opponent betting patterns")
    print(f"   ✅ More stable convergence with variance tracking")
    
    return single_trainer, multiple_trainer, single_results, multiple_results


def demo_configurable_parameters():
    """
    Demonstrate various parameter configurations for the enhanced trainer
    """
    print(f"\n\n🎛️ DEMO: Configurable Parameters")
    print("=" * 60)
    
    # Generate scenarios
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:2]  # Small set for parameter demo
    
    # Test different parameter combinations
    configs = [
        {
            'name': 'Conservative',
            'rollouts_per_visit': 1,
            'iterations_per_scenario': 50,
            'stopping_condition_window': 10,
            'regret_stability_threshold': 0.2,
            'min_rollouts_before_convergence': 25
        },
        {
            'name': 'Balanced',
            'rollouts_per_visit': 2,
            'iterations_per_scenario': 100,
            'stopping_condition_window': 20,
            'regret_stability_threshold': 0.1,
            'min_rollouts_before_convergence': 50
        },
        {
            'name': 'Aggressive',
            'rollouts_per_visit': 5,
            'iterations_per_scenario': 200,
            'stopping_condition_window': 30,
            'regret_stability_threshold': 0.05,
            'min_rollouts_before_convergence': 100
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🎯 Testing {config['name']} Configuration:")
        print(f"   Rollouts per visit: {config['rollouts_per_visit']}")
        print(f"   Min rollouts before convergence: {config['min_rollouts_before_convergence']}")
        print(f"   Regret stability threshold: {config['regret_stability_threshold']}")
        
        start_time = time.time()
        
        trainer = SequentialScenarioTrainer(
            scenarios=demo_scenarios,
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        config_results = trainer.run_sequential_training()
        elapsed_time = time.time() - start_time
        
        # Calculate summary stats
        total_rollouts = sum(r.get('total_rollouts', r['iterations_completed']) for r in config_results)
        avg_variance = np.mean([r.get('payoff_variance', 0.0) for r in config_results])
        
        results[config['name']] = {
            'time': elapsed_time,
            'total_rollouts': total_rollouts,
            'avg_variance': avg_variance,
            'scenarios_completed': len(config_results)
        }
        
        print(f"   ✅ Completed in {elapsed_time:.2f}s, {total_rollouts} total rollouts")
    
    # Summary comparison
    print(f"\n📊 CONFIGURATION COMPARISON:")
    print(f"{'Config':<12} {'Time (s)':<10} {'Rollouts':<10} {'Avg Var':<10}")
    print("-" * 45)
    
    for name, stats in results.items():
        print(f"{name:<12} {stats['time']:<10.2f} {stats['total_rollouts']:<10} {stats['avg_variance']:<10.4f}")
    
    return results


def demo_advanced_stopping_conditions():
    """
    Demonstrate the enhanced stopping conditions with minimum rollouts requirement
    """
    print(f"\n\n🛑 DEMO: Advanced Stopping Conditions")
    print("=" * 60)
    
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:2]
    
    # Test with different minimum rollout requirements
    min_rollout_configs = [25, 50, 100]
    
    for min_rollouts in min_rollout_configs:
        print(f"\n🎯 Testing with min_rollouts_before_convergence = {min_rollouts}")
        
        trainer = SequentialScenarioTrainer(
            scenarios=demo_scenarios,
            rollouts_per_visit=2,
            iterations_per_scenario=150,
            stopping_condition_window=15,
            regret_stability_threshold=0.1,
            min_rollouts_before_convergence=min_rollouts
        )
        
        start_time = time.time()
        results = trainer.run_sequential_training()
        elapsed_time = time.time() - start_time
        
        # Analyze stopping reasons
        stop_reasons = [r['stop_reason'] for r in results]
        total_rollouts = sum(r.get('total_rollouts', r['iterations_completed']) for r in results)
        
        print(f"   Results: {elapsed_time:.2f}s, {total_rollouts} total rollouts")
        print(f"   Stop reasons: {', '.join(set(stop_reasons))}")


def comprehensive_example():
    """
    Comprehensive example showing all features of the enhanced SequentialScenarioTrainer
    """
    print(f"\n\n🎉 COMPREHENSIVE EXAMPLE: All Features")
    print("=" * 60)
    
    # Generate scenarios
    all_scenarios = generate_enhanced_scenarios()
    example_scenarios = all_scenarios[:5]  # Use 5 scenarios for comprehensive demo
    
    print(f"📊 Configuration for comprehensive example:")
    print(f"   Scenarios: {len(example_scenarios)}")
    print(f"   Rollouts per visit: 3 (multiple rollouts with averaged returns)")
    print(f"   Iterations per scenario: 200")
    print(f"   Stopping window: 25")
    print(f"   Regret threshold: 0.05 (strict convergence)")
    print(f"   Min rollouts before convergence: 75")
    
    # Create trainer with all enhanced features
    trainer = SequentialScenarioTrainer(
        scenarios=example_scenarios,
        rollouts_per_visit=3,                    # Multiple rollouts per visit
        iterations_per_scenario=200,             # Max iterations per scenario
        stopping_condition_window=25,            # Window for convergence check
        regret_stability_threshold=0.05,         # Strict convergence threshold
        min_rollouts_before_convergence=75       # Minimum rollouts required
    )
    
    print(f"\n🚀 Starting comprehensive sequential training...")
    start_time = time.time()
    
    # Run the complete training
    results = trainer.run_sequential_training()
    
    elapsed_time = time.time() - start_time
    
    print(f"\n🎊 COMPREHENSIVE TRAINING COMPLETE!")
    print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
    print(f"📁 Generated files:")
    print(f"   - sequential_cfr_strategies.csv")
    print(f"   - scenario_completion_report.csv")
    
    # Show comprehensive statistics
    total_rollouts = sum(r.get('total_rollouts', r['iterations_completed']) for r in results)
    total_iterations = sum(r['iterations_completed'] for r in results)
    avg_rollouts_per_scenario = total_rollouts / len(results)
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total scenarios processed: {len(results)}")
    print(f"   Total iterations: {total_iterations}")
    print(f"   Total rollouts: {total_rollouts}")
    print(f"   Average rollouts per scenario: {avg_rollouts_per_scenario:.1f}")
    
    # Show rollout variance analysis
    variances = [r.get('payoff_variance', 0.0) for r in results if r.get('payoff_variance', 0.0) > 0]
    if variances:
        print(f"   Rollout variance statistics:")
        print(f"     Mean: {np.mean(variances):.4f}")
        print(f"     Std:  {np.std(variances):.4f}")
        print(f"     Min:  {min(variances):.4f}")
        print(f"     Max:  {max(variances):.4f}")
    
    return trainer, results


if __name__ == "__main__":
    print("🚀 Enhanced SequentialScenarioTrainer Example Suite")
    print("Demonstrating multiple rollouts per visit and advanced stopping conditions")
    print("=" * 70)
    
    # Run all demonstrations
    single_trainer, multiple_trainer, single_results, multiple_results = demo_single_vs_multiple_rollouts()
    
    config_results = demo_configurable_parameters()
    
    demo_advanced_stopping_conditions()
    
    trainer, comprehensive_results = comprehensive_example()
    
    print(f"\n🎉 ALL EXAMPLES COMPLETED!")
    print(f"✅ The enhanced SequentialScenarioTrainer is ready for production use")
    print(f"📚 Key benefits:")
    print(f"   • Multiple rollouts per visit with averaged returns")
    print(f"   • Configurable minimum rollouts before convergence")
    print(f"   • Enhanced variance tracking and statistics") 
    print(f"   • Comprehensive logging with rollout distribution")
    print(f"   • Full backward compatibility maintained")