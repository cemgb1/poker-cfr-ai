#!/usr/bin/env python3
"""
Example configuration file for Enhanced CFR Trainer with advanced pruning techniques

This file demonstrates how to use the new pruning and stopping condition features
in both EnhancedCFRTrainer and SequentialScenarioTrainer classes.
"""

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer, SequentialScenarioTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios


def example_basic_pruning():
    """Example: Basic usage with default pruning settings"""
    print("🎯 Example 1: Basic Enhanced CFR Trainer with Default Pruning")
    print("=" * 60)
    
    # Create trainer with default pruning settings (backward compatible)
    trainer = EnhancedCFRTrainer()
    
    # Training loop (simplified)
    for i in range(100):
        scenario = trainer.select_balanced_scenario()
        result = trainer.play_enhanced_scenario(scenario)
        trainer.scenario_counter[result['scenario_key']] += 1
    
    # Get pruning statistics
    stats = trainer.get_pruning_statistics()
    print(f"📊 Pruning Statistics:")
    print(f"   Regret pruned: {stats['regret_pruned_count']}")
    print(f"   Strategy pruned: {stats['strategy_pruned_count']}")
    print(f"   Actions restored: {stats['actions_restored_count']}")
    print()


def example_custom_pruning():
    """Example: Custom pruning thresholds for aggressive pruning"""
    print("🎯 Example 2: Aggressive Pruning Configuration")
    print("=" * 60)
    
    # Create trainer with aggressive pruning settings
    trainer = EnhancedCFRTrainer(
        enable_pruning=True,
        regret_pruning_threshold=-100.0,  # More aggressive than default -300.0
        strategy_pruning_threshold=0.05   # Prune actions below 5% probability
    )
    
    # Simulate some regrets for demonstration
    scenario_key = "premium_pairs|BTN|medium|low"
    trainer.regret_sum[scenario_key]["fold"] = -150.0  # Below threshold
    trainer.regret_sum[scenario_key]["call_small"] = 20.0
    trainer.regret_sum[scenario_key]["raise_small"] = 15.0
    
    # Test strategy calculation with pruning
    available_actions = ["fold", "call_small", "raise_small"]
    strategy = trainer.get_strategy(scenario_key, available_actions)
    
    print(f"📈 Strategy with aggressive pruning:")
    for action, prob in strategy.items():
        print(f"   {action}: {prob:.3f}")
    
    print(f"✂️ Pruned actions: {list(trainer.pruned_actions[scenario_key])}")
    print()


def example_disabled_pruning():
    """Example: Disable pruning for comparison"""
    print("🎯 Example 3: Training with Pruning Disabled")
    print("=" * 60)
    
    # Create trainer with pruning disabled
    trainer = EnhancedCFRTrainer(enable_pruning=False)
    
    # All actions should be considered regardless of regret
    scenario_key = "trash|BB|short|high"
    trainer.regret_sum[scenario_key]["fold"] = -500.0  # Very low regret
    trainer.regret_sum[scenario_key]["call_small"] = 5.0
    trainer.regret_sum[scenario_key]["raise_high"] = 2.0
    
    available_actions = ["fold", "call_small", "raise_high"]
    strategy = trainer.get_strategy(scenario_key, available_actions)
    
    print(f"📈 Strategy without pruning (all actions considered):")
    for action, prob in strategy.items():
        print(f"   {action}: {prob:.3f}")
    print()


def example_sequential_training_basic():
    """Example: Sequential training with default stopping conditions"""
    print("🎯 Example 4: Sequential Training with Default Stopping")
    print("=" * 60)
    
    # Get a small set of scenarios for demonstration
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:5]
    
    # Create sequential trainer with default settings
    trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        rollouts_per_visit=3,
        iterations_per_scenario=200,
        min_rollouts_before_convergence=50
    )
    
    # Process one scenario for demonstration
    scenario = demo_scenarios[0]
    print(f"🎮 Processing scenario: {trainer.get_scenario_key(scenario)}")
    
    # Simulate training loop for one scenario
    scenario_key = trainer.get_scenario_key(scenario)
    for i in range(10):  # Simulate 10 iterations
        # Multiple rollouts per visit
        results = trainer.play_scenario_multiple_rollouts(scenario)
        
        # Track regret for stopping condition
        current_regret = trainer.get_current_scenario_regret(scenario_key)
        trainer.scenario_regret_history[scenario_key].append(current_regret)
        
        # Check stopping condition
        should_stop, reason = trainer.check_stopping_condition(scenario_key, i+1)
        if should_stop:
            print(f"✅ Stopped after {i+1} iterations: {reason}")
            break
    
    # Get stopping statistics
    stats = trainer.get_stopping_statistics()
    print(f"📊 Stopping Statistics: {stats}")
    print()


def example_sequential_training_custom_stopping():
    """Example: Sequential training with custom stopping conditions"""
    print("🎯 Example 5: Custom Stopping Conditions Configuration")
    print("=" * 60)
    
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:3]
    
    # Create trainer with custom stopping conditions
    trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        rollouts_per_visit=2,
        iterations_per_scenario=300,
        enable_min_rollouts_stopping=True,
        enable_regret_stability_stopping=True,
        enable_max_iterations_stopping=False,  # Disable max iterations stopping
        stopping_condition_mode='strict',       # All enabled conditions must be met
        regret_stability_threshold=0.02,        # Stricter stability requirement
        min_rollouts_before_convergence=100
    )
    
    print(f"⚙️ Stopping Configuration:")
    print(f"   Min rollouts stopping: {trainer.enable_min_rollouts_stopping}")
    print(f"   Regret stability stopping: {trainer.enable_regret_stability_stopping}")
    print(f"   Max iterations stopping: {trainer.enable_max_iterations_stopping}")
    print(f"   Stopping mode: {trainer.stopping_condition_mode}")
    print(f"   Stability threshold: {trainer.regret_stability_threshold}")
    
    # Demonstrate runtime reconfiguration
    print(f"\n🔄 Reconfiguring stopping conditions at runtime...")
    trainer.reconfigure_stopping_conditions(
        stopping_condition_mode='flexible',
        regret_stability_threshold=0.1
    )
    print()


def example_combined_features():
    """Example: Using both pruning and custom stopping conditions together"""
    print("🎯 Example 6: Combined Pruning and Stopping Features")
    print("=" * 60)
    
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:5]
    
    # Create trainer with both pruning and stopping customizations
    trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        rollouts_per_visit=2,
        iterations_per_scenario=150,
        
        # Stopping conditions
        enable_min_rollouts_stopping=True,
        enable_regret_stability_stopping=True,
        enable_max_iterations_stopping=True,
        stopping_condition_mode='flexible',
        min_rollouts_before_convergence=30,
        regret_stability_threshold=0.08,
        
        # Pruning settings
        enable_pruning=True,
        regret_pruning_threshold=-200.0,
        strategy_pruning_threshold=0.02
    )
    
    print(f"🎛️ Combined Configuration:")
    print(f"   Pruning enabled: {trainer.enable_pruning}")
    print(f"   Regret threshold: {trainer.regret_pruning_threshold}")
    print(f"   Strategy threshold: {trainer.strategy_pruning_threshold}")
    print(f"   Stopping mode: {trainer.stopping_condition_mode}")
    print(f"   Min rollouts: {trainer.min_rollouts_before_convergence}")
    
    # Export strategies with pruning
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        df = trainer.export_strategies_to_csv(tmp.name)
        if df is not None:
            print(f"\n📊 Strategy Export with Pruning:")
            print(f"   Total scenarios exported: {len(df)}")
            if 'actions_pruned_count' in df.columns:
                total_pruned = df['actions_pruned_count'].sum()
                print(f"   Total actions pruned in export: {total_pruned}")
    print()


def example_benchmarking_setup():
    """Example: Setting up benchmarking between pruned and non-pruned training"""
    print("🎯 Example 7: Benchmarking Setup")
    print("=" * 60)
    
    all_scenarios = generate_enhanced_scenarios()
    test_scenarios = all_scenarios[:10]
    
    # Trainer with pruning
    trainer_with_pruning = EnhancedCFRTrainer(
        scenarios=test_scenarios,
        enable_pruning=True,
        regret_pruning_threshold=-150.0,
        strategy_pruning_threshold=0.03
    )
    
    # Trainer without pruning
    trainer_without_pruning = EnhancedCFRTrainer(
        scenarios=test_scenarios,
        enable_pruning=False
    )
    
    print(f"🏁 Benchmark Setup Complete:")
    print(f"   Scenarios for testing: {len(test_scenarios)}")
    print(f"   Trainer 1: Pruning enabled")
    print(f"   Trainer 2: Pruning disabled")
    print(f"   Ready for performance comparison!")
    print()


def main():
    """Run all examples"""
    print("🚀 Enhanced CFR Trainer Configuration Examples")
    print("=" * 80)
    print()
    
    # Run all examples
    example_basic_pruning()
    example_custom_pruning()
    example_disabled_pruning()
    example_sequential_training_basic()
    example_sequential_training_custom_stopping()
    example_combined_features()
    example_benchmarking_setup()
    
    print("✅ All examples completed!")
    print("\n💡 Key Takeaways:")
    print("   • Pruning is enabled by default for backward compatibility")
    print("   • Regret-based pruning removes poor actions during strategy calculation")
    print("   • Strategy pruning removes low-probability actions from exports")
    print("   • Action space pruning filters contextually irrelevant actions")
    print("   • Sequential trainer supports flexible, strict, and custom stopping modes")
    print("   • All features can be reconfigured at runtime")
    print("   • Comprehensive statistics are available for performance analysis")


if __name__ == '__main__':
    main()