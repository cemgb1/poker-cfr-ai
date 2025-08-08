#!/usr/bin/env python3
"""
Quick test of tournament survival penalty in training to show realistic behavior change.
"""

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios


def test_training_with_different_penalties():
    """Test short training runs with different penalty settings"""
    print("🎯 Testing Training Behavior with Different Tournament Penalties")
    print("=" * 70)
    
    # Generate small scenario set focused on short stacks
    all_scenarios = generate_enhanced_scenarios()
    short_stack_scenarios = [s for s in all_scenarios if s['stack_category'] in ['ultra_short', 'short']][:10]
    
    penalty_configs = [
        (0.4, "Aggressive"),
        (1.0, "Original (Harsh)")
    ]
    
    print(f"Training on {len(short_stack_scenarios)} short-stack scenarios...")
    print(f"Each trainer will run 100 iterations\n")
    
    for penalty, name in penalty_configs:
        print(f"🔧 {name} Training (penalty={penalty})")
        print("-" * 40)
        
        trainer = EnhancedCFRTrainer(
            scenarios=short_stack_scenarios,
            tournament_survival_penalty=penalty
        )
        
        # Run short training
        action_counts = {'fold': 0, 'call': 0, 'raise': 0}
        payoff_results = []
        
        for i in range(100):
            scenario = trainer.select_balanced_scenario()
            result = trainer.play_enhanced_scenario(scenario)
            
            # Count action types
            action = result['hero_action']
            if 'fold' in action:
                action_counts['fold'] += 1
            elif 'call' in action:
                action_counts['call'] += 1
            elif 'raise' in action:
                action_counts['raise'] += 1
            
            payoff_results.append(result['payoff'])
        
        # Show results
        total_actions = sum(action_counts.values())
        fold_percent = (action_counts['fold'] / total_actions) * 100
        call_percent = (action_counts['call'] / total_actions) * 100
        raise_percent = (action_counts['raise'] / total_actions) * 100
        avg_payoff = sum(payoff_results) / len(payoff_results)
        
        print(f"   Fold rate: {fold_percent:.1f}%")
        print(f"   Call rate: {call_percent:.1f}%") 
        print(f"   Raise rate: {raise_percent:.1f}%")
        print(f"   Avg payoff: {avg_payoff:.3f}")
        print()
    
    print("💡 Expected behavior:")
    print("   • Aggressive: Lower fold rate, more willing to take risks")
    print("   • Original: Higher fold rate, overly conservative on short stacks")


if __name__ == "__main__":
    test_training_with_different_penalties()