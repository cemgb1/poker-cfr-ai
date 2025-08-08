#!/usr/bin/env python3
"""
Demonstration of Tournament Survival Penalty Parameter Effects

This script shows how different tournament_survival_penalty values affect
the CFR trainer's decision-making, particularly on short stacks.
"""

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios


def demonstrate_penalty_effects():
    """Demonstrate how penalty parameter affects bust penalties"""
    print("🎰 Tournament Survival Penalty Demonstration")
    print("=" * 60)
    
    # Generate a small set of scenarios for demo
    scenarios = generate_enhanced_scenarios()[:5]
    
    # Test different penalty settings
    penalty_configs = [
        (0.3, "Very Aggressive", "Encourages high risk-taking, minimal bust penalty"),
        (0.6, "Balanced (Default)", "Moderate penalty, realistic tournament play"),
        (1.0, "Original (Too Harsh)", "High penalty, causes over-folding")
    ]
    
    print(f"\n📊 Bust Penalty Comparison (Short Stack = 10bb):")
    print(f"{'Setting':<20} {'Penalty':<10} {'Description'}")
    print("-" * 60)
    
    for penalty_factor, name, description in penalty_configs:
        trainer = EnhancedCFRTrainer(
            scenarios=scenarios,
            tournament_survival_penalty=penalty_factor
        )
        
        # Simulate short stack bust
        bust_penalty = trainer.apply_stack_adjustments(
            base_payoff=0.0,
            stack_before=10,  # Short stack
            stack_after=0,    # Busted
            busted=True
        )
        
        print(f"{name:<20} {bust_penalty:<10.1f} {description}")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Lower penalty = More likely to call/shove with reasonable hands")
    print(f"   • Higher penalty = More likely to fold everything except nuts")
    print(f"   • Default 0.6 = 40% less punishing than original harsh penalties")
    print(f"   • Recommended range: 0.3 (very aggressive) to 0.8 (conservative)")
    
    print(f"\n🔧 Usage Examples:")
    print(f"   # For aggressive short-stack play:")
    print(f"   trainer = EnhancedCFRTrainer(tournament_survival_penalty=0.4)")
    print(f"   ")
    print(f"   # For balanced tournament strategy (default):")
    print(f"   trainer = EnhancedCFRTrainer(tournament_survival_penalty=0.6)")
    print(f"   ")
    print(f"   # For conservative chip preservation:")
    print(f"   trainer = EnhancedCFRTrainer(tournament_survival_penalty=0.8)")


def test_penalty_in_action():
    """Test actual payoff differences in realistic scenarios"""
    print(f"\n🎯 Realistic Scenario Testing")
    print("-" * 40)
    
    scenarios = generate_enhanced_scenarios()[:3]
    
    # Create aggressive and conservative trainers
    aggressive = EnhancedCFRTrainer(scenarios=scenarios, tournament_survival_penalty=0.4)
    conservative = EnhancedCFRTrainer(scenarios=scenarios, tournament_survival_penalty=0.8)
    
    test_scenarios = [
        (8, 0, True, "Ultra-short stack bust"),
        (15, 0, True, "Short stack bust"),
        (25, 0, True, "Medium stack bust"),
        (12, 18, False, "Short stack survival + growth"),
        (60, 90, False, "Deep stack growth")
    ]
    
    print(f"{'Scenario':<25} {'Aggressive':<12} {'Conservative':<12} {'Difference'}")
    print("-" * 65)
    
    for stack_before, stack_after, busted, description in test_scenarios:
        agg_payoff = aggressive.apply_stack_adjustments(0.0, stack_before, stack_after, busted)
        con_payoff = conservative.apply_stack_adjustments(0.0, stack_before, stack_after, busted)
        difference = agg_payoff - con_payoff
        
        print(f"{description:<25} {agg_payoff:<12.2f} {con_payoff:<12.2f} {difference:>+8.2f}")
    
    print(f"\n📈 Analysis:")
    print(f"   • Positive differences = Aggressive trainer is less punishing")
    print(f"   • Larger differences on bust scenarios = More willingness to take risks")
    print(f"   • Similar values on growth scenarios = Both reward success equally")


if __name__ == "__main__":
    demonstrate_penalty_effects()
    test_penalty_in_action()
    
    print(f"\n✅ Tournament survival penalty adjustment complete!")
    print(f"🎯 The model should now be more willing to take risks with reasonable hands")
    print(f"   when short-stacked, leading to more realistic tournament play.")