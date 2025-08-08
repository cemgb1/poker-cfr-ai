#!/usr/bin/env python3
"""
Test suite for tournament survival penalty parameter functionality.

This test validates that the new tournament_survival_penalty parameter:
1. Is properly initialized with default value
2. Can be customized during trainer initialization
3. Correctly scales bust penalties in apply_stack_adjustments
4. Maintains backward compatibility
5. Produces expected payoff differences for different penalty levels
"""

import unittest
import numpy as np
from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer, SequentialScenarioTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios


class TestTournamentSurvivalPenalty(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create small scenario set for testing
        all_scenarios = generate_enhanced_scenarios()
        self.test_scenarios = all_scenarios[:3]  # Use first 3 scenarios for speed
    
    def test_default_tournament_survival_penalty(self):
        """Test that default tournament survival penalty is set correctly"""
        trainer = EnhancedCFRTrainer(scenarios=self.test_scenarios)
        
        # Check default value
        self.assertEqual(trainer.tournament_survival_penalty, 0.6)
        self.assertLess(trainer.tournament_survival_penalty, 1.0)  # Should be less punishing than original
    
    def test_custom_tournament_survival_penalty(self):
        """Test custom tournament survival penalty values"""
        # Test aggressive setting (low penalty)
        aggressive_trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.3
        )
        self.assertEqual(aggressive_trainer.tournament_survival_penalty, 0.3)
        
        # Test conservative setting (high penalty)
        conservative_trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.9
        )
        self.assertEqual(conservative_trainer.tournament_survival_penalty, 0.9)
        
        # Test original harsh setting
        original_trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=1.0
        )
        self.assertEqual(original_trainer.tournament_survival_penalty, 1.0)
    
    def test_penalty_scaling_in_stack_adjustments(self):
        """Test that bust penalties are correctly scaled by the parameter"""
        # Create trainers with different penalty levels
        aggressive_trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.4
        )
        
        conservative_trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.8
        )
        
        original_trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=1.0
        )
        
        # Test short stack bust penalties (should be -8.0 * penalty_factor)
        base_payoff = 0.0
        stack_before = 10  # Short stack
        stack_after = 0    # Busted
        busted = True
        
        aggressive_payoff = aggressive_trainer.apply_stack_adjustments(
            base_payoff, stack_before, stack_after, busted
        )
        conservative_payoff = conservative_trainer.apply_stack_adjustments(
            base_payoff, stack_before, stack_after, busted
        )
        original_payoff = original_trainer.apply_stack_adjustments(
            base_payoff, stack_before, stack_after, busted
        )
        
        # Check expected scaled penalties
        expected_aggressive = -8.0 * 0.4  # -3.2
        expected_conservative = -8.0 * 0.8  # -6.4
        expected_original = -8.0 * 1.0  # -8.0
        
        self.assertAlmostEqual(aggressive_payoff, expected_aggressive, places=2)
        self.assertAlmostEqual(conservative_payoff, expected_conservative, places=2)
        self.assertAlmostEqual(original_payoff, expected_original, places=2)
        
        # Verify ordering: aggressive < conservative < original (less negative = less punishing)
        self.assertGreater(aggressive_payoff, conservative_payoff)
        self.assertGreater(conservative_payoff, original_payoff)
    
    def test_medium_stack_bust_penalties(self):
        """Test penalty scaling for medium stack busts"""
        trainer_half = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.5
        )
        
        # Medium stack bust (should be -5.0 * 0.5 = -2.5)
        payoff = trainer_half.apply_stack_adjustments(
            base_payoff=0.0, 
            stack_before=25,  # Medium stack
            stack_after=0,    # Busted
            busted=True
        )
        
        expected_payoff = -5.0 * 0.5  # -2.5
        self.assertAlmostEqual(payoff, expected_payoff, places=2)
    
    def test_deep_stack_bust_penalties(self):
        """Test penalty scaling for deep stack busts"""
        trainer_quarter = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.25
        )
        
        # Deep stack bust (should be -3.0 * 0.25 = -0.75)
        payoff = trainer_quarter.apply_stack_adjustments(
            base_payoff=0.0,
            stack_before=80,  # Deep stack
            stack_after=0,    # Busted
            busted=True
        )
        
        expected_payoff = -3.0 * 0.25  # -0.75
        self.assertAlmostEqual(payoff, expected_payoff, places=2)
    
    def test_non_bust_scenarios_unchanged(self):
        """Test that non-bust scenarios are not affected by penalty scaling"""
        trainer_low = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.3
        )
        
        trainer_high = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.9
        )
        
        # Test survival scenario (no bust)
        base_payoff = 2.0
        stack_before = 12  # Short stack
        stack_after = 15   # Gained chips
        busted = False
        
        payoff_low = trainer_low.apply_stack_adjustments(
            base_payoff, stack_before, stack_after, busted
        )
        payoff_high = trainer_high.apply_stack_adjustments(
            base_payoff, stack_before, stack_after, busted
        )
        
        # Should be the same since no bust penalty is applied
        self.assertAlmostEqual(payoff_low, payoff_high, places=3)
        
        # Should include survival bonus (unchanged by penalty parameter)
        expected_survival_bonus = 2.0 * (stack_after - stack_before) / stack_before
        expected_preservation_bonus = 0.5  # Didn't lose much
        expected_total = base_payoff + expected_survival_bonus + expected_preservation_bonus
        
        self.assertAlmostEqual(payoff_low, expected_total, places=2)
    
    def test_sequential_trainer_penalty_parameter(self):
        """Test that SequentialScenarioTrainer correctly accepts and uses the penalty parameter"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            rollouts_per_visit=1,
            iterations_per_scenario=10,
            tournament_survival_penalty=0.4
        )
        
        # Check that parameter was passed through correctly
        self.assertEqual(trainer.tournament_survival_penalty, 0.4)
        
        # Test that it's used in stack adjustments
        payoff = trainer.apply_stack_adjustments(
            base_payoff=0.0,
            stack_before=8,   # Ultra short
            stack_after=0,    # Busted
            busted=True
        )
        
        expected_payoff = -8.0 * 0.4  # -3.2
        self.assertAlmostEqual(payoff, expected_payoff, places=2)
    
    def test_extreme_penalty_values(self):
        """Test behavior with extreme penalty values"""
        # Very aggressive (almost no penalty)
        very_aggressive = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=0.1
        )
        
        # Very conservative (extra harsh penalty)
        very_conservative = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            tournament_survival_penalty=1.5
        )
        
        # Test short stack bust
        aggressive_payoff = very_aggressive.apply_stack_adjustments(
            0.0, 10, 0, True
        )
        conservative_payoff = very_conservative.apply_stack_adjustments(
            0.0, 10, 0, True
        )
        
        expected_aggressive = -8.0 * 0.1   # -0.8 (very mild penalty)
        expected_conservative = -8.0 * 1.5  # -12.0 (extra harsh penalty)
        
        self.assertAlmostEqual(aggressive_payoff, expected_aggressive, places=2)
        self.assertAlmostEqual(conservative_payoff, expected_conservative, places=2)
        
        # Verify extreme difference
        self.assertGreater(aggressive_payoff, -1.0)  # Almost no penalty
        self.assertLess(conservative_payoff, -10.0)  # Very harsh penalty
    
    def test_backward_compatibility(self):
        """Test that existing code without the parameter still works"""
        # This should work and use default value
        trainer = EnhancedCFRTrainer(scenarios=self.test_scenarios)
        self.assertEqual(trainer.tournament_survival_penalty, 0.6)
        
        # Test with only some parameters specified
        trainer2 = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=False
        )
        self.assertEqual(trainer2.tournament_survival_penalty, 0.6)
        self.assertFalse(trainer2.enable_pruning)


def run_tournament_penalty_tests():
    """Run all tournament survival penalty tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTournamentSurvivalPenalty))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Tournament Survival Penalty Test Suite")
    print("=" * 60)
    success = run_tournament_penalty_tests()
    if success:
        print("✅ All tournament penalty tests passed!")
        print("\n💡 Tournament Survival Penalty Usage Examples:")
        print("   # More aggressive (encourage risk-taking on short stacks)")
        print("   trainer = EnhancedCFRTrainer(tournament_survival_penalty=0.4)")
        print("   # Default balanced setting")
        print("   trainer = EnhancedCFRTrainer(tournament_survival_penalty=0.6)")
        print("   # More conservative (discourage busting)")
        print("   trainer = EnhancedCFRTrainer(tournament_survival_penalty=0.8)")
    else:
        print("❌ Some tournament penalty tests failed!")
        exit(1)