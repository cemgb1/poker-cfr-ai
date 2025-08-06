#!/usr/bin/env python3
"""
Comprehensive test suite for advanced pruning techniques and stopping conditions

This test suite validates:
1. Regret-based pruning with restoration mechanics
2. Strategy pruning with probability thresholds
3. Action space pruning with contextual filtering
4. Modular stopping conditions
5. Backward compatibility
"""

import unittest
import numpy as np
import tempfile
import os
from collections import defaultdict
from unittest.mock import patch, MagicMock

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer, SequentialScenarioTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios


class TestPruningFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create small scenario set for testing
        all_scenarios = generate_enhanced_scenarios()
        self.test_scenarios = all_scenarios[:5]  # Use first 5 scenarios for speed
        
        # Test parameters
        self.regret_threshold = -100.0
        self.strategy_threshold = 0.01
    
    def test_enhanced_cfr_trainer_initialization_with_pruning(self):
        """Test EnhancedCFRTrainer initialization with pruning parameters"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True,
            regret_pruning_threshold=self.regret_threshold,
            strategy_pruning_threshold=self.strategy_threshold
        )
        
        # Check pruning parameters
        self.assertTrue(trainer.enable_pruning)
        self.assertEqual(trainer.regret_pruning_threshold, self.regret_threshold)
        self.assertEqual(trainer.strategy_pruning_threshold, self.strategy_threshold)
        
        # Check pruning data structures
        self.assertIsInstance(trainer.pruned_actions, defaultdict)
        self.assertIsInstance(trainer.pruning_statistics, dict)
        self.assertIsInstance(trainer.restoration_chances, defaultdict)
        
        # Check required statistics keys
        required_stats = ['regret_pruned_count', 'strategy_pruned_count', 
                         'actions_restored_count', 'total_pruning_events']
        for key in required_stats:
            self.assertIn(key, trainer.pruning_statistics)
    
    def test_backward_compatibility_without_pruning(self):
        """Test that trainer works without pruning parameters (backward compatibility)"""
        trainer = EnhancedCFRTrainer(scenarios=self.test_scenarios)
        
        # Should default to pruning enabled
        self.assertTrue(trainer.enable_pruning)
        self.assertEqual(trainer.regret_pruning_threshold, -300.0)
        self.assertEqual(trainer.strategy_pruning_threshold, 0.001)
    
    def test_pruning_disabled_mode(self):
        """Test trainer with pruning explicitly disabled"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=False
        )
        
        self.assertFalse(trainer.enable_pruning)
        
        # Test get_relevant_actions returns all actions when pruning disabled
        scenario = self.test_scenarios[0]
        betting_context = {'bet_size': 10}
        relevant_actions = trainer.get_relevant_actions(scenario, betting_context)
        
        expected_actions = ["fold", "call_small", "call_mid", "call_high", 
                           "raise_small", "raise_mid", "raise_high"]
        self.assertEqual(set(relevant_actions), set(expected_actions))
    
    def test_action_space_pruning(self):
        """Test contextual action space pruning"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True
        )
        
        # Test trash hand - should have limited actions
        trash_scenario = {
            'hand_category': 'trash',
            'stack_category': 'medium'
        }
        betting_context = {'bet_size': 10}
        
        relevant_actions = trainer.get_relevant_actions(trash_scenario, betting_context)
        self.assertIn('fold', relevant_actions)
        self.assertIn('call_small', relevant_actions)
        # Should not have high raises for trash hands
        self.assertNotIn('raise_high', relevant_actions)
        
        # Test premium hand - should have more actions
        premium_scenario = {
            'hand_category': 'premium_pairs',
            'stack_category': 'deep'
        }
        
        premium_actions = trainer.get_relevant_actions(premium_scenario, betting_context)
        self.assertIn('fold', premium_actions)
        self.assertIn('raise_small', premium_actions)
        self.assertIn('raise_mid', premium_actions)
        # Premium hands should have more aggressive options
        
        # Test short stack - should remove high actions
        short_scenario = {
            'hand_category': 'medium_aces',
            'stack_category': 'ultra_short'
        }
        
        short_actions = trainer.get_relevant_actions(short_scenario, betting_context)
        self.assertNotIn('call_high', short_actions)
        self.assertNotIn('raise_high', short_actions)
    
    def test_regret_based_pruning_in_get_strategy(self):
        """Test regret-based pruning in strategy calculation"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True,
            regret_pruning_threshold=-50.0  # Higher threshold for testing
        )
        
        scenario_key = "test_scenario"
        available_actions = ["fold", "call_small", "raise_small"]
        
        # Set up regrets - one below threshold
        trainer.regret_sum[scenario_key]["fold"] = -100.0  # Below threshold
        trainer.regret_sum[scenario_key]["call_small"] = 10.0  # Above threshold
        trainer.regret_sum[scenario_key]["raise_small"] = 5.0  # Above threshold
        
        # Get strategy with pruning
        strategy = trainer.get_strategy(scenario_key, available_actions)
        
        # Should not include the action below threshold
        self.assertNotIn("fold", strategy)
        self.assertIn("call_small", strategy)
        self.assertIn("raise_small", strategy)
        
        # Check that fold was added to pruned actions
        self.assertIn("fold", trainer.pruned_actions[scenario_key])
    
    def test_action_restoration_mechanism(self):
        """Test 1% restoration chance for pruned actions"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True
        )
        
        scenario_key = "test_scenario"
        trainer.pruned_actions[scenario_key].add("fold")
        trainer.pruned_actions[scenario_key].add("call_small")
        
        # Mock random to guarantee restoration
        with patch('random.random', return_value=0.005):  # < 1%
            trainer._restore_pruned_actions(scenario_key)
        
        # Should have restored one action and incremented counter
        self.assertEqual(trainer.pruning_statistics['actions_restored_count'], 1)
        self.assertTrue(trainer.restoration_chances[scenario_key] >= 1)
    
    def test_regret_update_with_pruning(self):
        """Test that regret updates skip pruned actions"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True,
            regret_pruning_threshold=-50.0
        )
        
        scenario_key = "test_scenario"
        available_actions = ["fold", "call_small", "raise_small"]
        
        # Set up pruned action
        trainer.pruned_actions[scenario_key].add("fold")
        trainer.regret_sum[scenario_key]["fold"] = -100.0  # Below threshold
        
        strategy = {"call_small": 0.7, "raise_small": 0.3}
        payoff_result = {"payoff": 5.0}
        
        initial_regret = trainer.regret_sum[scenario_key]["fold"]
        
        trainer.update_enhanced_regrets(
            scenario_key, "call_small", strategy, payoff_result, available_actions
        )
        
        # Pruned action regret should not be updated
        self.assertEqual(trainer.regret_sum[scenario_key]["fold"], initial_regret)
    
    def test_strategy_pruning_in_export(self):
        """Test strategy pruning during CSV export"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True,
            strategy_pruning_threshold=0.1  # 10% threshold for testing
        )
        
        scenario_key = "premium_pairs|BTN|medium|low"
        
        # Set up strategy counts with one low-probability action
        trainer.strategy_sum[scenario_key]["fold"] = 5      # 5%
        trainer.strategy_sum[scenario_key]["call_small"] = 70   # 70%
        trainer.strategy_sum[scenario_key]["raise_small"] = 25  # 25%
        trainer.scenario_counter[scenario_key] = 100
        
        # Export strategies
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            df = trainer.export_strategies_to_csv(tmp.name)
            
            # Check that low-probability action was pruned
            row = df[df['scenario_key'] == scenario_key].iloc[0]
            self.assertEqual(row['actions_pruned_count'], 1)  # fold should be pruned
            self.assertEqual(row['fold_prob'], 0.0)
            
            # Check that probabilities were renormalized
            self.assertAlmostEqual(row['call_small_prob'] + row['raise_small_prob'], 1.0, places=2)
            
            os.unlink(tmp.name)
    
    def test_pruning_statistics_collection(self):
        """Test comprehensive pruning statistics collection"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True
        )
        
        # Simulate some pruning events
        trainer.pruning_statistics['regret_pruned_count'] = 10
        trainer.pruning_statistics['strategy_pruned_count'] = 5
        trainer.pruning_statistics['actions_restored_count'] = 2
        trainer.pruning_statistics['total_pruning_events'] = 8
        
        # Add some scenarios to regret_sum
        trainer.regret_sum['scenario1']['fold'] = -100
        trainer.regret_sum['scenario2']['call'] = -50
        trainer.pruned_actions['scenario1'].add('fold')
        trainer.restoration_chances['scenario1'] = 3
        
        stats = trainer.get_pruning_statistics()
        
        # Check basic statistics
        self.assertEqual(stats['regret_pruned_count'], 10)
        self.assertEqual(stats['strategy_pruned_count'], 5)
        self.assertEqual(stats['actions_restored_count'], 2)
        
        # Check calculated metrics
        self.assertEqual(stats['scenarios_with_pruning'], 1)
        self.assertEqual(stats['scenarios_with_restorations'], 1)
        self.assertEqual(stats['total_restoration_attempts'], 3)


class TestSequentialTrainerStoppingConditions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        all_scenarios = generate_enhanced_scenarios()
        self.test_scenarios = all_scenarios[:3]  # Use first 3 scenarios
    
    def test_sequential_trainer_initialization_with_stopping_conditions(self):
        """Test SequentialScenarioTrainer initialization with new stopping parameters"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            enable_min_rollouts_stopping=True,
            enable_regret_stability_stopping=True,
            enable_max_iterations_stopping=False,
            stopping_condition_mode='flexible'
        )
        
        # Check stopping condition parameters
        self.assertTrue(trainer.enable_min_rollouts_stopping)
        self.assertTrue(trainer.enable_regret_stability_stopping)
        self.assertFalse(trainer.enable_max_iterations_stopping)
        self.assertEqual(trainer.stopping_condition_mode, 'flexible')
        
        # Check stopping tracking structures
        self.assertIsInstance(trainer.stopping_reasons, defaultdict)
        self.assertIsInstance(trainer.stopping_statistics, dict)
    
    def test_modular_stopping_conditions_strict_mode(self):
        """Test strict stopping condition mode"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            enable_min_rollouts_stopping=True,
            enable_regret_stability_stopping=True,
            enable_max_iterations_stopping=True,
            stopping_condition_mode='strict',
            min_rollouts_before_convergence=10
        )
        
        scenario_key = "test_scenario"
        
        # Set up scenario state - only min rollouts met
        trainer.scenario_rollout_counts[scenario_key] = 15
        trainer.scenario_iteration_counts[scenario_key] = 50
        trainer.scenario_regret_history[scenario_key] = [1.0] * 50  # Not stable
        
        # Should not stop in strict mode (not all conditions met)
        should_stop, reason = trainer.check_stopping_condition(scenario_key, 50)
        self.assertFalse(should_stop)
    
    def test_modular_stopping_conditions_flexible_mode(self):
        """Test flexible stopping condition mode"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            enable_min_rollouts_stopping=True,
            enable_regret_stability_stopping=True,
            enable_max_iterations_stopping=True,
            stopping_condition_mode='flexible',
            min_rollouts_before_convergence=10,
            iterations_per_scenario=100
        )
        
        scenario_key = "test_scenario"
        
        # Set up scenario state - min rollouts met, max iterations reached
        trainer.scenario_rollout_counts[scenario_key] = 15
        trainer.scenario_iteration_counts[scenario_key] = 100
        
        # Should stop in flexible mode (min rollouts + max iterations met)
        should_stop, reason = trainer.check_stopping_condition(scenario_key, 100)
        self.assertTrue(should_stop)
        self.assertEqual(reason, "max_iterations_reached")
    
    def test_regret_stability_detection(self):
        """Test regret stability detection mechanism"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            stopping_condition_window=10,
            regret_stability_threshold=0.05
        )
        
        # Create stable regret history
        stable_history = [10.0] * 15 + [10.1] * 10 + [10.05] * 10  # Stable at ~10
        self.assertTrue(trainer._check_regret_stability(stable_history))
        
        # Create unstable regret history
        unstable_history = [10.0] * 15 + [15.0] * 10 + [20.0] * 10  # Increasing
        self.assertFalse(trainer._check_regret_stability(unstable_history))
    
    def test_stopping_condition_reconfiguration(self):
        """Test runtime reconfiguration of stopping conditions"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            enable_regret_stability_stopping=True,
            regret_stability_threshold=0.05
        )
        
        # Reconfigure stopping conditions
        trainer.reconfigure_stopping_conditions(
            enable_regret_stability_stopping=False,
            regret_stability_threshold=0.1,
            stopping_condition_mode='strict'
        )
        
        # Check that parameters were updated
        self.assertFalse(trainer.enable_regret_stability_stopping)
        self.assertEqual(trainer.regret_stability_threshold, 0.1)
        self.assertEqual(trainer.stopping_condition_mode, 'strict')
    
    def test_stopping_statistics_collection(self):
        """Test stopping statistics collection and reporting"""
        trainer = SequentialScenarioTrainer(scenarios=self.test_scenarios)
        
        # Simulate some stopping events
        trainer.stopping_statistics['regret_stability_stops'] = 5
        trainer.stopping_statistics['max_iterations_stops'] = 3
        trainer.completed_scenarios = ['s1', 's2', 's3']
        trainer.stopping_reasons['s1'] = ['regret_stabilized']
        trainer.stopping_reasons['s2'] = ['max_iterations_reached']
        
        stats = trainer.get_stopping_statistics()
        
        self.assertEqual(stats['regret_stability_stops'], 5)
        self.assertEqual(stats['max_iterations_stops'], 3)
        self.assertEqual(stats['total_scenarios_completed'], 3)
        self.assertIn('stopping_reason_distribution', stats)
    
    def test_custom_stopping_logic_override(self):
        """Test custom stopping logic can be overridden"""
        class CustomTrainer(SequentialScenarioTrainer):
            def _custom_stopping_logic(self, scenario_key, conditions_met, current_iteration):
                # Custom logic: stop after 50 iterations regardless of other conditions
                return current_iteration >= 50, "custom_iteration_limit"
        
        trainer = CustomTrainer(
            scenarios=self.test_scenarios,
            stopping_condition_mode='custom'
        )
        
        scenario_key = "test_scenario"
        conditions_met = [('min_rollouts', False)]
        
        should_stop, reason = trainer._custom_stopping_logic(scenario_key, conditions_met, 60)
        self.assertTrue(should_stop)
        self.assertEqual(reason, "custom_iteration_limit")


class TestIntegrationAndCompatibility(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        all_scenarios = generate_enhanced_scenarios()
        self.test_scenarios = all_scenarios[:2]  # Minimal set for integration testing
    
    def test_enhanced_trainer_with_all_pruning_features(self):
        """Test EnhancedCFRTrainer with all pruning features enabled"""
        trainer = EnhancedCFRTrainer(
            scenarios=self.test_scenarios,
            enable_pruning=True,
            regret_pruning_threshold=-100.0,
            strategy_pruning_threshold=0.05
        )
        
        # Run a few scenarios to test integration
        for scenario in self.test_scenarios:
            result = trainer.play_enhanced_scenario(scenario)
            
            # Check that result has expected fields
            self.assertIn('scenario_key', result)
            self.assertIn('available_actions', result)
            self.assertIn('initial_actions', result)  # New field for pruning tracking
            
            # Verify action space pruning worked
            self.assertLessEqual(len(result['available_actions']), 
                               len(result['initial_actions']))
    
    def test_sequential_trainer_with_all_features(self):
        """Test SequentialScenarioTrainer with all pruning and stopping features"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            rollouts_per_visit=2,
            iterations_per_scenario=50,
            enable_min_rollouts_stopping=True,
            enable_regret_stability_stopping=True,
            enable_max_iterations_stopping=True,
            stopping_condition_mode='flexible',
            enable_pruning=True,
            regret_pruning_threshold=-200.0,
            strategy_pruning_threshold=0.01,
            min_rollouts_before_convergence=20
        )
        
        # Test that trainer is properly initialized
        self.assertTrue(trainer.enable_pruning)
        self.assertEqual(trainer.stopping_condition_mode, 'flexible')
        
        # Test pruning statistics are available
        stats = trainer.get_pruning_statistics()
        self.assertIsInstance(stats, dict)
        
        # Test stopping statistics are available
        stop_stats = trainer.get_stopping_statistics()
        self.assertIsInstance(stop_stats, dict)
    
    def test_backward_compatibility_with_existing_code(self):
        """Test that existing code still works without modification"""
        # Test original EnhancedCFRTrainer initialization
        trainer1 = EnhancedCFRTrainer(scenarios=self.test_scenarios)
        self.assertIsInstance(trainer1, EnhancedCFRTrainer)
        
        # Test original SequentialScenarioTrainer initialization
        trainer2 = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            rollouts_per_visit=1,
            iterations_per_scenario=100
        )
        self.assertIsInstance(trainer2, SequentialScenarioTrainer)
        
        # Test that methods still work
        result = trainer1.play_enhanced_scenario(self.test_scenarios[0])
        self.assertIsInstance(result, dict)


def run_pruning_tests():
    """Run all pruning functionality tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPruningFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestSequentialTrainerStoppingConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationAndCompatibility))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Comprehensive Pruning and Stopping Conditions Test Suite")
    print("=" * 80)
    success = run_pruning_tests()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        exit(1)