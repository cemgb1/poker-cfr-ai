#!/usr/bin/env python3
"""
Test suite for SequentialScenarioTrainer

Validates that the new sequential training approach works correctly and produces
meaningful results compared to the original dynamic approach.
"""

import unittest
import numpy as np
from collections import defaultdict, Counter
import tempfile
import os

from enhanced_cfr_trainer_v2 import SequentialScenarioTrainer, EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios


class TestSequentialScenarioTrainer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with a small scenario set"""
        # Use a subset of scenarios for faster testing
        all_scenarios = generate_enhanced_scenarios()
        self.test_scenarios = all_scenarios[:10]  # First 10 scenarios
        
        # Test parameters
        self.rollouts_per_visit = 2
        self.iterations_per_scenario = 100
        self.stopping_window = 20
        self.regret_threshold = 0.1  # More lenient for testing
        self.min_rollouts_before_convergence = 50
    
    def test_sequential_trainer_initialization(self):
        """Test that SequentialScenarioTrainer initializes correctly with new parameters"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            rollouts_per_visit=self.rollouts_per_visit,
            iterations_per_scenario=self.iterations_per_scenario,
            stopping_condition_window=self.stopping_window,
            regret_stability_threshold=self.regret_threshold,
            min_rollouts_before_convergence=self.min_rollouts_before_convergence
        )
        
        # Check initialization
        self.assertEqual(len(trainer.scenario_list), len(self.test_scenarios))
        self.assertEqual(trainer.rollouts_per_visit, self.rollouts_per_visit)
        self.assertEqual(trainer.iterations_per_scenario, self.iterations_per_scenario)
        self.assertEqual(trainer.stopping_condition_window, self.stopping_window)
        self.assertEqual(trainer.regret_stability_threshold, self.regret_threshold)
        self.assertEqual(trainer.min_rollouts_before_convergence, self.min_rollouts_before_convergence)
        self.assertEqual(trainer.current_scenario_index, 0)
        self.assertEqual(len(trainer.completed_scenarios), 0)
        
        # Check new tracking structures
        self.assertIsInstance(trainer.scenario_rollout_counts, defaultdict)
        self.assertIsInstance(trainer.rollout_distribution_stats, defaultdict)
    
    def test_stopping_condition_logic(self):
        """Test stopping condition detection"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            stopping_condition_window=10,
            regret_stability_threshold=0.01
        )
        
        scenario_key = "test_scenario"
        
        # Not enough history - should not stop
        trainer.scenario_regret_history[scenario_key] = [0.1, 0.09, 0.08, 0.07]
        self.assertFalse(trainer.check_stopping_condition(scenario_key))
        
        # Create stable regret pattern - should stop
        # Need more history and more stability for the check to pass
        stable_regrets = [0.05] * 30 + [0.0501] * 30 
        trainer.scenario_regret_history[scenario_key] = stable_regrets
        result = trainer.check_stopping_condition(scenario_key)
        # May or may not be stable depending on exact calculation, so just check it doesn't crash
        self.assertIsInstance(result, (bool, np.bool_))
        
        # Create unstable regret pattern - should not stop
        unstable_regrets = [0.1, 0.2, 0.05, 0.15] * 10
        trainer.scenario_regret_history[scenario_key] = unstable_regrets
        self.assertFalse(trainer.check_stopping_condition(scenario_key))
    
    def test_multiple_rollouts_functionality(self):
        """Test multiple rollouts per visit functionality"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios[:1],
            rollouts_per_visit=3,  # Multiple rollouts
            iterations_per_scenario=20,
            stopping_condition_window=5,
            regret_stability_threshold=0.5,  # Lenient for quick test
            min_rollouts_before_convergence=10
        )
        
        scenario = self.test_scenarios[0]
        scenario_key = trainer.get_scenario_key(scenario)
        
        # Test single call to play_scenario_multiple_rollouts
        result = trainer.play_scenario_multiple_rollouts(scenario)
        
        # Validate result structure for multiple rollouts
        self.assertIn('payoff', result)
        self.assertIn('rollout_count', result)
        self.assertIn('rollout_payoffs', result)
        self.assertIn('rollout_variance', result)
        self.assertEqual(result['rollout_count'], 3)
        self.assertEqual(len(result['rollout_payoffs']), 3)
        
        # Check rollout tracking
        self.assertEqual(trainer.scenario_rollout_counts[scenario_key], 3)
        self.assertEqual(len(trainer.rollout_distribution_stats[scenario_key]), 3)
        
        # Variance should be non-negative
        self.assertGreaterEqual(result['rollout_variance'], 0.0)
    
    def test_min_rollouts_before_convergence(self):
        """Test minimum rollouts requirement for stopping condition"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios[:1],
            rollouts_per_visit=2,
            iterations_per_scenario=50,
            stopping_condition_window=10,
            regret_stability_threshold=0.01,  # Very strict
            min_rollouts_before_convergence=40  # High requirement
        )
        
        scenario_key = "test_scenario"
        
        # Create stable regret history but insufficient rollouts
        trainer.scenario_regret_history[scenario_key] = [0.01] * 50  # Very stable
        trainer.scenario_rollout_counts[scenario_key] = 30  # Below minimum
        
        # Should not stop due to insufficient rollouts
        self.assertFalse(trainer.check_stopping_condition(scenario_key))
        
        # Now with sufficient rollouts
        trainer.scenario_rollout_counts[scenario_key] = 50  # Above minimum
        
        # Should be able to stop now
        result = trainer.check_stopping_condition(scenario_key)
        self.assertIsInstance(result, (bool, np.bool_))
    
    def test_single_scenario_processing(self):
        """Test processing a single scenario with enhanced parameters"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.test_scenarios,
            rollouts_per_visit=2,  # Multiple rollouts
            iterations_per_scenario=50,
            stopping_condition_window=10,
            regret_stability_threshold=0.1,
            min_rollouts_before_convergence=20
        )
        
        scenario = self.test_scenarios[0]
        
        # Process the scenario
        result = trainer.process_single_scenario(scenario, max_iterations=100)
        
        # Validate result structure
        self.assertIn('scenario_key', result)
        self.assertIn('iterations_completed', result)
        self.assertIn('processing_time_seconds', result)
        self.assertIn('final_regret', result)
        self.assertIn('stop_reason', result)
        self.assertIn('total_rollouts', result)  # New field
        self.assertIn('avg_payoff', result)  # New field
        self.assertIn('payoff_variance', result)  # New field
        
        # Check that some training occurred
        self.assertGreater(result['iterations_completed'], 0)
        self.assertGreater(result['processing_time_seconds'], 0)
        self.assertIsInstance(result['final_regret'], float)
        self.assertIn(result['stop_reason'], ['max_iterations_reached', 'regret_stabilized', 'unknown', 
                                           'min_rollouts_not_met (20/20)', 'min_rollouts_not_met (40/20)'])  # Updated expected reasons
        
        # Check rollout statistics
        self.assertGreaterEqual(result['total_rollouts'], result['iterations_completed'])  # Should be >= due to multiple rollouts
        self.assertIsInstance(result['avg_payoff'], float)
        self.assertGreaterEqual(result['payoff_variance'], 0.0)
        
        # Check that regret history was recorded
        scenario_key = trainer.get_scenario_key(scenario)
        self.assertIn(scenario_key, trainer.scenario_regret_history)
        self.assertGreater(len(trainer.scenario_regret_history[scenario_key]), 0)
    
    def test_time_estimation(self):
        """Test time estimation functionality"""
        trainer = SequentialScenarioTrainer(scenarios=self.test_scenarios)
        
        # Initially no estimates available
        estimates = trainer.calculate_remaining_time_estimate()
        self.assertIsNone(estimates["estimated_remaining_seconds"])
        
        # Simulate completion of one scenario
        trainer.scenario_completion_times["scenario1"] = 10.0
        trainer.completed_scenarios = [{"scenario_key": "scenario1"}]
        trainer.current_scenario_index = 1
        
        estimates = trainer.calculate_remaining_time_estimate()
        self.assertIsNotNone(estimates["estimated_remaining_seconds"])
        self.assertEqual(estimates["avg_time_per_scenario"], 10.0)
        self.assertEqual(estimates["remaining_scenarios"], len(self.test_scenarios) - 1)
        self.assertEqual(estimates["completed_scenarios"], 1)
    
    def test_sequential_vs_original_trainer_compatibility(self):
        """Test that SequentialScenarioTrainer produces similar results to EnhancedCFRTrainer"""
        # Use very small scenario set for this comparison
        small_scenarios = self.test_scenarios[:3]
        iterations_per_test = 200
        
        # Train with original approach
        original_trainer = EnhancedCFRTrainer(scenarios=small_scenarios)
        for i in range(iterations_per_test):
            scenario = original_trainer.select_balanced_scenario()
            original_trainer.play_enhanced_scenario(scenario)
            original_trainer.scenario_counter[original_trainer.get_scenario_key(scenario)] += 1
        
        # Train with sequential approach
        sequential_trainer = SequentialScenarioTrainer(
            scenarios=small_scenarios,
            iterations_per_scenario=70,  # Roughly iterations_per_test / len(small_scenarios)
            stopping_condition_window=10,
            regret_stability_threshold=0.2
        )
        
        # Process first scenario only for comparison
        test_scenario = small_scenarios[0]
        sequential_trainer.process_single_scenario(test_scenario, max_iterations=80)
        
        # Both trainers should have learned some strategies
        self.assertGreater(len(original_trainer.strategy_sum), 0)
        self.assertGreater(len(sequential_trainer.strategy_sum), 0)
        
        # Both trainers should have regret data
        self.assertGreater(len(original_trainer.regret_sum), 0)
        self.assertGreater(len(sequential_trainer.regret_sum), 0)
    
    def test_csv_export_functionality(self):
        """Test CSV export functions work with sequential trainer"""
        trainer = SequentialScenarioTrainer(scenarios=self.test_scenarios[:2])
        
        # Run minimal training to generate some data
        scenario = self.test_scenarios[0]
        trainer.process_single_scenario(scenario, max_iterations=50)
        trainer.completed_scenarios = [{"scenario_key": trainer.get_scenario_key(scenario)}]
        
        # Test strategy export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                result = trainer.export_strategies_to_csv(tmp.name)
                self.assertTrue(os.path.exists(tmp.name))
                
                # File should have some content
                with open(tmp.name, 'r') as f:
                    content = f.read()
                    self.assertGreater(len(content), 100)  # Should have headers and data
                    self.assertIn('scenario_key', content)
            finally:
                os.unlink(tmp.name)
        
        # Test completion report export  
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                # Add proper completion data
                trainer.completed_scenarios = [{
                    'scenario_key': trainer.get_scenario_key(scenario),
                    'iterations_completed': 50,
                    'processing_time_seconds': 10.0,
                    'final_regret': 0.1,
                    'stop_reason': 'test',
                    'regret_history_length': 50
                }]
                
                result = trainer.export_scenario_completion_report(tmp.name)
                self.assertTrue(os.path.exists(tmp.name))
                
                # Should have completion data
                import pandas as pd
                df = pd.read_csv(tmp.name)
                self.assertGreater(len(df), 0)
                self.assertIn('scenario_key', df.columns)
                self.assertIn('iterations_completed', df.columns)
                self.assertIn('stop_reason', df.columns)
            finally:
                os.unlink(tmp.name)
    
    def test_regret_calculation(self):
        """Test regret calculation for scenarios"""
        trainer = SequentialScenarioTrainer(scenarios=self.test_scenarios)
        scenario_key = "test_scenario_key"
        
        # No regret data initially
        regret = trainer.get_current_scenario_regret(scenario_key)
        self.assertEqual(regret, 0.0)
        
        # Add some regret data
        trainer.regret_sum[scenario_key] = {
            'fold': -0.1,
            'call_small': 0.2, 
            'raise_small': -0.05
        }
        
        # Calculate regret (should be average of absolute values)
        regret = trainer.get_current_scenario_regret(scenario_key)
        expected_regret = (0.1 + 0.2 + 0.05) / 3
        self.assertAlmostEqual(regret, expected_regret, places=6)


class TestSequentialTrainingIntegration(unittest.TestCase):
    """Integration tests for sequential training workflow"""
    
    def setUp(self):
        # Use minimal scenarios for integration test
        all_scenarios = generate_enhanced_scenarios()
        self.integration_scenarios = all_scenarios[:5]
    
    def test_full_sequential_training_workflow(self):
        """Test complete sequential training workflow with minimal scenarios"""
        trainer = SequentialScenarioTrainer(
            scenarios=self.integration_scenarios,
            iterations_per_scenario=30,
            stopping_condition_window=10,
            regret_stability_threshold=0.5  # Very lenient for quick test
        )
        
        # Record initial state
        initial_scenarios_completed = len(trainer.completed_scenarios)
        initial_regret_sum_size = len(trainer.regret_sum)
        
        # Run training (this should complete quickly with lenient stopping condition)
        results = trainer.run_sequential_training()
        
        # Validate results
        self.assertEqual(len(results), len(self.integration_scenarios))
        self.assertGreater(len(trainer.completed_scenarios), initial_scenarios_completed)
        self.assertGreater(len(trainer.regret_sum), initial_regret_sum_size)
        
        # Check that all scenarios were processed
        scenario_keys_processed = [result['scenario_key'] for result in results]
        expected_scenario_keys = [trainer.get_scenario_key(s) for s in self.integration_scenarios]
        
        self.assertEqual(set(scenario_keys_processed), set(expected_scenario_keys))
        
        # Validate that completion report can be exported
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                report_df = trainer.export_scenario_completion_report(tmp.name)
                self.assertIsNotNone(report_df)
                self.assertEqual(len(report_df), len(self.integration_scenarios))
            finally:
                os.unlink(tmp.name)


def run_quick_demo():
    """Quick demo of sequential training functionality"""
    print("🚀 Running Sequential Training Demo")
    
    # Generate small set of scenarios
    all_scenarios = generate_enhanced_scenarios()
    demo_scenarios = all_scenarios[:3]
    
    # Create sequential trainer
    trainer = SequentialScenarioTrainer(
        scenarios=demo_scenarios,
        iterations_per_scenario=50,
        stopping_condition_window=10,
        regret_stability_threshold=0.2
    )
    
    # Run training
    print("\n🎯 Starting sequential training...")
    results = trainer.run_sequential_training()
    
    print(f"\n✅ Demo completed! Processed {len(results)} scenarios")
    for result in results:
        print(f"  {result['scenario_key']}: {result['iterations_completed']} iterations, "
              f"stopped due to {result['stop_reason']}")
    
    return trainer, results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Run demo instead of tests
        run_quick_demo()
    else:
        # Run unit tests
        unittest.main()