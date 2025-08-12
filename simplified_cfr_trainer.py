# simplified_cfr_trainer.py - Simplified CFR trainer with direct hole card scenarios

"""
Simplified CFR Trainer for Preflop Poker with Direct Hole Card Scenarios

This module implements a simplified CFR trainer that:
1. Uses direct hole card combinations instead of hand categories
2. Focuses on preflop-only simulation with immediate showdown
3. Randomly selects scenarios (hole cards + stack sizes) for each iteration
4. Tracks coverage to ensure all hole card combinations are eventually visited
5. Supports heads-up match mode where players continue until one is busted

Key Features:
- Monte Carlo scenario selection (random hole cards + stack sizes)
- Preflop-only simulation (no postflop actions)
- Coverage tracking for all 1326 hole card combinations
- Heads-up match mode with configurable stack resets
- Clean separation between scenario generation and CFR logic
- Checkpointing and resuming functionality
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import time
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta

from simplified_scenario_generator import (
    generate_random_scenario, create_scenario_key, get_available_actions,
    simulate_preflop_showdown, get_scenario_coverage_stats, ACTIONS
)


class SimplifiedCFRTrainer:
    """
    Simplified CFR Trainer using direct hole card scenarios.
    
    This trainer focuses on preflop-only poker with random scenario generation,
    ensuring full coverage of all possible hole card combinations over time.
    """
    
    def __init__(self, epsilon_exploration=0.1, min_visit_threshold=5, 
                 starting_stack_bb=100, logger=None):
        """
        Initialize Simplified CFR Trainer.
        
        Args:
            epsilon_exploration: Probability of forced exploration
            min_visit_threshold: Minimum visits before considering scenario trained
            starting_stack_bb: Starting stack size for heads-up mode
            logger: Logger instance for logging (optional)
        """
        # Store logger
        from logging_config import get_logger
        self.logger = logger if logger else get_logger("simplified_cfr_trainer")
        
        # Training parameters
        self.epsilon_exploration = epsilon_exploration
        self.min_visit_threshold = min_visit_threshold
        self.starting_stack_bb = starting_stack_bb
        
        # Strategy storage: scenario_key -> action -> (regret_sum, strategy_sum)
        self.hero_regrets = defaultdict(lambda: defaultdict(float))
        self.hero_strategy_sums = defaultdict(lambda: defaultdict(float))
        self.villain_regrets = defaultdict(lambda: defaultdict(float))
        self.villain_strategy_sums = defaultdict(lambda: defaultdict(float))
        
        # Visit tracking
        self.scenario_visits = defaultdict(int)
        self.state_action_visits = defaultdict(int)
        
        # Training statistics
        self.iterations_completed = 0
        self.scenarios_recorded = []
        self.visited_scenarios = set()
        
        # Heads-up match tracking
        self.hero_stack = starting_stack_bb
        self.villain_stack = starting_stack_bb
        self.hands_played = 0
        
        self.logger.info("🎲 Simplified CFR Trainer Initialized!")
        self.logger.info(f"   🎯 Epsilon exploration: {epsilon_exploration}")
        self.logger.info(f"   📊 Min visit threshold: {min_visit_threshold}")
        self.logger.info(f"   💰 Starting stack: {starting_stack_bb}bb")
    
    def get_strategy(self, scenario_key, available_actions, is_hero=True):
        """
        Get current strategy for a scenario using regret matching.
        
        Args:
            scenario_key: String key identifying the scenario
            available_actions: List of available action names
            is_hero: Whether this is for hero or villain
            
        Returns:
            dict: Action probabilities
        """
        regrets = self.hero_regrets if is_hero else self.villain_regrets
        
        # Calculate positive regrets
        positive_regrets = {}
        total_positive_regret = 0
        
        for action in available_actions:
            regret = max(0, regrets[scenario_key][action])
            positive_regrets[action] = regret
            total_positive_regret += regret
        
        # Convert to strategy
        strategy = {}
        if total_positive_regret > 0:
            for action in available_actions:
                strategy[action] = positive_regrets[action] / total_positive_regret
        else:
            # Uniform random if no positive regrets
            prob = 1.0 / len(available_actions)
            for action in available_actions:
                strategy[action] = prob
        
        return strategy
    
    def select_action(self, scenario_key, available_actions, is_hero=True):
        """
        Select an action using current strategy with epsilon-greedy exploration.
        
        Args:
            scenario_key: String key identifying the scenario
            available_actions: List of available action names
            is_hero: Whether this is for hero or villain
            
        Returns:
            str: Selected action name
        """
        # Check if we should explore
        visits = self.scenario_visits[scenario_key]
        should_explore = (random.random() < self.epsilon_exploration or 
                         visits < self.min_visit_threshold)
        
        if should_explore:
            return random.choice(available_actions)
        
        # Use current strategy
        strategy = self.get_strategy(scenario_key, available_actions, is_hero)
        
        # Sample from strategy distribution
        actions = list(strategy.keys())
        probabilities = list(strategy.values())
        
        return np.random.choice(actions, p=probabilities)
    
    def update_regrets(self, scenario_key, available_actions, chosen_action, 
                      action_values, is_hero=True):
        """
        Update regrets for a scenario using CFR.
        
        Args:
            scenario_key: String key identifying the scenario
            available_actions: List of available action names
            chosen_action: Action that was taken
            action_values: Dictionary of action -> expected value
            is_hero: Whether this is for hero or villain
        """
        regrets = self.hero_regrets if is_hero else self.villain_regrets
        strategy_sums = self.hero_strategy_sums if is_hero else self.villain_strategy_sums
        
        # Calculate regrets for each action
        chosen_value = action_values[chosen_action]
        strategy = self.get_strategy(scenario_key, available_actions, is_hero)
        
        for action in available_actions:
            # Regret = value of action - value of chosen action
            regret = action_values[action] - chosen_value
            regrets[scenario_key][action] += regret
            
            # Update strategy sum (for average strategy calculation)
            strategy_sums[scenario_key][action] += strategy[action]
    
    def simulate_scenario(self, scenario=None, heads_up_mode=False):
        """
        Simulate a single scenario and update strategies.
        
        Args:
            scenario: Pre-defined scenario (if None, generates random)
            heads_up_mode: Whether to use heads-up match stacks
            
        Returns:
            dict: Simulation results
        """
        # Generate or use provided scenario
        if scenario is None:
            scenario = generate_random_scenario()
            
        # Use heads-up stacks if in heads-up mode
        if heads_up_mode:
            scenario['hero_stack_bb'] = self.hero_stack
            scenario['villain_stack_bb'] = self.villain_stack
        
        # Create scenario keys
        hero_key = create_scenario_key(scenario['hero_cards'], scenario['hero_stack_bb'])
        villain_key = create_scenario_key(scenario['villain_cards'], scenario['villain_stack_bb'])
        
        # Get available actions
        bet_to_call = 1.5  # Big blind (simplified preflop betting)
        hero_actions = get_available_actions(scenario['hero_stack_bb'], bet_to_call)
        villain_actions = get_available_actions(scenario['villain_stack_bb'], bet_to_call)
        
        # Players select actions
        hero_action = self.select_action(hero_key, hero_actions, is_hero=True)
        villain_action = self.select_action(villain_key, villain_actions, is_hero=False)
        
        # Simulate showdown
        result = simulate_preflop_showdown(
            scenario['hero_cards'], scenario['villain_cards'],
            hero_action, villain_action,
            scenario['hero_stack_bb'], scenario['villain_stack_bb'],
            bet_to_call
        )
        
        # Calculate counterfactual values for regret updates
        hero_values = self._calculate_action_values(
            scenario, hero_actions, villain_action, is_hero=True
        )
        villain_values = self._calculate_action_values(
            scenario, villain_actions, hero_action, is_hero=False
        )
        
        # Update regrets
        self.update_regrets(hero_key, hero_actions, hero_action, hero_values, is_hero=True)
        self.update_regrets(villain_key, villain_actions, villain_action, villain_values, is_hero=False)
        
        # Update tracking
        self.scenario_visits[hero_key] += 1
        self.scenario_visits[villain_key] += 1
        self.visited_scenarios.add(hero_key)
        self.visited_scenarios.add(villain_key)
        self.iterations_completed += 1
        
        # Update heads-up stacks if in heads-up mode
        if heads_up_mode:
            self.hero_stack += result['hero_stack_change']
            self.villain_stack -= result['hero_stack_change']
            self.hands_played += 1
        
        # Record scenario
        scenario_record = {
            **scenario,
            'hero_action': hero_action,
            'villain_action': villain_action,
            'result': result['result'],
            'hero_stack_change': result['hero_stack_change'],
            'iteration': self.iterations_completed
        }
        self.scenarios_recorded.append(scenario_record)
        
        return scenario_record
    
    def _calculate_action_values(self, scenario, available_actions, opponent_action, is_hero=True):
        """Calculate expected values for each available action."""
        values = {}
        
        player_cards = scenario['hero_cards'] if is_hero else scenario['villain_cards']
        opponent_cards = scenario['villain_cards'] if is_hero else scenario['hero_cards']
        player_stack = scenario['hero_stack_bb'] if is_hero else scenario['villain_stack_bb']
        opponent_stack = scenario['villain_stack_bb'] if is_hero else scenario['hero_stack_bb']
        
        for action in available_actions:
            # Simulate result for this action
            if is_hero:
                result = simulate_preflop_showdown(
                    player_cards, opponent_cards, action, opponent_action,
                    player_stack, opponent_stack, 1.5
                )
                values[action] = result['hero_stack_change']
            else:
                result = simulate_preflop_showdown(
                    opponent_cards, player_cards, opponent_action, action,
                    opponent_stack, player_stack, 1.5
                )
                values[action] = -result['hero_stack_change']  # Villain's perspective
        
        return values
    
    def train(self, num_iterations=1000, heads_up_mode=False, save_interval=None):
        """
        Run CFR training for specified number of iterations.
        
        Args:
            num_iterations: Number of training iterations
            heads_up_mode: Whether to use heads-up match mode
            save_interval: Save checkpoint every N iterations (optional)
            
        Returns:
            dict: Training summary
        """
        start_time = time.time()
        self.logger.info("🚀 Starting Simplified CFR Training")
        self.logger.info(f"   🎲 Iterations: {num_iterations}")
        self.logger.info(f"   🎮 Heads-up mode: {heads_up_mode}")
        
        initial_iterations = self.iterations_completed
        
        for i in range(num_iterations):
            # Check if heads-up match is over
            if heads_up_mode and (self.hero_stack <= 0 or self.villain_stack <= 0):
                self.logger.info(f"🏁 Heads-up match completed after {self.hands_played} hands")
                if self.hero_stack <= 0:
                    self.logger.info("   Villain wins!")
                else:
                    self.logger.info("   Hero wins!")
                break
            
            # Simulate one scenario
            result = self.simulate_scenario(heads_up_mode=heads_up_mode)
            
            # Periodic logging
            if (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                rate = (i + 1) / (elapsed_time / 60)  # iterations per minute
                coverage = get_scenario_coverage_stats(self.visited_scenarios)
                
                self.logger.info(f"Iteration {i + 1}: "
                               f"{len(self.visited_scenarios)} scenarios, "
                               f"coverage={coverage['coverage_percent']:.1f}%, "
                               f"rate={rate:.1f}/min")
                
                if heads_up_mode:
                    self.logger.info(f"   Stacks: Hero={self.hero_stack}bb, Villain={self.villain_stack}bb")
            
            # Save checkpoint if requested
            if save_interval and (i + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.iterations_completed}")
        
        training_time = time.time() - start_time
        coverage = get_scenario_coverage_stats(self.visited_scenarios)
        
        summary = {
            "iterations_completed": self.iterations_completed - initial_iterations,
            "total_iterations": self.iterations_completed,
            "training_time_minutes": training_time / 60,
            "scenarios_visited": len(self.visited_scenarios),
            "coverage_stats": coverage,
            "hands_played": self.hands_played if heads_up_mode else None,
            "final_stacks": {"hero": self.hero_stack, "villain": self.villain_stack} if heads_up_mode else None
        }
        
        self.logger.info("🎉 Training completed!")
        self.logger.info(f"   ⏱️  Time: {training_time/60:.1f} minutes")
        self.logger.info(f"   📊 Scenarios visited: {len(self.visited_scenarios)}")
        self.logger.info(f"   🎯 Coverage: {coverage['coverage_percent']:.1f}%")
        
        return summary
    
    def save_checkpoint(self, filename):
        """Save training state to checkpoint file."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        filepath = checkpoint_dir / f"{filename}.pkl"
        
        state = {
            "hero_regrets": dict(self.hero_regrets),
            "hero_strategy_sums": dict(self.hero_strategy_sums),
            "villain_regrets": dict(self.villain_regrets),
            "villain_strategy_sums": dict(self.villain_strategy_sums),
            "scenario_visits": dict(self.scenario_visits),
            "visited_scenarios": self.visited_scenarios,
            "iterations_completed": self.iterations_completed,
            "scenarios_recorded": self.scenarios_recorded,
            "hero_stack": self.hero_stack,
            "villain_stack": self.villain_stack,
            "hands_played": self.hands_played,
            "epsilon_exploration": self.epsilon_exploration,
            "min_visit_threshold": self.min_visit_threshold,
            "starting_stack_bb": self.starting_stack_bb
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"💾 Checkpoint saved: {filepath}")
        return filepath
    
    def load_checkpoint(self, filepath):
        """Load training state from checkpoint file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.hero_regrets = defaultdict(lambda: defaultdict(float), state["hero_regrets"])
        self.hero_strategy_sums = defaultdict(lambda: defaultdict(float), state["hero_strategy_sums"])
        self.villain_regrets = defaultdict(lambda: defaultdict(float), state["villain_regrets"])
        self.villain_strategy_sums = defaultdict(lambda: defaultdict(float), state["villain_strategy_sums"])
        self.scenario_visits = defaultdict(int, state["scenario_visits"])
        self.visited_scenarios = state["visited_scenarios"]
        self.iterations_completed = state["iterations_completed"]
        self.scenarios_recorded = state["scenarios_recorded"]
        self.hero_stack = state["hero_stack"]
        self.villain_stack = state["villain_stack"]
        self.hands_played = state["hands_played"]
        
        self.logger.info(f"📂 Checkpoint loaded: {filepath}")
        self.logger.info(f"   Iterations: {self.iterations_completed}")
        self.logger.info(f"   Scenarios: {len(self.visited_scenarios)}")
        
        return True
    
    def get_final_strategies(self):
        """Get final average strategies for both players."""
        hero_strategies = {}
        villain_strategies = {}
        
        # Calculate average strategies from strategy sums
        for scenario_key in self.visited_scenarios:
            if scenario_key in self.hero_strategy_sums:
                total_sum = sum(self.hero_strategy_sums[scenario_key].values())
                if total_sum > 0:
                    hero_strategies[scenario_key] = {
                        action: prob / total_sum 
                        for action, prob in self.hero_strategy_sums[scenario_key].items()
                    }
            
            if scenario_key in self.villain_strategy_sums:
                total_sum = sum(self.villain_strategy_sums[scenario_key].values())
                if total_sum > 0:
                    villain_strategies[scenario_key] = {
                        action: prob / total_sum 
                        for action, prob in self.villain_strategy_sums[scenario_key].items()
                    }
        
        return hero_strategies, villain_strategies
    
    def export_results(self, filename_prefix="simplified_cfr"):
        """Export training results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export scenarios
        scenarios_df = pd.DataFrame(self.scenarios_recorded)
        scenarios_file = f"{filename_prefix}_scenarios_{timestamp}.csv"
        scenarios_df.to_csv(scenarios_file, index=False)
        
        # Export strategies
        hero_strategies, villain_strategies = self.get_final_strategies()
        
        hero_df = pd.DataFrame([
            {"scenario": k, "action": action, "probability": prob}
            for k, v in hero_strategies.items()
            for action, prob in v.items()
        ])
        hero_file = f"{filename_prefix}_hero_strategies_{timestamp}.csv"
        hero_df.to_csv(hero_file, index=False)
        
        villain_df = pd.DataFrame([
            {"scenario": k, "action": action, "probability": prob}
            for k, v in villain_strategies.items()
            for action, prob in v.items()
        ])
        villain_file = f"{filename_prefix}_villain_strategies_{timestamp}.csv"
        villain_df.to_csv(villain_file, index=False)
        
        # Export coverage report
        coverage = get_scenario_coverage_stats(self.visited_scenarios)
        coverage_file = f"{filename_prefix}_coverage_{timestamp}.json"
        with open(coverage_file, 'w') as f:
            json.dump(coverage, f, indent=2)
        
        self.logger.info(f"📊 Results exported:")
        self.logger.info(f"   Scenarios: {scenarios_file}")
        self.logger.info(f"   Hero strategies: {hero_file}")
        self.logger.info(f"   Villain strategies: {villain_file}")
        self.logger.info(f"   Coverage report: {coverage_file}")
        
        return {
            "scenarios": scenarios_file,
            "hero_strategies": hero_file,
            "villain_strategies": villain_file,
            "coverage": coverage_file
        }


if __name__ == "__main__":
    print("🧪 Testing Simplified CFR Trainer")
    print("=" * 50)
    
    # Test basic training
    trainer = SimplifiedCFRTrainer(epsilon_exploration=0.3)
    summary = trainer.train(num_iterations=100)
    
    print(f"Training summary:")
    print(f"  Iterations: {summary['iterations_completed']}")
    print(f"  Time: {summary['training_time_minutes']:.1f} minutes")
    print(f"  Scenarios visited: {summary['scenarios_visited']}")
    print(f"  Coverage: {summary['coverage_stats']['coverage_percent']:.1f}%")
    
    # Test checkpoint save/load
    checkpoint_file = trainer.save_checkpoint("test_checkpoint")
    trainer2 = SimplifiedCFRTrainer()
    trainer2.load_checkpoint(checkpoint_file)
    
    print("✅ Simplified CFR trainer working correctly!")