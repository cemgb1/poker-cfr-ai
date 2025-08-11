# natural_game_cfr_trainer.py - Natural Monte Carlo CFR Training System

"""
Natural Game CFR Trainer for Poker Based on Monte Carlo Game Simulation

This module implements a new training system that uses natural Monte Carlo game simulation
instead of pre-defined scenarios. Key features:

1. Natural Game Simulation:
   - Deals random cards for both hero and villain
   - Randomizes position, stack sizes, blinds, and villain stack category
   - Lets models play out hands based on their learned strategies
   - Records actual scenarios that emerge in gameplay

2. Co-evolving Strategies:
   - Both hero and villain act using their own evolving strategies
   - No hardcoded opponent behavior
   - Realistic opponent modeling through mutual learning
   - Regret updates for both players based on actual gameplay

3. Multi-step Betting Sequences:
   - Handles preflop betting sequences (extendable to postflop)
   - Supports opening and response decisions
   - Proper game tree handling for complex betting patterns

4. Epsilon-greedy Exploration:
   - Tracks visit count per state-action pair
   - Forced exploration ensures rare/important scenarios get visited
   - Balances exploitation vs exploration

5. Natural Scenario Recording:
   - Records hand_category, position, stack_depth, blinds_level, villain_stack_category
   - Tracks opponent_action, is_3bet, full action history, and payoffs
   - All scenarios emerge naturally from gameplay

Classes:
- NaturalGameCFRTrainer: Main trainer implementing natural Monte Carlo CFR
"""

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import (
    PREFLOP_HAND_RANGES, STACK_CATEGORIES, ACTIONS,
    cards_to_str, simulate_enhanced_showdown
)
from treys import Card, Deck, Evaluator
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import time
import pickle
import json
from pathlib import Path
from datetime import datetime


class NaturalGameCFRTrainer(EnhancedCFRTrainer):
    """
    Natural Game CFR Trainer using Monte Carlo game simulation.
    
    Instead of training on pre-defined scenarios, this trainer:
    1. Simulates natural poker games with random cards and conditions
    2. Records scenarios that emerge naturally during gameplay
    3. Applies CFR updates to both hero and villain strategies
    4. Allows strategies to co-evolve through self-play
    """
    
    def __init__(self, enable_pruning=True, regret_pruning_threshold=-300.0,
                 strategy_pruning_threshold=0.001, tournament_survival_penalty=0.6,
                 epsilon_exploration=0.1, min_visit_threshold=5):
        """
        Initialize Natural Game CFR Trainer.
        
        Args:
            enable_pruning: Enable CFR pruning techniques
            regret_pruning_threshold: Threshold for regret-based pruning
            strategy_pruning_threshold: Threshold for strategy pruning  
            tournament_survival_penalty: Factor to scale tournament bust penalties
            epsilon_exploration: Probability of forced exploration
            min_visit_threshold: Minimum visits before considering scenario trained
        """
        # Initialize base trainer without pre-defined scenarios
        super().__init__(
            scenarios=None,  # No pre-defined scenarios
            enable_pruning=enable_pruning,
            regret_pruning_threshold=regret_pruning_threshold,
            strategy_pruning_threshold=strategy_pruning_threshold,
            tournament_survival_penalty=tournament_survival_penalty
        )
        
        # Natural game simulation parameters
        self.epsilon_exploration = epsilon_exploration
        self.min_visit_threshold = min_visit_threshold
        
        # Visit tracking for epsilon-greedy exploration
        self.state_action_visits = defaultdict(lambda: defaultdict(int))
        self.scenario_visits = defaultdict(int)
        
        # Natural scenario recording
        self.natural_scenarios = []
        self.natural_scenario_counter = Counter()
        
        # Villain strategy tracking (separate from hero)
        self.villain_regret_sum = defaultdict(lambda: defaultdict(float))
        self.villain_strategy_sum = defaultdict(lambda: defaultdict(float))
        
        # Game history tracking
        self.game_history = []
        self.hand_outcomes = []
        
        # Performance metrics for natural training
        self.natural_metrics = {
            'games_played': 0,
            'scenarios_discovered': 0,
            'unique_scenarios': 0,
            'hero_win_rate': 0.0,
            'villain_win_rate': 0.0,
            'avg_pot_size': 0.0,
            'exploration_rate': 0.0
        }
        
        print(f"🎲 Natural Game CFR Trainer Initialized!")
        print(f"   🎯 Epsilon exploration: {self.epsilon_exploration}")
        print(f"   📊 Min visit threshold: {self.min_visit_threshold}")
        print(f"   🏆 Tournament survival penalty: {self.tournament_survival_penalty}")
        print(f"   ✂️ Pruning enabled: {self.enable_pruning}")
    
    def classify_hand_category(self, cards):
        """
        Classify a hand into one of the predefined hand categories.
        
        Args:
            cards: List of Card objects [card1, card2]
            
        Returns:
            str: Hand category from PREFLOP_HAND_RANGES
        """
        if len(cards) != 2:
            return "trash"
        
        # Convert cards to string representation for lookup
        hand_str = cards_to_str(cards)
        
        # Try both suited and offsuit versions
        for category, hands in PREFLOP_HAND_RANGES.items():
            if hand_str in hands:
                return category
        
        # If not found in any category, classify as trash
        return "trash"
    
    def classify_stack_category(self, stack_bb):
        """
        Classify stack size into category.
        
        Args:
            stack_bb: Stack size in big blinds
            
        Returns:
            str: Stack category
        """
        for category, (min_stack, max_stack) in STACK_CATEGORIES.items():
            if min_stack <= stack_bb <= max_stack:
                return category
        
        # Handle edge cases
        if stack_bb < 8:
            return "ultra_short"
        elif stack_bb > 200:
            return "very_deep"
        else:
            return "medium"
    
    def generate_random_game_state(self):
        """
        Generate a random game state for natural Monte Carlo simulation.
        
        Returns:
            dict: Game state with random cards, positions, stacks, blinds
        """
        # Deal random cards
        deck = Deck()
        hero_cards = deck.draw(2)
        villain_cards = deck.draw(2)
        
        # Random position assignment
        hero_position = random.choice(["BTN", "BB"])
        villain_position = "BB" if hero_position == "BTN" else "BTN"
        
        # Random stack sizes (can be different for hero and villain)
        hero_stack_bb = random.randint(8, 200)
        villain_stack_bb = random.randint(8, 200)
        
        # Random blinds level
        blinds_level = random.choice(["low", "medium", "high"])
        
        # Classify hands and stacks
        hero_hand_category = self.classify_hand_category(hero_cards)
        villain_hand_category = self.classify_hand_category(villain_cards)
        hero_stack_category = self.classify_stack_category(hero_stack_bb)
        villain_stack_category = self.classify_stack_category(villain_stack_bb)
        
        return {
            'hero_cards': hero_cards,
            'villain_cards': villain_cards,
            'hero_cards_str': cards_to_str(hero_cards),
            'villain_cards_str': cards_to_str(villain_cards),
            'hero_position': hero_position,
            'villain_position': villain_position,
            'hero_stack_bb': hero_stack_bb,
            'villain_stack_bb': villain_stack_bb,
            'hero_hand_category': hero_hand_category,
            'villain_hand_category': villain_hand_category,
            'hero_stack_category': hero_stack_category,
            'villain_stack_category': villain_stack_category,
            'blinds_level': blinds_level,
            'pot_bb': 1.5,  # Small blind + big blind
            'to_act': hero_position,  # BTN acts first preflop
            'action_history': [],
            'is_3bet': False,
            'betting_round': 'preflop'
        }
    
    def get_available_actions_for_game_state(self, game_state, is_hero=True):
        """
        Get available actions based on current game state.
        
        Args:
            game_state: Current game state dictionary
            is_hero: Whether actions are for hero (True) or villain (False)
            
        Returns:
            list: Available actions for the current player
        """
        stack_bb = game_state['hero_stack_bb'] if is_hero else game_state['villain_stack_bb']
        pot_bb = game_state['pot_bb']
        action_history = game_state['action_history']
        
        # If no previous action, can check or bet
        if not action_history:
            # First to act can check or bet
            actions = ['fold', 'call_small', 'raise_small', 'raise_mid', 'raise_high']
            
            # Filter by stack size
            if stack_bb <= 15:
                actions = ['fold', 'call_small', 'raise_small']  # Short stack options
            elif stack_bb <= 30:
                actions = ['fold', 'call_small', 'raise_small', 'raise_mid']
            
            return actions
        
        # If there was a previous action, respond accordingly
        last_action = action_history[-1]['action']
        
        if last_action in ['call_small', 'call_mid', 'call_high']:
            # Can check, bet, or fold
            actions = ['fold', 'call_small', 'raise_small', 'raise_mid', 'raise_high']
        elif last_action in ['raise_small', 'raise_mid', 'raise_high']:
            # Facing a raise - can call, re-raise, or fold
            actions = ['fold', 'call_small', 'call_mid', 'call_high', 'raise_small', 'raise_mid', 'raise_high']
        else:
            # Default action set
            actions = ['fold', 'call_small', 'call_mid', 'call_high', 'raise_small', 'raise_mid', 'raise_high']
        
        # Filter by stack size for short stacks
        if stack_bb <= 15:
            actions = [a for a in actions if a in ['fold', 'call_small', 'raise_small']]
        elif stack_bb <= 30:
            actions = [a for a in actions if a not in ['call_high', 'raise_high']]
        
        return actions
    
    def get_scenario_key_from_game_state(self, game_state, is_hero=True):
        """
        Generate scenario key from current game state.
        
        Args:
            game_state: Current game state
            is_hero: Whether key is for hero (True) or villain (False)
            
        Returns:
            str: Scenario key for strategy lookup
        """
        if is_hero:
            hand_category = game_state['hero_hand_category']
            position = game_state['hero_position']
            stack_category = game_state['hero_stack_category']
        else:
            hand_category = game_state['villain_hand_category']
            position = game_state['villain_position']
            stack_category = game_state['villain_stack_category']
        
        blinds_level = game_state['blinds_level']
        
        return f"{hand_category}|{position}|{stack_category}|{blinds_level}"
    
    def should_explore(self, scenario_key, action, is_hero=True):
        """
        Determine if we should explore this action (epsilon-greedy).
        
        Args:
            scenario_key: Current scenario key
            action: Action being considered
            is_hero: Whether this is for hero (True) or villain (False)
            
        Returns:
            bool: True if should explore (force random action)
        """
        if random.random() < self.epsilon_exploration:
            return True
        
        # Check if this state-action pair needs more visits
        visits = self.state_action_visits[scenario_key][action]
        if visits < self.min_visit_threshold:
            return True
        
        return False
    
    def get_player_action(self, game_state, is_hero=True, force_action=None):
        """
        Get action for a player (hero or villain) using their learned strategy.
        
        Args:
            game_state: Current game state
            is_hero: Whether getting action for hero (True) or villain (False)  
            force_action: Force specific action (for exploration)
            
        Returns:
            str: Chosen action
        """
        scenario_key = self.get_scenario_key_from_game_state(game_state, is_hero)
        available_actions = self.get_available_actions_for_game_state(game_state, is_hero)
        
        if force_action and force_action in available_actions:
            return force_action
        
        # Check if we should explore
        if self.should_explore(scenario_key, None, is_hero):
            return random.choice(available_actions)
        
        # Use learned strategy
        if is_hero:
            strategy = self.get_strategy(scenario_key, available_actions)
        else:
            # Get villain strategy
            strategy = self.get_villain_strategy(scenario_key, available_actions)
        
        # Sample action from strategy (using safe sampling)
        return self.safe_sample_action(strategy, available_actions)
    
    def safe_sample_action(self, strategy, available_actions):
        """
        Safely sample action from strategy probabilities, handling action mismatches.
        
        Args:
            strategy: Strategy probabilities dictionary
            available_actions: List of currently available actions
            
        Returns:
            str: Chosen action from available actions
        """
        # Only consider actions that are both in strategy AND available
        valid_actions = [action for action in available_actions if action in strategy]
        
        if not valid_actions:
            # If no valid actions in strategy, use uniform over available actions
            return random.choice(available_actions)
        
        # Get probabilities for valid actions only
        probs = [strategy.get(action, 0.0) for action in valid_actions]
        
        # Normalize probabilities to ensure they sum to 1
        prob_sum = sum(probs)
        if prob_sum == 0:
            # All probabilities are 0, use uniform distribution over valid actions
            probs = [1.0 / len(valid_actions)] * len(valid_actions)
        else:
            # Normalize to sum to 1
            probs = [p / prob_sum for p in probs]
        
        chosen_idx = np.random.choice(len(valid_actions), p=probs)
        return valid_actions[chosen_idx]
    
    def get_villain_strategy(self, scenario_key, available_actions):
        """
        Get villain strategy using regret matching (separate from hero).
        
        Args:
            scenario_key: Scenario identifier
            available_actions: Available actions for villain
            
        Returns:
            dict: Strategy probabilities for villain
        """
        if scenario_key not in self.villain_regret_sum:
            self.villain_regret_sum[scenario_key] = defaultdict(float)
        
        regrets = self.villain_regret_sum[scenario_key]
        
        # Calculate positive regrets
        action_regrets = [max(regrets.get(action, 0.0), 0.0) for action in available_actions]
        regret_sum = sum(action_regrets)
        
        if regret_sum > 0:
            strategy_probs = [regret / regret_sum for regret in action_regrets]
        else:
            # Uniform strategy if no positive regrets
            strategy_probs = [1.0 / len(available_actions)] * len(available_actions)
        
        return {action: prob for action, prob in zip(available_actions, strategy_probs)}
    
    def monte_carlo_game_simulation(self):
        """
        Simulate a complete poker hand using Monte Carlo approach.
        
        Both players act according to their learned strategies.
        Records natural scenarios that emerge during gameplay.
        
        Returns:
            dict: Complete game simulation result
        """
        # Generate random game state
        game_state = self.generate_random_game_state()
        
        # Track if this is exploration
        exploration_used = False
        
        # Simulate preflop betting sequence
        betting_complete = False
        max_actions = 4  # Prevent infinite betting loops
        action_count = 0
        
        while not betting_complete and action_count < max_actions:
            # Determine who acts (alternates, BTN acts first)
            is_hero_turn = (action_count % 2 == 0 and game_state['hero_position'] == 'BTN') or \
                          (action_count % 2 == 1 and game_state['hero_position'] == 'BB')
            
            # Get action from current player
            if is_hero_turn:
                action = self.get_player_action(game_state, is_hero=True)
                actor = 'hero'
            else:
                action = self.get_player_action(game_state, is_hero=False)
                actor = 'villain'
            
            # Record action in history
            game_state['action_history'].append({
                'actor': actor,
                'action': action,
                'is_hero': is_hero_turn
            })
            
            # Update game state based on action
            self.update_game_state_with_action(game_state, action, is_hero_turn)
            
            # Check if betting round is complete
            betting_complete = self.is_betting_complete(game_state)
            action_count += 1
        
        # Calculate final payoffs
        payoff_result = self.calculate_game_payoffs(game_state)
        
        # Record natural scenario that emerged
        natural_scenario = self.record_natural_scenario(game_state, payoff_result)
        
        # Update strategies for both players
        self.update_strategies_from_game(game_state, payoff_result)
        
        # Track metrics
        self.natural_metrics['games_played'] += 1
        
        return {
            'game_state': game_state,
            'natural_scenario': natural_scenario,
            'payoff_result': payoff_result,
            'exploration_used': exploration_used
        }
    
    def update_game_state_with_action(self, game_state, action, is_hero):
        """
        Update game state based on player action.
        
        Args:
            game_state: Current game state (modified in place)
            action: Action taken
            is_hero: Whether action was by hero
        """
        # Update pot based on action
        stack_bb = game_state['hero_stack_bb'] if is_hero else game_state['villain_stack_bb']
        
        if action == 'fold':
            game_state['folded'] = True
            game_state['folder'] = 'hero' if is_hero else 'villain'
        elif action in ['call_small', 'call_mid', 'call_high']:
            # Add call amount to pot (simplified)
            call_amount = min(stack_bb * 0.1, stack_bb)  # Call sizing logic
            game_state['pot_bb'] += call_amount
            if is_hero:
                game_state['hero_stack_bb'] -= call_amount
            else:
                game_state['villain_stack_bb'] -= call_amount
        elif action in ['raise_small', 'raise_mid', 'raise_high']:
            # Add raise amount to pot
            if action == 'raise_small':
                raise_amount = min(game_state['pot_bb'] * 2.5, stack_bb)
            elif action == 'raise_mid':
                raise_amount = min(game_state['pot_bb'] * 3.0, stack_bb)
            else:  # raise_high
                raise_amount = min(game_state['pot_bb'] * 4.0, stack_bb)
            
            game_state['pot_bb'] += raise_amount
            if is_hero:
                game_state['hero_stack_bb'] -= raise_amount
            else:
                game_state['villain_stack_bb'] -= raise_amount
            
            # Check if this makes it a 3-bet
            raise_count = sum(1 for a in game_state['action_history'] 
                            if 'raise' in a['action'])
            if raise_count >= 2:
                game_state['is_3bet'] = True
    
    def is_betting_complete(self, game_state):
        """
        Check if betting round is complete.
        
        Args:
            game_state: Current game state
            
        Returns:
            bool: True if betting is complete
        """
        # If someone folded, betting is complete
        if game_state.get('folded', False):
            return True
        
        # If no actions yet, continue
        if not game_state['action_history']:
            return False
        
        # Need at least one action from each player
        if len(game_state['action_history']) < 2:
            return False
        
        # Check if both players have acted and actions are compatible
        last_action = game_state['action_history'][-1]['action']
        second_last_action = game_state['action_history'][-2]['action']
        
        # If both called or checked, betting is complete
        if ('call' in last_action and 'call' in second_last_action):
            return True
        
        # If facing a raise and opponent called, complete
        if ('raise' in second_last_action and 'call' in last_action):
            return True
        
        return False
    
    def calculate_game_payoffs(self, game_state):
        """
        Calculate final payoffs for the game.
        
        Args:
            game_state: Final game state
            
        Returns:
            dict: Payoff results for both players
        """
        # If someone folded, other player wins the pot
        if game_state.get('folded', False):
            folder = game_state.get('folder', 'hero')
            if folder == 'hero':
                return {
                    'hero_payoff': -game_state['pot_bb'] / 2,  # Lost the pot
                    'villain_payoff': game_state['pot_bb'] / 2,  # Won the pot
                    'hero_won': False,
                    'showdown': False
                }
            else:
                return {
                    'hero_payoff': game_state['pot_bb'] / 2,
                    'villain_payoff': -game_state['pot_bb'] / 2,
                    'hero_won': True,
                    'showdown': False
                }
        
        # If no fold, simulate showdown
        hero_cards = game_state['hero_cards']
        villain_cards = game_state['villain_cards']
        
        # Use simplified equity calculation for now
        hero_equity = self.estimate_hand_equity(hero_cards, villain_cards)
        
        # Determine winner based on equity (with some randomness)
        if random.random() < hero_equity:
            return {
                'hero_payoff': game_state['pot_bb'] / 2,
                'villain_payoff': -game_state['pot_bb'] / 2,
                'hero_won': True,
                'showdown': True,
                'hero_equity': hero_equity
            }
        else:
            return {
                'hero_payoff': -game_state['pot_bb'] / 2,
                'villain_payoff': game_state['pot_bb'] / 2,
                'hero_won': False,
                'showdown': True,
                'hero_equity': hero_equity
            }
    
    def estimate_hand_equity(self, hero_cards, villain_cards, simulations=100):
        """
        Estimate hand equity using Monte Carlo simulation.
        
        Args:
            hero_cards: Hero's hole cards
            villain_cards: Villain's hole cards
            simulations: Number of simulations to run
            
        Returns:
            float: Hero's equity (0.0 to 1.0)
        """
        wins = 0
        ties = 0
        
        for _ in range(simulations):
            # Create deck and remove known cards
            deck = Deck()
            for card in hero_cards + villain_cards:
                if card in deck.cards:
                    deck.cards.remove(card)
            
            # Deal board
            board = deck.draw(5)
            
            # Evaluate hands
            evaluator = Evaluator()
            hero_score = evaluator.evaluate(hero_cards, board)
            villain_score = evaluator.evaluate(villain_cards, board)
            
            if hero_score < villain_score:  # Lower score wins in treys
                wins += 1
            elif hero_score == villain_score:
                ties += 1
        
        return (wins + ties * 0.5) / simulations
    
    def record_natural_scenario(self, game_state, payoff_result):
        """
        Record a natural scenario that emerged during gameplay.
        
        Args:
            game_state: Final game state
            payoff_result: Payoff results
            
        Returns:
            dict: Recorded natural scenario
        """
        # Determine opponent action (last action by opponent)
        opponent_action = "unknown"
        for action_info in reversed(game_state['action_history']):
            if not action_info['is_hero']:  # Villain action
                opponent_action = action_info['action']
                break
        
        natural_scenario = {
            'hand_category': game_state['hero_hand_category'],
            'position': game_state['hero_position'],
            'stack_depth': game_state['hero_stack_category'],
            'blinds_level': game_state['blinds_level'],
            'villain_stack_category': game_state['villain_stack_category'],
            'opponent_action': opponent_action,
            'is_3bet': game_state['is_3bet'],
            'action_history': game_state['action_history'].copy(),
            'final_pot_bb': game_state['pot_bb'],
            'hero_payoff': payoff_result['hero_payoff'],
            'hero_won': payoff_result['hero_won'],
            'showdown': payoff_result.get('showdown', False),
            'hero_cards': game_state['hero_cards_str'],
            'villain_cards': game_state['villain_cards_str'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate scenario key for tracking
        scenario_key = self.get_scenario_key_from_game_state(game_state, is_hero=True)
        natural_scenario['scenario_key'] = scenario_key
        
        # Add to collections
        self.natural_scenarios.append(natural_scenario)
        self.natural_scenario_counter[scenario_key] += 1
        
        # Update unique scenarios count
        self.natural_metrics['scenarios_discovered'] += 1
        self.natural_metrics['unique_scenarios'] = len(self.natural_scenario_counter)
        
        return natural_scenario
    
    def update_strategies_from_game(self, game_state, payoff_result):
        """
        Update both hero and villain strategies based on game outcome.
        
        Args:
            game_state: Final game state
            payoff_result: Payoff results
        """
        # Update hero strategy
        hero_scenario_key = self.get_scenario_key_from_game_state(game_state, is_hero=True)
        hero_available_actions = self.get_available_actions_for_game_state(game_state, is_hero=True)
        
        # Find hero's action
        hero_action = None
        for action_info in game_state['action_history']:
            if action_info['is_hero']:
                hero_action = action_info['action']
                break
        
        if hero_action and hero_action in hero_available_actions:
            # Get hero strategy and update regrets
            hero_strategy = self.get_strategy(hero_scenario_key, hero_available_actions)
            self.update_enhanced_regrets(
                hero_scenario_key, hero_action, hero_strategy, 
                {'payoff': payoff_result['hero_payoff']}, hero_available_actions
            )
            
            # Update strategy sum
            if hero_scenario_key not in self.strategy_sum:
                self.strategy_sum[hero_scenario_key] = defaultdict(float)
            for action in hero_available_actions:
                self.strategy_sum[hero_scenario_key][action] += hero_strategy.get(action, 0.0)
        
        # Update villain strategy
        villain_scenario_key = self.get_scenario_key_from_game_state(game_state, is_hero=False)
        villain_available_actions = self.get_available_actions_for_game_state(game_state, is_hero=False)
        
        # Find villain's action
        villain_action = None
        for action_info in game_state['action_history']:
            if not action_info['is_hero']:
                villain_action = action_info['action']
                break
        
        if villain_action and villain_action in villain_available_actions:
            # Get villain strategy and update regrets
            villain_strategy = self.get_villain_strategy(villain_scenario_key, villain_available_actions)
            self.update_villain_regrets(
                villain_scenario_key, villain_action, villain_strategy,
                {'payoff': payoff_result['villain_payoff']}, villain_available_actions
            )
            
            # Update villain strategy sum
            if villain_scenario_key not in self.villain_strategy_sum:
                self.villain_strategy_sum[villain_scenario_key] = defaultdict(float)
            for action in villain_available_actions:
                self.villain_strategy_sum[villain_scenario_key][action] += villain_strategy.get(action, 0.0)
        
        # Update visit counts for exploration (only for valid actions)
        if hero_action and hero_action in hero_available_actions:
            self.state_action_visits[hero_scenario_key][hero_action] += 1
        if villain_action and villain_action in villain_available_actions:
            self.state_action_visits[villain_scenario_key][villain_action] += 1
    
    def update_villain_regrets(self, scenario_key, action_taken, strategy, 
                             payoff_result, available_actions):
        """
        Update villain regrets (similar to hero regret updates).
        
        Args:
            scenario_key: Scenario identifier
            action_taken: Action that was taken
            strategy: Strategy probabilities used
            payoff_result: Result of the action
            available_actions: Actions that were available
        """
        if scenario_key not in self.villain_regret_sum:
            self.villain_regret_sum[scenario_key] = defaultdict(float)
        
        actual_payoff = payoff_result['payoff']
        
        # Estimate counterfactual payoffs for other actions
        for action in available_actions:
            if action == action_taken:
                # No regret for chosen action
                continue
            else:
                # Estimate what would have happened with different action
                estimated_payoff = self.estimate_counterfactual_payoff(
                    action, payoff_result, available_actions
                )
                
                # Regret = what could have got - what actually got
                regret = estimated_payoff - actual_payoff
                self.villain_regret_sum[scenario_key][action] += regret
    
    def train(self, n_games=10000, save_interval=1000, log_interval=100):
        """
        Main training loop for natural game CFR.
        
        Args:
            n_games: Number of games to simulate
            save_interval: How often to save progress
            log_interval: How often to log progress
            
        Returns:
            dict: Training statistics
        """
        print(f"🚀 Starting Natural Game CFR Training")
        print(f"   🎲 Games to simulate: {n_games:,}")
        print(f"   💾 Save interval: every {save_interval} games")
        print(f"   📝 Log interval: every {log_interval} games")
        print("=" * 60)
        
        training_start_time = time.time()
        
        for game_num in range(n_games):
            # Simulate one game
            game_result = self.monte_carlo_game_simulation()
            
            # Log progress
            if (game_num + 1) % log_interval == 0:
                self.log_training_progress(game_num + 1, training_start_time)
            
            # Save progress
            if (game_num + 1) % save_interval == 0:
                self.save_training_state(f"natural_cfr_checkpoint_{game_num + 1}.pkl")
        
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print(f"\n🎉 Natural Game CFR Training Complete!")
        print(f"   ⏱️  Total time: {total_training_time/60:.1f} minutes")
        print(f"   🎲 Games played: {self.natural_metrics['games_played']:,}")
        print(f"   📊 Unique scenarios: {self.natural_metrics['unique_scenarios']}")
        print(f"   🎯 Total scenarios recorded: {len(self.natural_scenarios)}")
        
        return {
            'games_played': self.natural_metrics['games_played'],
            'unique_scenarios': self.natural_metrics['unique_scenarios'],
            'total_training_time': total_training_time,
            'natural_scenarios': len(self.natural_scenarios),
            'hero_strategy_scenarios': len(self.strategy_sum),
            'villain_strategy_scenarios': len(self.villain_strategy_sum)
        }
    
    def log_training_progress(self, games_completed, start_time):
        """
        Log training progress with key metrics.
        
        Args:
            games_completed: Number of games completed
            start_time: Training start timestamp
        """
        elapsed_time = time.time() - start_time
        games_per_minute = (games_completed / elapsed_time) * 60
        
        # Calculate win rates
        if self.natural_scenarios:
            hero_wins = sum(1 for s in self.natural_scenarios if s['hero_won'])
            hero_win_rate = hero_wins / len(self.natural_scenarios)
        else:
            hero_win_rate = 0.0
        
        print(f"Game {games_completed:6,}: "
              f"{self.natural_metrics['unique_scenarios']:3d} scenarios, "
              f"hero_wr={hero_win_rate:.2f}, "
              f"rate={games_per_minute:.1f}/min")
    
    def save_training_state(self, filename):
        """
        Save complete training state to file.
        
        Args:
            filename: File to save to
        """
        save_data = {
            'hero_regret_sum': dict(self.regret_sum),
            'hero_strategy_sum': dict(self.strategy_sum),
            'villain_regret_sum': dict(self.villain_regret_sum),
            'villain_strategy_sum': dict(self.villain_strategy_sum),
            'natural_scenarios': self.natural_scenarios,
            'natural_scenario_counter': dict(self.natural_scenario_counter),
            'state_action_visits': dict(self.state_action_visits),
            'natural_metrics': self.natural_metrics,
            'training_parameters': {
                'epsilon_exploration': self.epsilon_exploration,
                'min_visit_threshold': self.min_visit_threshold,
                'tournament_survival_penalty': self.tournament_survival_penalty
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Training state saved to {filename}")
    
    def load_training_state(self, filename):
        """
        Load training state from file.
        
        Args:
            filename: File to load from
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore state
            self.regret_sum = defaultdict(lambda: defaultdict(float), save_data['hero_regret_sum'])
            self.strategy_sum = defaultdict(lambda: defaultdict(float), save_data['hero_strategy_sum'])
            self.villain_regret_sum = defaultdict(lambda: defaultdict(float), save_data['villain_regret_sum'])
            self.villain_strategy_sum = defaultdict(lambda: defaultdict(float), save_data['villain_strategy_sum'])
            self.natural_scenarios = save_data['natural_scenarios']
            self.natural_scenario_counter = Counter(save_data['natural_scenario_counter'])
            self.state_action_visits = defaultdict(lambda: defaultdict(int), save_data['state_action_visits'])
            self.natural_metrics = save_data['natural_metrics']
            
            print(f"✅ Training state loaded from {filename}")
            print(f"   🎲 Games played: {self.natural_metrics['games_played']:,}")
            print(f"   📊 Unique scenarios: {self.natural_metrics['unique_scenarios']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load training state: {e}")
            return False
    
    def export_natural_scenarios_csv(self, filename="natural_scenarios.csv"):
        """
        Export all natural scenarios to CSV for analysis.
        
        Args:
            filename: Output CSV filename
        """
        if not self.natural_scenarios:
            print("❌ No natural scenarios to export")
            return
        
        df = pd.DataFrame(self.natural_scenarios)
        df.to_csv(filename, index=False)
        
        print(f"📊 Exported {len(self.natural_scenarios)} natural scenarios to {filename}")
        
        # Show summary statistics
        print(f"\n📈 NATURAL SCENARIOS SUMMARY:")
        print(f"Total scenarios: {len(self.natural_scenarios)}")
        print(f"Unique scenario keys: {len(self.natural_scenario_counter)}")
        print(f"Hero win rate: {sum(s['hero_won'] for s in self.natural_scenarios) / len(self.natural_scenarios):.3f}")
        print(f"Showdown rate: {sum(s['showdown'] for s in self.natural_scenarios) / len(self.natural_scenarios):.3f}")
        print(f"3-bet rate: {sum(s['is_3bet'] for s in self.natural_scenarios) / len(self.natural_scenarios):.3f}")
    
    def export_strategies_csv(self, filename="natural_strategies.csv"):
        """
        Export learned strategies to CSV.
        
        Args:
            filename: Output CSV filename
        """
        # Export hero strategies using parent method
        hero_df = self.export_strategies_to_csv(f"hero_{filename}")
        
        # Export villain strategies
        if self.villain_strategy_sum:
            villain_data = []
            
            for scenario_key, strategy_counts in self.villain_strategy_sum.items():
                if sum(strategy_counts.values()) > 0:
                    parts = scenario_key.split("|")
                    if len(parts) >= 4:
                        hand_category, position, stack_category, blinds_level = parts[:4]
                    else:
                        continue
                    
                    # Calculate normalized probabilities
                    total_count = sum(strategy_counts.values())
                    action_probs = {}
                    for action_name in ACTIONS.keys():
                        action_probs[f"{action_name}_prob"] = strategy_counts.get(action_name, 0.0) / total_count
                    
                    # Find best action
                    best_action = max(strategy_counts.items(), key=lambda x: x[1])[0]
                    confidence = max(strategy_counts.values()) / total_count
                    
                    row = {
                        'scenario_key': scenario_key,
                        'hand_category': hand_category,
                        'position': position,
                        'stack_depth': stack_category,
                        'blinds_level': blinds_level,
                        'training_games': self.natural_scenario_counter.get(scenario_key, 0),
                        'best_action': best_action.upper(),
                        'confidence': round(confidence, 3),
                        **{k: round(v, 3) for k, v in action_probs.items()}
                    }
                    villain_data.append(row)
            
            if villain_data:
                villain_df = pd.DataFrame(villain_data)
                villain_df = villain_df.sort_values(['confidence', 'training_games'], ascending=[False, False])
                villain_df.to_csv(f"villain_{filename}", index=False)
                print(f"📊 Exported villain strategies to villain_{filename}")
        
        print(f"✅ Strategy export complete")


if __name__ == "__main__":
    print("Natural Game CFR Trainer - Ready for training!")
    
    # Example usage
    def demo_natural_training(n_games=1000):
        """Demo natural game CFR training."""
        print("🎲 Demo Natural Game CFR Training")
        
        trainer = NaturalGameCFRTrainer(
            epsilon_exploration=0.1,
            min_visit_threshold=5,
            tournament_survival_penalty=0.6
        )
        
        # Run training
        results = trainer.train(n_games=n_games, log_interval=100)
        
        # Export results
        trainer.export_natural_scenarios_csv("demo_natural_scenarios.csv")
        trainer.export_strategies_csv("demo_natural_strategies.csv")
        
        return trainer, results
    
    # Uncomment to run demo:
    # trainer, results = demo_natural_training()