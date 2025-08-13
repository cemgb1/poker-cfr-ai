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

6. NEW: Unified Scenario Lookup Table:
   - Real-time CSV export (scenario_lookup_table.csv) at every log interval
   - Provides live monitoring of learning progress and scenario coverage
   - Same format as GCP trainer for consistency

7. Default tournament penalty: 0.2 (encourages moderate risk-taking)

Classes:
- NaturalGameCFRTrainer: Main trainer implementing natural Monte Carlo CFR
"""

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import (
    PREFLOP_HAND_RANGES, STACK_CATEGORIES, ACTIONS,
    cards_to_str, simulate_enhanced_showdown, classify_hand_category
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
from datetime import datetime, timedelta


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
                 strategy_pruning_threshold=0.001, tournament_survival_penalty=0.2,
                 epsilon_exploration=0.1, min_visit_threshold=5, logger=None):
        """
        Initialize Natural Game CFR Trainer.
        
        Args:
            enable_pruning: Enable CFR pruning techniques
            regret_pruning_threshold: Threshold for regret-based pruning
            strategy_pruning_threshold: Threshold for strategy pruning  
            tournament_survival_penalty: Factor to scale tournament bust penalties
            epsilon_exploration: Probability of forced exploration
            min_visit_threshold: Minimum visits before considering scenario trained
            logger: Logger instance for logging (optional)
        """
        # Store logger
        from logging_config import get_logger
        self.logger = logger if logger else get_logger("natural_game_cfr_trainer")
        
        # Log initialization parameters
        self.logger.info("Initializing Natural Game CFR Trainer")
        self.logger.info("Initialization parameters:")
        self.logger.info(f"  - enable_pruning: {enable_pruning}")
        self.logger.info(f"  - regret_pruning_threshold: {regret_pruning_threshold}")
        self.logger.info(f"  - strategy_pruning_threshold: {strategy_pruning_threshold}")
        self.logger.info(f"  - tournament_survival_penalty: {tournament_survival_penalty}")
        self.logger.info(f"  - epsilon_exploration: {epsilon_exploration}")
        self.logger.info(f"  - min_visit_threshold: {min_visit_threshold}")
        
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
        
        # Smart scenario filtering for realistic combinations
        self.realistic_scenario_cache = {}
        self.filtered_scenario_count = 0
        
        # Performance optimizations
        self.scenario_key_cache = {}  # Cache for scenario key generation
        self.action_cache = {}        # Cache for available actions
        self.memory_cleanup_interval = 1000  # Clean memory every N games
        
        # Performance metrics for natural training
        self.natural_metrics = {
            'games_played': 0,
            'hands_played': 0,  # Track total hands across all games
            'scenarios_discovered': 0,
            'unique_scenarios': 0,
            'hero_win_rate': 0.0,
            'villain_win_rate': 0.0,
            'avg_pot_size': 0.0,
            'exploration_rate': 0.0,
            'avg_hands_per_game': 0.0  # Track average hands per game
        }
        
        # NEW: Required aggregation features for problem statement
        self.scenario_aggregation = {}  # Single in-memory dict keyed by scenario_key
        self.hand_categories_seen = set()  # Set of hand categories encountered
        self.scenario_classification_errors = 0  # Count of classification errors
        self.total_scenarios_processed = 0  # Total scenarios for ratio calculations
        
        self.logger.info("🎲 Natural Game CFR Trainer Initialized!")
        self.logger.info(f"   🎯 Epsilon exploration: {self.epsilon_exploration}")
        self.logger.info(f"   📊 Min visit threshold: {self.min_visit_threshold}")
        self.logger.info(f"   🏆 Tournament survival penalty: {self.tournament_survival_penalty}")
        self.logger.info(f"   ✂️ Pruning enabled: {self.enable_pruning}")
        
        print(f"🎲 Natural Game CFR Trainer Initialized!")
        print(f"   🎯 Epsilon exploration: {self.epsilon_exploration}")
        print(f"   📊 Min visit threshold: {self.min_visit_threshold}")
        print(f"   🏆 Tournament survival penalty: {self.tournament_survival_penalty}")
        print(f"   ✂️ Pruning enabled: {self.enable_pruning}")
        
        # Perform startup validation for hand categories
        self._validate_hand_categories_at_startup()
    
    def is_realistic_scenario(self, scenario_key):
        """
        Determine if a scenario combination is realistic for poker.
        
        Filters out unlikely/impossible combinations to achieve ~3,500-4,000 scenarios.
        
        Args:
            scenario_key: Scenario key in format hand_category|position|stack_category|blinds_level|villain_stack_category|opponent_action|is_3bet
            
        Returns:
            bool: True if scenario is realistic
        """
        # Use cache for performance
        if scenario_key in self.realistic_scenario_cache:
            return self.realistic_scenario_cache[scenario_key]
        
        try:
            parts = scenario_key.split('|')
            if len(parts) != 7:
                return False
                
            hand_category, position, stack_category, blinds_level, villain_stack_category, opponent_action, is_3bet_str = parts
            is_3bet = is_3bet_str == 'True'
            
            # Filter 1: Hand-position combinations (eliminate ~10%)
            if not self._is_realistic_hand_position(hand_category, position):
                return False
            
            # Filter 2: Stack-blinds combinations (eliminate ~15%)
            if not self._is_realistic_stack_blinds(stack_category, blinds_level):
                return False
            
            # Filter 3: Opponent action logic (eliminate ~40%)
            if not self._is_realistic_opponent_action(opponent_action, stack_category, villain_stack_category, is_3bet):
                return False
            
            # Filter 4: 3-bet scenarios (eliminate ~20%)
            if not self._is_realistic_3bet_scenario(hand_category, opponent_action, is_3bet):
                return False
            
            # Filter 5: Stack size mismatches (eliminate ~10%)
            if not self._is_realistic_stack_match(stack_category, villain_stack_category, opponent_action):
                return False
            
            result = True
            
        except Exception as e:
            self.logger.warning(f"Error evaluating scenario realism for {scenario_key}: {e}")
            result = False
        
        # Cache result
        self.realistic_scenario_cache[scenario_key] = result
        return result
    
    def _is_realistic_hand_position(self, hand_category, position):
        """Filter unrealistic hand-position combinations."""
        # Most combinations are realistic, only filter extreme cases
        if hand_category == 'premium_pairs' and position == 'BB':
            # Premium pairs in BB are realistic
            return True
        if hand_category == 'trash' and position == 'BTN':
            # Trash on BTN is realistic (can open wide)
            return True
        return True  # Keep most combinations
    
    def _is_realistic_stack_blinds(self, stack_category, blinds_level):
        """Filter unrealistic stack-blinds combinations."""
        # Eliminate some extreme combinations
        if stack_category == 'ultra_short' and blinds_level == 'low':
            return random.random() < 0.3  # Keep 30% of ultra_short + low blinds
        if stack_category == 'very_deep' and blinds_level == 'high':
            return random.random() < 0.4  # Keep 40% of very_deep + high blinds
        return True
    
    def _is_realistic_opponent_action(self, opponent_action, stack_category, villain_stack_category, is_3bet):
        """Filter unrealistic opponent actions."""
        # Major filtering here (~40% reduction)
        
        # Filter 1: No action scenarios (keep some)
        if opponent_action == 'none':
            return random.random() < 0.6  # Keep 60% of none actions
        
        # Filter 2: Ultra-short stack actions
        if stack_category == 'ultra_short':
            # Ultra-short stacks mostly shove or fold
            if opponent_action not in ['fold', 'shove', 'call_low', 'raise_low']:
                return random.random() < 0.2  # Keep 20% of other actions
        
        # Filter 3: Very deep stack actions  
        if stack_category == 'very_deep':
            # Deep stacks rarely shove preflop
            if opponent_action == 'shove':
                return random.random() < 0.1  # Keep 10% of shoves
        
        # Filter 4: 3-bet scenarios with weak actions
        if is_3bet and opponent_action in ['check', 'call_low']:
            return random.random() < 0.3  # Keep 30% of weak 3-bet responses
        
        # Filter 5: Mismatched villain stack actions
        if villain_stack_category == 'ultra_short' and opponent_action in ['call_high', 'raise_high']:
            return False  # Ultra-short villains can't make big bets
        
        return True
    
    def _is_realistic_3bet_scenario(self, hand_category, opponent_action, is_3bet):
        """Filter unrealistic 3-bet scenarios."""
        # 3-bet scenarios should be less common overall
        if is_3bet:
            # 3-bets mostly with strong hands or aggressive actions
            if hand_category == 'trash' and opponent_action in ['call_low', 'check']:
                return random.random() < 0.2  # Keep 20% of trash 3-bet + weak response
            
            # 3-bets more likely with premium hands
            if hand_category in ['premium_pairs', 'premium_aces']:
                return True  # Keep all premium 3-bet scenarios
            
            # Medium filtering for other hands
            return random.random() < 0.7  # Keep 70% of other 3-bet scenarios
        else:
            # Non-3-bet scenarios - keep most
            return random.random() < 0.9  # Keep 90% of non-3-bet scenarios
    
    def _is_realistic_stack_match(self, stack_category, villain_stack_category, opponent_action):
        """Filter unrealistic stack size mismatches."""
        # Both players ultra-short is rare
        if stack_category == 'ultra_short' and villain_stack_category == 'ultra_short':
            return random.random() < 0.4  # Keep 40%
        
        # Both players very deep is less common  
        if stack_category == 'very_deep' and villain_stack_category == 'very_deep':
            return random.random() < 0.6  # Keep 60%
        
        return True
    
    def _validate_hand_categories_at_startup(self):
        """
        At startup, classify a sample for each expected category and log PASS/FAIL.
        
        This ensures the hand categorizer is working correctly for all expected categories.
        """
        self.logger.info("🔍 Validating hand categorizer at startup...")
        
        # Sample hands for each category (using first hand from each range)
        validation_samples = {}
        
        # Create sample cards for each category
        from treys import Card
        
        test_cases = {
            'premium_pairs': [Card.new('Ah'), Card.new('As')],  # AA
            'medium_pairs': [Card.new('9h'), Card.new('9d')],   # 99
            'small_pairs': [Card.new('5h'), Card.new('5d')],    # 55
            'premium_aces': [Card.new('Ah'), Card.new('Ks')],   # AK offsuit
            'medium_aces': [Card.new('Ah'), Card.new('Ts')],    # AT offsuit
            'suited_broadway': [Card.new('Kh'), Card.new('Qh')], # KQ suited
            'offsuit_broadway': [Card.new('Kh'), Card.new('Qs')], # KQ offsuit
            'suited_connectors': [Card.new('9h'), Card.new('8h')], # 98 suited
            'suited_gappers': [Card.new('Th'), Card.new('8h')],  # T8 suited
            'weak_aces': [Card.new('Ah'), Card.new('4s')],      # A4 offsuit
            'trash': [Card.new('7h'), Card.new('2c')]           # 72 offsuit
        }
        
        passed = 0
        failed = 0
        
        for expected_category, test_cards in test_cases.items():
            try:
                result = classify_hand_category(test_cards)
                if result == expected_category:
                    self.logger.info(f"  ✅ PASS: {expected_category} - {cards_to_str(test_cards)}")
                    passed += 1
                else:
                    self.logger.error(f"  ❌ FAIL: {expected_category} - {cards_to_str(test_cards)} -> {result}")
                    failed += 1
            except Exception as e:
                self.logger.error(f"  ❌ ERROR: {expected_category} - {cards_to_str(test_cards)} -> {e}")
                failed += 1
        
        if failed == 0:
            self.logger.info(f"🎉 Hand categorizer validation: ALL {passed} categories PASSED")
            print(f"🎉 Hand categorizer validation: ALL {passed} categories PASSED")
        else:
            self.logger.error(f"⚠️ Hand categorizer validation: {passed} PASSED, {failed} FAILED")
            print(f"⚠️ Hand categorizer validation: {passed} PASSED, {failed} FAILED")
    
    def classify_hand_category_with_logging(self, cards, game_id=None, position=None):
        """
        Classify a hand using the centralized categorizer with enhanced logging.
        
        Always uses the hand categorizer from enhanced_cfr_preflop_generator_v2.py.
        Only assigns trash if classifier returns trash. Logs WARNING if classifier 
        returns None/invalid.
        
        Args:
            cards: List of Card objects [card1, card2]
            game_id: Game ID for logging (optional)
            position: Player position for logging (optional)
            
        Returns:
            str: Hand category or 'trash' as fallback
        """
        try:
            result = classify_hand_category(cards)
            
            if result is None:
                # Classifier returned None/invalid - log WARNING
                readable_cards = cards_to_str(cards)
                self.scenario_classification_errors += 1
                self.logger.warning(f"⚠️ Hand classifier returned None/invalid: "
                                   f"cards={readable_cards}, game_id={game_id}, position={position}")
                return 'trash'  # Only assign trash as fallback
            
            # Track hand category seen
            self.hand_categories_seen.add(result)
            return result
            
        except Exception as e:
            # Unexpected error in classification - log WARNING
            readable_cards = cards_to_str(cards) if cards else "None"
            self.scenario_classification_errors += 1
            self.logger.warning(f"⚠️ Hand classification error: "
                               f"cards={readable_cards}, game_id={game_id}, position={position}, error={e}")
            return 'trash'  # Only assign trash as fallback
    
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
    
    def generate_game_parameters(self):
        """
        Generate random game parameters (stack sizes, blinds) for a full game.
        These parameters will be fixed for the entire game until one player is busted.
        
        Returns:
            dict: Game parameters with stack sizes and blinds level
        """
        # Random stack sizes - both players start with the same stack for this game
        initial_stack_bb = random.randint(8, 200)
        
        # Random blinds level
        blinds_level = random.choice(["low", "medium", "high"])
        
        return {
            'initial_stack_bb': initial_stack_bb,
            'blinds_level': blinds_level
        }

    def generate_random_game_state(self, hero_stack_bb=None, villain_stack_bb=None, blinds_level=None):
        """
        Generate a random game state for natural Monte Carlo simulation.
        
        Args:
            hero_stack_bb: Hero's current stack size (if None, randomized)
            villain_stack_bb: Villain's current stack size (if None, randomized) 
            blinds_level: Current blinds level (if None, randomized)
        
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
        
        # Use provided stack sizes or generate random ones (backward compatibility)
        if hero_stack_bb is None:
            hero_stack_bb = random.randint(8, 200)
        if villain_stack_bb is None:
            villain_stack_bb = random.randint(8, 200)
        
        # Use provided blinds level or generate random one (backward compatibility)
        if blinds_level is None:
            blinds_level = random.choice(["low", "medium", "high"])
        
        # Classify hands and stacks using the centralized categorizer
        hero_hand_category = self.classify_hand_category_with_logging(
            hero_cards, game_id=f"game_{self.natural_metrics['games_played']}", position=hero_position
        )
        villain_hand_category = self.classify_hand_category_with_logging(
            villain_cards, game_id=f"game_{self.natural_metrics['games_played']}", position=villain_position
        )
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
        Get context-aware available actions based on current game state.
        
        Updated with realistic action filtering considering:
        - Betting context (opening, facing raise, 3-bet situation)
        - Stack size relative to blinds and pot
        - Bet history and position
        
        Args:
            game_state: Current game state dictionary
            is_hero: Whether actions are for hero (True) or villain (False)
            
        Returns:
            list: Available actions for the current player
        """
        stack_bb = game_state['hero_stack_bb'] if is_hero else game_state['villain_stack_bb']
        pot_bb = game_state['pot_bb']
        action_history = game_state['action_history']
        position = game_state['hero_position'] if is_hero else game_state['villain_position']
        is_3bet_situation = game_state.get('is_3bet', False)
        
        # Base action set
        actions = ['fold']
        
        # Context-aware action filtering
        if not action_history:
            # First to act - opening actions
            actions.extend(self._get_opening_actions(stack_bb, position))
        else:
            # Responding to previous action
            last_action = action_history[-1]['action']
            actions.extend(self._get_response_actions(stack_bb, last_action, pot_bb, is_3bet_situation))
        
        # Stack-size adjustments
        actions = self._filter_by_stack_size(actions, stack_bb, pot_bb)
        
        # Remove duplicates and sort
        actions = list(dict.fromkeys(actions))  # Preserve order while removing duplicates
        
        return actions
    
    def _get_opening_actions(self, stack_bb, position):
        """Get available opening actions based on stack and position."""
        actions = []
        
        if stack_bb <= 10:
            # Ultra-short stack: push/fold strategy
            actions = ['check', 'shove']
        elif stack_bb <= 20:
            # Short stack: limited options
            actions = ['check', 'raise_low', 'raise_mid', 'shove']
        else:
            # Standard stack: full range
            if position == 'BTN':
                # BTN: Can open-raise or limp
                actions = ['check', 'call_low', 'raise_low', 'raise_mid', 'raise_high']
            else:
                # BB: Can check or raise
                actions = ['check', 'raise_low', 'raise_mid', 'raise_high']
        
        return actions
    
    def _get_response_actions(self, stack_bb, last_action, pot_bb, is_3bet_situation):
        """Get available response actions based on opponent's last action."""
        actions = []
        
        if last_action == 'check':
            # Facing a check - can check back or bet
            actions = ['check', 'raise_low', 'raise_mid', 'raise_high']
            if stack_bb <= 15:
                actions.append('shove')
                
        elif 'call' in last_action:
            # Facing a call - can check or bet
            actions = ['check', 'raise_low', 'raise_mid', 'raise_high']
            
        elif 'raise' in last_action:
            # Facing a raise - can call, re-raise, or fold
            if is_3bet_situation:
                # 3-bet situation - more aggressive options
                actions = ['call_low', 'call_mid', 'raise_mid', 'raise_high']
                if stack_bb <= 25:
                    actions.append('shove')
            else:
                # Standard raise response
                actions = ['call_low', 'call_mid', 'call_high', 'raise_low', 'raise_mid', 'raise_high']
                
        elif last_action == 'shove':
            # Facing shove - call or fold only
            actions = ['call_high']  # Calling an all-in is always call_high
            
        else:
            # Default fallback
            actions = ['check', 'call_low', 'raise_low', 'raise_mid']
        
        return actions
    
    def _filter_by_stack_size(self, actions, stack_bb, pot_bb):
        """Filter actions based on stack size constraints."""
        filtered_actions = ['fold']  # Fold always available
        
        for action in actions:
            if action == 'fold':
                continue  # Already added
                
            # Stack size filtering
            if stack_bb <= 10:
                # Ultra-short: only check, shove, or call_low
                if action in ['check', 'shove', 'call_low']:
                    filtered_actions.append(action)
            elif stack_bb <= 20:
                # Short: limited to smaller bets
                if action not in ['call_high', 'raise_high']:
                    filtered_actions.append(action)
            elif stack_bb <= 50:
                # Medium: no huge bets
                if action != 'raise_high' or pot_bb < stack_bb * 0.3:
                    filtered_actions.append(action)
            else:
                # Deep: all actions available
                filtered_actions.append(action)
        
        return filtered_actions
    
    def get_scenario_key_from_game_state(self, game_state, is_hero=True):
        """
        Generate unified scenario key from current game state.
        
        Updated to required format: '{hand_cat}|{position}|{stack_cat}|{blinds_level}|{villain_stack_cat}|{preflop_context}'
        
        The scenario_key is identical for hero/villain in the same situation to enable
        unified aggregation as specified in the requirements.
        
        Args:
            game_state: Current game state
            is_hero: Whether key is for hero (True) or villain (False)
            
        Returns:
            str: Unified scenario key for strategy lookup and aggregation
        """
        # Create cache key
        cache_key = (
            game_state.get('hero_hand_category', ''),
            game_state.get('hero_position', ''),
            game_state.get('hero_stack_category', ''),
            game_state.get('villain_hand_category', ''),
            game_state.get('villain_position', ''),
            game_state.get('villain_stack_category', ''),
            game_state.get('blinds_level', ''),
            len(game_state.get('action_history', [])),
            game_state.get('is_3bet', False),
            is_hero
        )
        
        # Check cache first
        if cache_key in self.scenario_key_cache:
            return self.scenario_key_cache[cache_key]
        
        if is_hero:
            hand_category = game_state['hero_hand_category']
            position = game_state['hero_position']
            stack_category = game_state['hero_stack_category']
            villain_stack_category = game_state['villain_stack_category']
        else:
            hand_category = game_state['villain_hand_category']
            position = game_state['villain_position']
            stack_category = game_state['villain_stack_category']
            villain_stack_category = game_state['hero_stack_category']  # Villain's perspective
        
        blinds_level = game_state['blinds_level']
        
        # Build preflop context from action history and game state
        action_history = game_state.get('action_history', [])
        is_3bet = game_state.get('is_3bet', False)
        opponent_action = self._get_last_opponent_action(action_history, is_hero)
        
        # Create preflop context string 
        context_parts = []
        if opponent_action and opponent_action != 'none':
            context_parts.append(f"opp_{opponent_action}")
        if is_3bet:
            context_parts.append("3bet")
        if len(action_history) == 0:
            context_parts.append("first_to_act")
        
        preflop_context = "_".join(context_parts) if context_parts else "standard"
        
        # Build unified scenario key as specified: '{hand_cat}|{position}|{stack_cat}|{blinds_level}|{villain_stack_cat}|{preflop_context}'
        scenario_key = f"{hand_category}|{position}|{stack_category}|{blinds_level}|{villain_stack_category}|{preflop_context}"
        
        # Cache the result
        self.scenario_key_cache[cache_key] = scenario_key
        
        return scenario_key
    
    def _get_last_opponent_action(self, action_history, is_hero):
        """
        Extract the last opponent action from action history.
        
        Args:
            action_history: List of action dictionaries
            is_hero: Whether we're looking from hero's perspective
            
        Returns:
            str: Last opponent action or None if no opponent action
        """
        if not action_history:
            return 'none'
        
        # Look through history backwards for last opponent action
        for action_record in reversed(action_history):
            # If we're hero, look for villain actions (is_hero=False)
            # If we're villain, look for hero actions (is_hero=True)
            if action_record.get('is_hero') != is_hero:
                return action_record.get('action', 'none')
        
        return 'none'
    
    def update_scenario_aggregation(self, scenario_key, action_taken):
        """
        Update the single in-memory dict keyed by scenario_key.
        
        Tracks visits and sum_{action} for each action bucket as specified.
        Updates dict for acting player's action regardless of role.
        
        Args:
            scenario_key: The unified scenario key
            action_taken: The action that was taken
        """
        if scenario_key not in self.scenario_aggregation:
            # Initialize entry for new scenario
            self.scenario_aggregation[scenario_key] = {
                'visits': 0,
                'sum_fold': 0,
                'sum_call_small': 0,
                'sum_raise_small': 0, 
                'sum_raise_mid': 0,
                'sum_raise_high': 0
            }
        
        # Increment visit count
        self.scenario_aggregation[scenario_key]['visits'] += 1
        
        # Update action sum (map actions to the 5 specified buckets)
        action_mapping = {
            'fold': 'sum_fold',
            'check': 'sum_call_small',  # Check maps to call_small bucket
            'call_low': 'sum_call_small',
            'call_mid': 'sum_call_small', 
            'call_high': 'sum_call_small',
            'raise_low': 'sum_raise_small',
            'raise_mid': 'sum_raise_mid',
            'raise_high': 'sum_raise_high',
            'shove': 'sum_raise_high'  # Shove maps to raise_high bucket
        }
        
        if action_taken in action_mapping:
            bucket = action_mapping[action_taken]
            self.scenario_aggregation[scenario_key][bucket] += 1
        
        # Update total scenarios processed for ratios
        self.total_scenarios_processed += 1
    
    def log_periodic_metrics(self):
        """
        Log periodic metrics: category_coverage, trash_ratio, distinct_scenarios.
        
        Also logs HIGH PRIORITY WARNING if trash_ratio > 80% after 1,000 scenarios.
        """
        if self.total_scenarios_processed == 0:
            return
        
        # Calculate metrics
        category_coverage = len(self.hand_categories_seen)
        distinct_scenarios = len(self.scenario_aggregation)
        
        # Calculate trash ratio
        trash_count = sum(1 for cat in self.hand_categories_seen if cat == 'trash')
        trash_ratio = trash_count / len(self.hand_categories_seen) if self.hand_categories_seen else 0.0
        
        # Log metrics
        self.logger.info(f"📊 Periodic metrics: category_coverage={category_coverage}, "
                        f"distinct_scenarios={distinct_scenarios}, trash_ratio={trash_ratio:.3f}")
        
        # Check for high trash ratio warning after 1,000 scenarios
        if self.total_scenarios_processed >= 1000 and trash_ratio > 0.8:
            self.logger.warning(f"🚨 HIGH PRIORITY WARNING: trash_ratio > 80% "
                              f"({trash_ratio:.1%}) after {self.total_scenarios_processed} scenarios")
            print(f"🚨 HIGH PRIORITY WARNING: trash_ratio > 80% ({trash_ratio:.1%})")
    
    def log_game_summary(self):
        """
        Log game summary with hand categories seen.
        """
        hand_cats_seen = sorted(list(self.hand_categories_seen))
        self.logger.info(f"🎮 Game summary: hand_cats_seen={hand_cats_seen}")
        
        # Also log current aggregation stats
        if self.scenario_aggregation:
            total_visits = sum(entry['visits'] for entry in self.scenario_aggregation.values())
            avg_visits = total_visits / len(self.scenario_aggregation)
            self.logger.info(f"📈 Aggregation stats: {len(self.scenario_aggregation)} scenarios, "
                           f"{total_visits} total visits, {avg_visits:.1f} avg visits per scenario")
    
    def should_explore(self, scenario_key, action, is_hero=True):
        """
        Determine if we should explore this action with progressive epsilon decay.
        
        Enhanced exploration strategy:
        - Progressive epsilon decay over time
        - Prioritize under-explored scenarios and actions
        - Higher exploration for rare/important scenarios
        
        Args:
            scenario_key: Current scenario key
            action: Action being considered
            is_hero: Whether this is for hero (True) or villain (False)
            
        Returns:
            bool: True if should explore (force random action)
        """
        # Get current visit counts
        scenario_visits = self.scenario_visits[scenario_key]
        action_visits = self.state_action_visits[scenario_key][action]
        
        # Progressive epsilon decay based on total games played
        games_played = self.natural_metrics['games_played']
        base_epsilon = self.epsilon_exploration
        
        # Decay formula: start high, decay slowly, minimum floor
        decay_factor = max(0.1, 1.0 - (games_played / 10000))  # Decay over 10k games
        current_epsilon = base_epsilon * decay_factor
        
        # Boost exploration for under-explored scenarios
        if scenario_visits < self.min_visit_threshold:
            current_epsilon *= 2.0  # Double epsilon for new scenarios
        elif action_visits < 3:
            current_epsilon *= 1.5  # 50% boost for under-explored actions
        
        # Boost exploration for important scenarios (3-bet, short stacks)
        if 'True' in scenario_key:  # 3-bet scenario
            current_epsilon *= 1.3
        if 'ultra_short' in scenario_key or 'short' in scenario_key:
            current_epsilon *= 1.2
        
        # Cap at maximum exploration rate
        current_epsilon = min(current_epsilon, 0.8)
        
        return random.random() < current_epsilon
    
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
    
    def simulate_single_hand(self, hero_stack_bb=None, villain_stack_bb=None, blinds_level=None):
        """
        Simulate a single poker hand using Monte Carlo approach.
        
        Both players act according to their learned strategies.
        Records natural scenarios that emerge during gameplay.
        
        Args:
            hero_stack_bb: Hero's current stack size (if None, randomized)
            villain_stack_bb: Villain's current stack size (if None, randomized)
            blinds_level: Current blinds level (if None, randomized)
        
        Returns:
            dict: Complete hand simulation result
        """
        # Generate game state with specified or random parameters
        game_state = self.generate_random_game_state(hero_stack_bb, villain_stack_bb, blinds_level)
        
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
            
            # Update scenario aggregation for the acting player
            scenario_key = self.get_scenario_key_from_game_state(game_state, is_hero_turn)
            self.update_scenario_aggregation(scenario_key, action)
            
            # Update game state based on action
            self.update_game_state_with_action(game_state, action, is_hero_turn)
            
            # Check if betting round is complete
            betting_complete = self.is_betting_complete(game_state)
            action_count += 1
        
        # Calculate final payoffs
        payoff_result = self.calculate_game_payoffs(game_state)
        
        # Record natural scenario that emerged (if it passes filtering)
        natural_scenario = self.record_natural_scenario(game_state, payoff_result)
        
        # Update strategies for both players (even if scenario was filtered)
        self.update_strategies_from_game(game_state, payoff_result)
        
        # Note: games_played is now tracked in simulate_full_game, not here
        
        return {
            'game_state': game_state,
            'natural_scenario': natural_scenario,  # May be None if filtered
            'payoff_result': payoff_result,
            'exploration_used': exploration_used
        }
    
    def monte_carlo_game_simulation(self, hero_stack_bb=None, villain_stack_bb=None, blinds_level=None):
        """
        Perform a single Monte Carlo game simulation.
        
        This method is an alias for simulate_single_hand to maintain compatibility
        with existing test suites while providing a clear interface for Monte Carlo simulation.
        
        Args:
            hero_stack_bb: Hero's current stack size (if None, randomized)
            villain_stack_bb: Villain's current stack size (if None, randomized) 
            blinds_level: Current blinds level (if None, randomized)
        
        Returns:
            dict: Complete simulation result with game_state, natural_scenario, payoff_result
        """
        return self.simulate_single_hand(hero_stack_bb, villain_stack_bb, blinds_level)
    
    def simulate_full_game(self):
        """
        Simulate a complete poker game consisting of multiple hands with fixed parameters.
        
        A game consists of:
        1. Randomly selecting initial stack size and blind level
        2. Playing multiple hands with these fixed parameters
        3. Continuing until one player is busted (stack = 0)
        
        Returns:
            dict: Complete game simulation result with all hands played
        """
        # Generate random game parameters (fixed for this entire game)
        game_params = self.generate_game_parameters()
        initial_stack_bb = game_params['initial_stack_bb']
        blinds_level = game_params['blinds_level']
        
        # Initialize player stacks
        hero_stack_bb = initial_stack_bb
        villain_stack_bb = initial_stack_bb
        
        hands_played = []
        game_start_time = time.time()
        hand_count = 0
        max_hands_per_game = 1000  # Safety limit to prevent infinite games
        
        self.logger.debug(f"Starting new game: initial_stack={initial_stack_bb}bb, blinds={blinds_level}")
        
        while hero_stack_bb > 0 and villain_stack_bb > 0 and hand_count < max_hands_per_game:
            # Check if either player can afford the blinds
            min_stack_for_blinds = 1.5  # Need at least 1.5bb for small blind + big blind
            if hero_stack_bb < min_stack_for_blinds or villain_stack_bb < min_stack_for_blinds:
                break
                
            # Simulate one hand with current stack sizes
            hand_result = self.simulate_single_hand(hero_stack_bb, villain_stack_bb, blinds_level)
            
            # Extract payoff information
            payoff_result = hand_result['payoff_result']
            hero_payoff = payoff_result['hero_payoff']
            villain_payoff = payoff_result['villain_payoff']
            
            # Update stack sizes based on hand result
            hero_stack_bb += hero_payoff
            villain_stack_bb += villain_payoff
            
            # Ensure stacks don't go negative (shouldn't happen but safety check)
            hero_stack_bb = max(0, hero_stack_bb)
            villain_stack_bb = max(0, villain_stack_bb)
            
            # Record hand information
            hand_info = {
                'hand_number': hand_count + 1,
                'hero_stack_before': hero_stack_bb - hero_payoff,
                'villain_stack_before': villain_stack_bb - villain_payoff,
                'hero_stack_after': hero_stack_bb,
                'villain_stack_after': villain_stack_bb,
                'hero_payoff': hero_payoff,
                'villain_payoff': villain_payoff,
                'natural_scenario': hand_result['natural_scenario'],
                'game_state': hand_result['game_state']
            }
            hands_played.append(hand_info)
            hand_count += 1
            
            self.logger.debug(f"Hand {hand_count}: Hero {hero_stack_bb:.1f}bb, Villain {villain_stack_bb:.1f}bb")
        
        # Determine game winner
        if hero_stack_bb <= 0:
            game_winner = 'villain'
        elif villain_stack_bb <= 0:
            game_winner = 'hero'
        else:
            # Game ended due to max hands limit
            game_winner = 'hero' if hero_stack_bb > villain_stack_bb else 'villain'
        
        game_duration = time.time() - game_start_time
        
        # Update game-level metrics
        self.natural_metrics['games_played'] += 1
        total_hands = len(hands_played)
        
        self.logger.info(f"Game completed: {total_hands} hands, winner: {game_winner}, "
                        f"duration: {game_duration:.2f}s, stacks: Hero {hero_stack_bb:.1f}bb, Villain {villain_stack_bb:.1f}bb")
        
        return {
            'game_params': game_params,
            'initial_stack_bb': initial_stack_bb,
            'blinds_level': blinds_level,
            'hands_played': hands_played,
            'hand_count': total_hands,
            'game_winner': game_winner,
            'final_hero_stack': hero_stack_bb,
            'final_villain_stack': villain_stack_bb,
            'game_duration': game_duration
        }
    
    def update_game_state_with_action(self, game_state, action, is_hero):
        """
        Update game state based on player action with realistic raise sizing.
        
        Implements context-aware raise sizing:
        - Opening: 2.5-3x BB
        - 3-bet: 3-4x previous raise
        - Isolation: 3-5x BB + 1x per limper
        - Stack-adjusted for short stacks
        
        Args:
            game_state: Current game state (modified in place)
            action: Action taken
            is_hero: Whether action was by hero
        """
        stack_bb = game_state['hero_stack_bb'] if is_hero else game_state['villain_stack_bb']
        position = game_state['hero_position'] if is_hero else game_state['villain_position']
        action_history = game_state['action_history']
        
        if action == 'fold':
            game_state['folded'] = True
            game_state['folder'] = 'hero' if is_hero else 'villain'
            
        elif action == 'check':
            # Check - no additional bet
            pass
            
        elif action in ['call_low', 'call_mid', 'call_high']:
            # Context-aware call sizing
            call_amount = self._calculate_call_amount(action, stack_bb, game_state)
            game_state['pot_bb'] += call_amount
            if is_hero:
                game_state['hero_stack_bb'] -= call_amount
            else:
                game_state['villain_stack_bb'] -= call_amount
                
        elif action in ['raise_low', 'raise_mid', 'raise_high']:
            # Context-aware raise sizing
            raise_amount = self._calculate_raise_amount(action, stack_bb, game_state, position, action_history)
            game_state['pot_bb'] += raise_amount
            if is_hero:
                game_state['hero_stack_bb'] -= raise_amount
            else:
                game_state['villain_stack_bb'] -= raise_amount
            
            # Update 3-bet detection
            self._update_3bet_status(game_state, action_history)
            
        elif action == 'shove':
            # All-in
            all_in_amount = stack_bb
            game_state['pot_bb'] += all_in_amount
            if is_hero:
                game_state['hero_stack_bb'] = 0
            else:
                game_state['villain_stack_bb'] = 0
    
    def _calculate_call_amount(self, action, stack_bb, game_state):
        """Calculate realistic call amount based on action type and context."""
        pot_bb = game_state['pot_bb']
        
        if action == 'call_low':
            # Small call: 0.5-1.5x BB or 10-20% of stack
            return min(max(1.0, pot_bb * 0.4), stack_bb * 0.15)
        elif action == 'call_mid':
            # Medium call: 1.5-3x BB or 20-40% of stack  
            return min(max(2.0, pot_bb * 0.6), stack_bb * 0.35)
        elif action == 'call_high':
            # Large call: 3x+ BB or 40%+ of stack
            return min(max(3.0, pot_bb * 0.8), stack_bb)
        
        return 1.0  # Fallback
    
    def _calculate_raise_amount(self, action, stack_bb, game_state, position, action_history):
        """Calculate realistic raise amount based on context."""
        pot_bb = game_state['pot_bb']
        is_3bet_situation = game_state.get('is_3bet', False)
        
        # Determine raise context
        if not action_history:
            # Opening raise
            base_size = self._get_opening_raise_size(position, stack_bb)
        elif is_3bet_situation:
            # 3-bet sizing
            base_size = self._get_3bet_raise_size(stack_bb, pot_bb)
        else:
            # Standard raise/re-raise
            base_size = self._get_standard_raise_size(stack_bb, pot_bb)
        
        # Apply action modifier
        if action == 'raise_low':
            return min(base_size * 0.8, stack_bb)
        elif action == 'raise_mid':
            return min(base_size, stack_bb)
        elif action == 'raise_high':
            return min(base_size * 1.3, stack_bb)
        
        return min(base_size, stack_bb)
    
    def _get_opening_raise_size(self, position, stack_bb):
        """Get opening raise size based on position and stack."""
        if stack_bb <= 15:
            # Short stack: smaller opens or shove
            return min(stack_bb, 2.2)
        elif position == 'BTN':
            # BTN: 2.5-3x BB standard open
            return 2.8
        else:
            # BB: 3-4x BB when raising
            return 3.2
    
    def _get_3bet_raise_size(self, stack_bb, pot_bb):
        """Get 3-bet raise size."""
        if stack_bb <= 25:
            # Short stack 3-bet: often shove or small 3-bet
            return min(stack_bb, pot_bb * 2.5)
        else:
            # Standard 3-bet: 3-4x previous raise
            return pot_bb * 3.2
    
    def _get_standard_raise_size(self, stack_bb, pot_bb):
        """Get standard raise size for non-opening, non-3-bet situations."""
        if stack_bb <= 20:
            return min(stack_bb, pot_bb * 2.0)
        else:
            return pot_bb * 2.5
    
    def _update_3bet_status(self, game_state, action_history):
        """Update 3-bet status based on action history."""
        raise_count = sum(1 for a in action_history if 'raise' in a.get('action', ''))
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
        Record a natural scenario that emerged during gameplay with smart filtering.
        
        Only records scenarios that pass realistic filtering to achieve
        ~3,500-4,000 meaningful combinations.
        
        Args:
            game_state: Final game state
            payoff_result: Payoff results
            
        Returns:
            dict: Recorded natural scenario (None if filtered out)
        """
        # Generate scenario key for filtering
        scenario_key = self.get_scenario_key_from_game_state(game_state, is_hero=True)
        
        # Apply smart scenario filtering
        if not self.is_realistic_scenario(scenario_key):
            self.filtered_scenario_count += 1
            return None
        
        # Determine opponent action (last action by opponent)
        opponent_action = "none"
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
        self.logger.info("🚀 Starting Natural Game CFR Training")
        self.logger.info(f"   🎲 Games to simulate: {n_games:,}")
        self.logger.info(f"   💾 Save interval: every {save_interval} games")
        self.logger.info(f"   📝 Log interval: every {log_interval} games")
        self.logger.info("=" * 60)
        
        print(f"🚀 Starting Natural Game CFR Training")
        print(f"   🎲 Games to simulate: {n_games:,}")
        print(f"   💾 Save interval: every {save_interval} games")
        print(f"   📝 Log interval: every {log_interval} games")
        print("=" * 60)
        
        training_start_time = time.time()
        total_hands_played = 0  # Track total hands across all games
        
        for game_num in range(n_games):
            # Simulate one full game (multiple hands until bust)
            try:
                game_result = self.simulate_full_game()
                # Update total hands count
                total_hands_played += game_result['hand_count']
            except Exception as e:
                self.logger.error(f"Error in game {game_num + 1}: {e}")
                from logging_config import log_exception
                log_exception(self.logger, f"Error in game {game_num + 1}")
                continue  # Skip this game and continue training
            
            # Log progress
            if (game_num + 1) % log_interval == 0:
                self.log_training_progress(game_num + 1, training_start_time, total_hands_played)
                # Add periodic metrics logging
                self.log_periodic_metrics()
            
            # Save progress and cleanup memory
            if (game_num + 1) % save_interval == 0:
                checkpoint_file = f"natural_cfr_checkpoint_{game_num + 1}.pkl"
                self.save_training_state(checkpoint_file)
                # Perform memory cleanup after saving
                self._cleanup_memory()
                # Export scenario lookup table regularly
                self.export_scenario_lookup_table_csv("scenario_lookup_table.csv")
            
            # Log game summary every 1000 games
            if (game_num + 1) % 1000 == 0:
                self.log_game_summary()
            
            # Additional memory cleanup at regular intervals
            elif (game_num + 1) % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
        
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        # Update final metrics
        self.natural_metrics['hands_played'] = total_hands_played
        if self.natural_metrics['games_played'] > 0:
            self.natural_metrics['avg_hands_per_game'] = total_hands_played / self.natural_metrics['games_played']
        
        self.logger.info("🎉 Natural Game CFR Training Complete!")
        self.logger.info(f"   ⏱️  Total time: {total_training_time/60:.1f} minutes")
        self.logger.info(f"   🎲 Games played: {self.natural_metrics['games_played']:,}")
        self.logger.info(f"   🃏 Hands played: {total_hands_played:,}")
        self.logger.info(f"   📊 Average hands per game: {self.natural_metrics['avg_hands_per_game']:.1f}")
        self.logger.info(f"   📊 Unique scenarios: {self.natural_metrics['unique_scenarios']}")
        self.logger.info(f"   🎯 Total scenarios recorded: {len(self.natural_scenarios)}")
        
        # Export final unified scenario lookup table
        self.logger.info("📊 Exporting final unified scenario lookup table...")
        try:
            self.export_unified_scenario_lookup_csv("scenario_lookup_table.csv")
        except Exception as export_error:
            self.logger.warning(f"⚠️ Final scenario lookup table export failed: {export_error}")
        
        print(f"\n🎉 Natural Game CFR Training Complete!")
        print(f"   ⏱️  Total time: {total_training_time/60:.1f} minutes")
        print(f"   🎲 Games played: {self.natural_metrics['games_played']:,}")
        print(f"   🃏 Hands played: {total_hands_played:,}")
        print(f"   📊 Average hands per game: {self.natural_metrics['avg_hands_per_game']:.1f}")
        print(f"   📊 Unique scenarios: {self.natural_metrics['unique_scenarios']}")
        print(f"   🎯 Total scenarios recorded: {len(self.natural_scenarios)}")
        
        return {
            'games_played': self.natural_metrics['games_played'],
            'hands_played': total_hands_played,
            'avg_hands_per_game': self.natural_metrics['avg_hands_per_game'],
            'unique_scenarios': self.natural_metrics['unique_scenarios'],
            'total_training_time': total_training_time,
            'natural_scenarios': len(self.natural_scenarios),
            'hero_strategy_scenarios': len(self.strategy_sum),
            'villain_strategy_scenarios': len(self.villain_strategy_sum)
        }
    
    def log_training_progress(self, games_completed, start_time, total_hands_played=None):
        """
        Log training progress with key metrics.
        
        Args:
            games_completed: Number of games completed
            start_time: Training start timestamp
            total_hands_played: Total number of hands played across all games
        """
        elapsed_time = time.time() - start_time
        games_per_minute = (games_completed / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Update metrics if hands tracking provided
        if total_hands_played is not None:
            self.natural_metrics['hands_played'] = total_hands_played
            if games_completed > 0:
                self.natural_metrics['avg_hands_per_game'] = total_hands_played / games_completed
        
        # Calculate win rates and other statistics
        if self.natural_scenarios:
            hero_wins = sum(1 for s in self.natural_scenarios if s['hero_won'])
            hero_win_rate = hero_wins / len(self.natural_scenarios)
            showdown_rate = sum(1 for s in self.natural_scenarios if s['showdown']) / len(self.natural_scenarios)
            three_bet_rate = sum(1 for s in self.natural_scenarios if s['is_3bet']) / len(self.natural_scenarios)
            avg_pot_size = sum(s['final_pot_bb'] for s in self.natural_scenarios) / len(self.natural_scenarios)
        else:
            hero_win_rate = 0.0
            showdown_rate = 0.0
            three_bet_rate = 0.0
            avg_pot_size = 0.0
        
        # Estimate time remaining
        if games_per_minute > 0:
            remaining_games = self.natural_metrics.get('games_target', 10000) - games_completed
            if remaining_games > 0:
                eta_minutes = remaining_games / games_per_minute
                eta_str = f", ETA: {eta_minutes:.1f}min"
            else:
                eta_str = ""
        else:
            eta_str = ""
        
        # Console output with hands information
        hands_info = f", {total_hands_played} hands" if total_hands_played else ""
        avg_hands_info = f", avg={self.natural_metrics['avg_hands_per_game']:.1f}h/g" if total_hands_played else ""
        console_msg = (f"Game {games_completed:6,}{hands_info}{avg_hands_info}: "
                      f"{self.natural_metrics['unique_scenarios']:3d} scenarios, "
                      f"hero_wr={hero_win_rate:.2f}, "
                      f"rate={games_per_minute:.1f}/min{eta_str}")
        print(console_msg)
        
        # Detailed logging
        self.logger.info(f"TRAINING PROGRESS - Game {games_completed:,}")
        self.logger.info(f"  Time elapsed: {elapsed_time/60:.1f} minutes")
        self.logger.info(f"  Games per minute: {games_per_minute:.1f}")
        self.logger.info(f"  Total games played: {self.natural_metrics['games_played']:,}")
        if total_hands_played:
            self.logger.info(f"  Total hands played: {total_hands_played:,}")
            self.logger.info(f"  Average hands per game: {self.natural_metrics['avg_hands_per_game']:.1f}")
        self.logger.info(f"  Monte Carlo iterations completed: {games_completed:,}")
        self.logger.info(f"  Unique scenarios discovered: {self.natural_metrics['unique_scenarios']}")
        self.logger.info(f"  Total scenarios recorded: {len(self.natural_scenarios)}")
        
        if self.natural_scenarios:
            self.logger.info(f"  Hero win rate: {hero_win_rate:.3f}")
            self.logger.info(f"  Showdown rate: {showdown_rate:.3f}")
            self.logger.info(f"  3-bet rate: {three_bet_rate:.3f}")
            self.logger.info(f"  Average pot size: {avg_pot_size:.2f} BB")
        
        # Log exploration statistics
        total_state_actions = sum(len(actions) for actions in self.state_action_visits.values())
        self.logger.info(f"  State-action pairs visited: {total_state_actions}")
        
        if eta_str:
            self.logger.info(f"  Estimated time remaining: {eta_minutes:.1f} minutes")
        
        # Update metrics
        self.natural_metrics['hero_win_rate'] = hero_win_rate
        self.natural_metrics['avg_pot_size'] = avg_pot_size
        
        # Export unified scenario lookup table for live monitoring
        try:
            self.export_unified_scenario_lookup_csv("scenario_lookup_table.csv")
        except Exception as export_error:
            self.logger.warning(f"⚠️ Scenario lookup table export failed: {export_error}")
    
    def save_training_state(self, filename):
        """
        Save complete training state to file in checkpoints directory.
        
        Args:
            filename: File to save to (will be placed in checkpoints/ directory)
        """
        try:
            # Create checkpoints directory if it doesn't exist
            checkpoints_dir = Path("checkpoints")
            checkpoints_dir.mkdir(exist_ok=True)
            
            # Ensure filename goes into checkpoints directory
            if not filename.startswith("checkpoints/"):
                filepath = checkpoints_dir / filename
            else:
                filepath = Path(filename)
            
            self.logger.info(f"Saving training state to {filepath}...")
            self.logger.info(f"Checkpoints directory: {checkpoints_dir.absolute()}")
            
            save_data = {
                'hero_regret_sum': dict(self.regret_sum),
                'hero_strategy_sum': dict(self.strategy_sum),
                'villain_regret_sum': dict(self.villain_regret_sum),
                'villain_strategy_sum': dict(self.villain_strategy_sum),
                'natural_scenarios': self.natural_scenarios,
                'natural_scenario_counter': dict(self.natural_scenario_counter),
                'state_action_visits': dict(self.state_action_visits),
                'natural_metrics': self.natural_metrics,
                # NEW: Include scenario aggregation data for checkpoint restoration
                'scenario_aggregation': dict(self.scenario_aggregation),
                'hand_categories_seen': list(self.hand_categories_seen),
                'scenario_classification_errors': self.scenario_classification_errors,
                'total_scenarios_processed': self.total_scenarios_processed,
                'training_parameters': {
                    'epsilon_exploration': self.epsilon_exploration,
                    'min_visit_threshold': self.min_visit_threshold,
                    'tournament_survival_penalty': self.tournament_survival_penalty
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            file_size_mb = filepath.stat().st_size / 1024 / 1024
            self.logger.info(f"✅ Training state saved successfully to {filepath}")
            self.logger.info(f"   Games played: {self.natural_metrics['games_played']:,}")
            self.logger.info(f"   Unique scenarios: {self.natural_metrics['unique_scenarios']}")
            self.logger.info(f"   File size: {file_size_mb:.1f} MB")
            self.logger.info(f"   Checkpoint path: {filepath.absolute()}")
            
            print(f"💾 Training state saved to {filepath}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to save training state to {filepath}: {e}"
            self.logger.error(error_msg)
            from logging_config import log_exception
            log_exception(self.logger, f"Failed to save training state to {filepath}")
            print(f"❌ Failed to save training state: {e}")
            return False
    
    def load_training_state(self, filename):
        """
        Load training state from file.
        
        Args:
            filename: File to load from
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            self.logger.info(f"Attempting to load training state from {filename}...")
            
            if not Path(filename).exists():
                error_msg = f"Checkpoint file {filename} does not exist"
                self.logger.error(error_msg)
                return False
            
            file_size = Path(filename).stat().st_size / 1024 / 1024
            self.logger.info(f"Loading checkpoint file (size: {file_size:.1f} MB)...")
            
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
            
            # NEW: Restore scenario aggregation if present, else initialize empty
            if 'scenario_aggregation' in save_data:
                self.scenario_aggregation = save_data['scenario_aggregation']
                self.hand_categories_seen = set(save_data.get('hand_categories_seen', []))
                self.scenario_classification_errors = save_data.get('scenario_classification_errors', 0)
                self.total_scenarios_processed = save_data.get('total_scenarios_processed', 0)
                self.logger.info(f"   ✅ Restored scenario aggregation: {len(self.scenario_aggregation)} scenarios")
            else:
                self.scenario_aggregation = {}
                self.hand_categories_seen = set()
                self.scenario_classification_errors = 0
                self.total_scenarios_processed = 0
                self.logger.info("   🔄 Initialized empty scenario aggregation (legacy checkpoint)")
            
            # Log loaded state information
            self.logger.info("✅ Training state loaded successfully")
            self.logger.info(f"   Checkpoint created: {save_data.get('timestamp', 'Unknown')}")
            self.logger.info(f"   Games played: {self.natural_metrics['games_played']:,}")
            self.logger.info(f"   Unique scenarios: {self.natural_metrics['unique_scenarios']}")
            self.logger.info(f"   Hero strategies: {len(self.strategy_sum)}")
            self.logger.info(f"   Villain strategies: {len(self.villain_strategy_sum)}")
            self.logger.info(f"   Natural scenarios recorded: {len(self.natural_scenarios)}")
            
            # Log training parameters from checkpoint
            if 'training_parameters' in save_data:
                params = save_data['training_parameters']
                self.logger.info("   Loaded training parameters:")
                for key, value in params.items():
                    self.logger.info(f"     {key}: {value}")
            
            print(f"✅ Training state loaded from {filename}")
            print(f"   🎲 Games played: {self.natural_metrics['games_played']:,}")
            print(f"   📊 Unique scenarios: {self.natural_metrics['unique_scenarios']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load training state from {filename}: {e}"
            self.logger.error(error_msg)
            from logging_config import log_exception
            log_exception(self.logger, f"Failed to load training state from {filename}")
            print(f"❌ Failed to load training state: {e}")
            return False
    
    def export_natural_scenarios_csv(self, filename="natural_scenarios.csv"):
        """
        Export all natural scenarios to CSV for analysis.
        
        Args:
            filename: Output CSV filename
        """
        if not self.natural_scenarios:
            error_msg = "No natural scenarios to export"
            self.logger.warning(error_msg)
            print("❌ No natural scenarios to export")
            return
        
        try:
            self.logger.info(f"Exporting {len(self.natural_scenarios)} natural scenarios to {filename}...")
            
            df = pd.DataFrame(self.natural_scenarios)
            df.to_csv(filename, index=False)
            
            file_size = Path(filename).stat().st_size / 1024
            self.logger.info(f"✅ Successfully exported natural scenarios to {filename} ({file_size:.1f} KB)")
            
            # Calculate and log summary statistics
            hero_win_rate = sum(s['hero_won'] for s in self.natural_scenarios) / len(self.natural_scenarios)
            showdown_rate = sum(s['showdown'] for s in self.natural_scenarios) / len(self.natural_scenarios)
            three_bet_rate = sum(s['is_3bet'] for s in self.natural_scenarios) / len(self.natural_scenarios)
            
            self.logger.info("NATURAL SCENARIOS SUMMARY:")
            self.logger.info(f"  Total scenarios: {len(self.natural_scenarios)}")
            self.logger.info(f"  Unique scenario keys: {len(self.natural_scenario_counter)}")
            self.logger.info(f"  Hero win rate: {hero_win_rate:.3f}")
            self.logger.info(f"  Showdown rate: {showdown_rate:.3f}")
            self.logger.info(f"  3-bet rate: {three_bet_rate:.3f}")
            
            print(f"📊 Exported {len(self.natural_scenarios)} natural scenarios to {filename}")
            
            # Show summary statistics
            print(f"\n📈 NATURAL SCENARIOS SUMMARY:")
            print(f"Total scenarios: {len(self.natural_scenarios)}")
            print(f"Unique scenario keys: {len(self.natural_scenario_counter)}")
            print(f"Hero win rate: {hero_win_rate:.3f}")
            print(f"Showdown rate: {showdown_rate:.3f}")
            print(f"3-bet rate: {three_bet_rate:.3f}")
            
        except Exception as e:
            error_msg = f"Failed to export natural scenarios to {filename}: {e}"
            self.logger.error(error_msg)
            from logging_config import log_exception
            log_exception(self.logger, f"Failed to export natural scenarios to {filename}")
            print(f"❌ Failed to export scenarios: {e}")
    
    def export_strategies_csv(self, filename="natural_strategies.csv"):
        """
        Export learned strategies to CSV.
        
        Args:
            filename: Output CSV filename
        """
        try:
            self.logger.info(f"Exporting strategies to {filename}...")
            
            # Export hero strategies using parent method
            hero_filename = f"hero_{filename}"
            self.logger.info(f"Exporting hero strategies to {hero_filename}...")
            hero_df = self.export_strategies_to_csv(hero_filename)
            
            # Export villain strategies
            villain_filename = f"villain_{filename}"
            if self.villain_strategy_sum:
                self.logger.info(f"Exporting villain strategies to {villain_filename}...")
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
                    villain_df.to_csv(villain_filename, index=False)
                    
                    villain_file_size = Path(villain_filename).stat().st_size / 1024
                    self.logger.info(f"✅ Exported {len(villain_data)} villain strategies to {villain_filename} ({villain_file_size:.1f} KB)")
                    print(f"📊 Exported villain strategies to {villain_filename}")
                else:
                    self.logger.warning("No villain strategy data to export")
            else:
                self.logger.warning("No villain strategies available to export")
            
            self.logger.info("✅ Strategy export completed successfully")
            print(f"✅ Strategy export complete")
            
        except Exception as e:
            error_msg = f"Failed to export strategies: {e}"
            self.logger.error(error_msg)
            from logging_config import log_exception
            log_exception(self.logger, "Failed to export strategies")
    def create_performance_summary(self, training_duration=0.0, output_format='csv'):
        """
        Create a performance summary file with key training metrics.
        
        Args:
            training_duration: Total training time in seconds
            output_format: Output format ('csv' or 'json')
            
        Returns:
            str: Path to created performance file
        """
        try:
            # Create performance directory if it doesn't exist
            performance_dir = Path("performance")
            performance_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Calculate key metrics
            games_played = self.natural_metrics['games_played']
            unique_scenarios = self.natural_metrics['unique_scenarios']
            total_scenarios_recorded = len(self.natural_scenarios)
            
            if self.natural_scenarios:
                hero_wins = sum(1 for s in self.natural_scenarios if s['hero_won'])
                hero_win_rate = hero_wins / len(self.natural_scenarios)
                villain_win_rate = 1.0 - hero_win_rate
                
                showdown_games = sum(1 for s in self.natural_scenarios if s['showdown'])
                showdown_rate = showdown_games / len(self.natural_scenarios)
                
                three_bet_games = sum(1 for s in self.natural_scenarios if s['is_3bet'])
                three_bet_rate = three_bet_games / len(self.natural_scenarios)
                
                avg_pot_size = sum(s['final_pot_bb'] for s in self.natural_scenarios) / len(self.natural_scenarios)
                
                # Calculate scenario coverage (out of theoretical max of 330)
                theoretical_max_scenarios = 330
                scenario_coverage = (unique_scenarios / theoretical_max_scenarios) * 100
                
                # Hand category distribution
                hand_categories = {}
                for scenario in self.natural_scenarios:
                    cat = scenario['hand_category']
                    hand_categories[cat] = hand_categories.get(cat, 0) + 1
            else:
                hero_win_rate = 0.0
                villain_win_rate = 0.0
                showdown_rate = 0.0
                three_bet_rate = 0.0
                avg_pot_size = 0.0
                scenario_coverage = 0.0
                hand_categories = {}
            
            # Calculate exploration statistics
            total_state_actions = sum(len(actions) for actions in self.state_action_visits.values())
            exploration_rate = self.epsilon_exploration
            
            # Games per minute
            games_per_minute = (games_played / (training_duration / 60)) if training_duration > 0 else 0
            
            # Prepare performance data
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'training_summary': {
                    'games_played': games_played,
                    'unique_scenarios_discovered': unique_scenarios,
                    'total_scenarios_recorded': total_scenarios_recorded,
                    'training_duration_seconds': training_duration,
                    'training_duration_minutes': training_duration / 60,
                    'games_per_minute': games_per_minute
                },
                'gameplay_metrics': {
                    'hero_win_rate': hero_win_rate,
                    'villain_win_rate': villain_win_rate,
                    'showdown_rate': showdown_rate,
                    'three_bet_rate': three_bet_rate,
                    'average_pot_size_bb': avg_pot_size
                },
                'coverage_metrics': {
                    'scenario_coverage_percentage': scenario_coverage,
                    'theoretical_max_scenarios': theoretical_max_scenarios,
                    'hand_categories_discovered': len(hand_categories),
                    'theoretical_max_hand_categories': 11
                },
                'exploration_metrics': {
                    'epsilon_exploration_rate': exploration_rate,
                    'min_visit_threshold': self.min_visit_threshold,
                    'total_state_action_pairs': total_state_actions
                },
                'strategy_metrics': {
                    'hero_strategy_scenarios': len(self.strategy_sum),
                    'villain_strategy_scenarios': len(self.villain_strategy_sum)
                },
                'hand_category_distribution': hand_categories,
                'training_parameters': {
                    'epsilon_exploration': self.epsilon_exploration,
                    'min_visit_threshold': self.min_visit_threshold,
                    'tournament_survival_penalty': self.tournament_survival_penalty,
                    'enable_pruning': getattr(self, 'enable_pruning', True),
                    'regret_pruning_threshold': getattr(self, 'regret_pruning_threshold', -300.0),
                    'strategy_pruning_threshold': getattr(self, 'strategy_pruning_threshold', 0.001)
                }
            }
            
            # Save file based on format
            if output_format.lower() == 'json':
                filename = f"natural_cfr_performance_{timestamp}.json"
                filepath = performance_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(performance_data, f, indent=2, default=str)
                
                self.logger.info(f"✅ Performance summary saved as JSON to {filepath}")
            
            else:  # CSV format
                filename = f"natural_cfr_performance_{timestamp}.csv"
                filepath = performance_dir / filename
                
                # Flatten data for CSV
                csv_data = []
                
                # Add summary metrics
                csv_data.append(['Metric', 'Value', 'Category'])
                csv_data.append(['Games Played', games_played, 'Training Summary'])
                csv_data.append(['Unique Scenarios Discovered', unique_scenarios, 'Training Summary'])
                csv_data.append(['Total Scenarios Recorded', total_scenarios_recorded, 'Training Summary'])
                csv_data.append(['Training Duration (minutes)', f"{training_duration/60:.1f}", 'Training Summary'])
                csv_data.append(['Games Per Minute', f"{games_per_minute:.1f}", 'Training Summary'])
                
                csv_data.append(['Hero Win Rate', f"{hero_win_rate:.3f}", 'Gameplay Metrics'])
                csv_data.append(['Villain Win Rate', f"{villain_win_rate:.3f}", 'Gameplay Metrics'])
                csv_data.append(['Showdown Rate', f"{showdown_rate:.3f}", 'Gameplay Metrics'])
                csv_data.append(['3-bet Rate', f"{three_bet_rate:.3f}", 'Gameplay Metrics'])
                csv_data.append(['Average Pot Size (BB)', f"{avg_pot_size:.2f}", 'Gameplay Metrics'])
                
                csv_data.append(['Scenario Coverage (%)', f"{scenario_coverage:.1f}", 'Coverage Metrics'])
                csv_data.append(['Hand Categories Discovered', len(hand_categories), 'Coverage Metrics'])
                
                csv_data.append(['Epsilon Exploration Rate', exploration_rate, 'Exploration Metrics'])
                csv_data.append(['Min Visit Threshold', self.min_visit_threshold, 'Exploration Metrics'])
                csv_data.append(['Total State-Action Pairs', total_state_actions, 'Exploration Metrics'])
                
                csv_data.append(['Hero Strategy Scenarios', len(self.strategy_sum), 'Strategy Metrics'])
                csv_data.append(['Villain Strategy Scenarios', len(self.villain_strategy_sum), 'Strategy Metrics'])
                
                # Add hand category distribution
                for category, count in hand_categories.items():
                    csv_data.append([f"Hand Category: {category}", count, 'Hand Distribution'])
                
                # Write CSV
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(csv_data)
                
                self.logger.info(f"✅ Performance summary saved as CSV to {filepath}")
            
            # Log summary to console and logger
            self.logger.info("PERFORMANCE SUMMARY CREATED:")
            self.logger.info(f"  File path: {filepath.absolute()}")
            self.logger.info(f"  Games played: {games_played:,}")
            self.logger.info(f"  Training duration: {training_duration/60:.1f} minutes")
            self.logger.info(f"  Unique scenarios: {unique_scenarios}")
            self.logger.info(f"  Scenario coverage: {scenario_coverage:.1f}%")
            self.logger.info(f"  Hero win rate: {hero_win_rate:.3f}")
            self.logger.info(f"  Showdown rate: {showdown_rate:.3f}")
            self.logger.info(f"  3-bet rate: {three_bet_rate:.3f}")
            
            print(f"📊 Performance summary saved to {filepath}")
            print(f"   📈 {games_played:,} games, {unique_scenarios} scenarios, {scenario_coverage:.1f}% coverage")
            
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Failed to create performance summary: {e}"
            self.logger.error(error_msg)
            from logging_config import log_exception
            log_exception(self.logger, "Failed to create performance summary")
            print(f"❌ Failed to create performance summary: {e}")
            return None
    
    def create_final_lookup_table(self, filename=None):
        """
        Create a final lookup table CSV with all discovered scenarios, attributes, 
        final estimated EV, recommended action, and confidence.
        
        Args:
            filename: Output filename (optional, will auto-generate if None)
            
        Returns:
            str: Path to created lookup table file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"natural_cfr_final_lookup_table_{timestamp}.csv"
            
            self.logger.info(f"Creating final lookup table: {filename}...")
            
            lookup_data = []
            
            # Process hero strategies
            for scenario_key, strategy_counts in self.strategy_sum.items():
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
                        action_probs[action_name] = strategy_counts.get(action_name, 0.0) / total_count
                    
                    # Find best action and confidence
                    best_action = max(strategy_counts.items(), key=lambda x: x[1])[0]
                    confidence = max(strategy_counts.values()) / total_count
                    
                    # Estimate EV from natural scenarios
                    scenario_payoffs = [s['hero_payoff'] for s in self.natural_scenarios 
                                      if s.get('scenario_key') == scenario_key]
                    if scenario_payoffs:
                        estimated_ev = sum(scenario_payoffs) / len(scenario_payoffs)
                        ev_std = np.std(scenario_payoffs) if len(scenario_payoffs) > 1 else 0.0
                    else:
                        estimated_ev = 0.0
                        ev_std = 0.0
                    
                    # Count training games for this scenario
                    training_games = self.natural_scenario_counter.get(scenario_key, 0)
                    
                    # Get visit counts for state-action pairs
                    state_action_visits = dict(self.state_action_visits.get(scenario_key, {}))
                    total_visits = sum(state_action_visits.values())
                    
                    # Calculate action percentages
                    action_percentages = {}
                    for action_name in ACTIONS.keys():
                        action_percentages[f"{action_name}_pct"] = round(action_probs[action_name] * 100, 1)
                    
                    row = {
                        'scenario_key': scenario_key,
                        'hand_category': hand_category,
                        'position': position,
                        'stack_depth': stack_category,
                        'blinds_level': blinds_level,
                        'training_games': training_games,
                        'total_visits': total_visits,
                        'estimated_ev': round(estimated_ev, 3),
                        'ev_std_dev': round(ev_std, 3),
                        'best_action': best_action.upper(),
                        'confidence': round(confidence, 3),
                        'player': 'HERO',
                        **action_percentages
                    }
                    lookup_data.append(row)
            
            # Process villain strategies
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
                        action_probs[action_name] = strategy_counts.get(action_name, 0.0) / total_count
                    
                    # Find best action and confidence
                    best_action = max(strategy_counts.items(), key=lambda x: x[1])[0]
                    confidence = max(strategy_counts.values()) / total_count
                    
                    # Estimate EV from natural scenarios (villain perspective)
                    scenario_payoffs = [s['villain_payoff'] if 'villain_payoff' in s else -s['hero_payoff'] 
                                      for s in self.natural_scenarios 
                                      if s.get('scenario_key') == scenario_key]
                    if scenario_payoffs:
                        estimated_ev = sum(scenario_payoffs) / len(scenario_payoffs)
                        ev_std = np.std(scenario_payoffs) if len(scenario_payoffs) > 1 else 0.0
                    else:
                        estimated_ev = 0.0
                        ev_std = 0.0
                    
                    # Count training games for this scenario
                    training_games = self.natural_scenario_counter.get(scenario_key, 0)
                    
                    # Get visit counts for state-action pairs
                    state_action_visits = dict(self.state_action_visits.get(scenario_key, {}))
                    total_visits = sum(state_action_visits.values())
                    
                    # Calculate action percentages
                    action_percentages = {}
                    for action_name in ACTIONS.keys():
                        action_percentages[f"{action_name}_pct"] = round(action_probs[action_name] * 100, 1)
                    
                    row = {
                        'scenario_key': scenario_key,
                        'hand_category': hand_category,
                        'position': position,
                        'stack_depth': stack_category,
                        'blinds_level': blinds_level,
                        'training_games': training_games,
                        'total_visits': total_visits,
                        'estimated_ev': round(estimated_ev, 3),
                        'ev_std_dev': round(ev_std, 3),
                        'best_action': best_action.upper(),
                        'confidence': round(confidence, 3),
                        'player': 'VILLAIN',
                        **action_percentages
                    }
                    lookup_data.append(row)
            
            if lookup_data:
                # Sort by confidence and training games
                lookup_data.sort(key=lambda x: (x['confidence'], x['training_games']), reverse=True)
                
                # Create DataFrame and save
                df = pd.DataFrame(lookup_data)
                df.to_csv(filename, index=False)
                
                file_size_kb = Path(filename).stat().st_size / 1024
                self.logger.info(f"✅ Final lookup table saved to {filename}")
                self.logger.info(f"   Entries: {len(lookup_data)} scenario strategies")
                self.logger.info(f"   Hero strategies: {len([r for r in lookup_data if r['player'] == 'HERO'])}")
                self.logger.info(f"   Villain strategies: {len([r for r in lookup_data if r['player'] == 'VILLAIN'])}")
                self.logger.info(f"   File size: {file_size_kb:.1f} KB")
                self.logger.info(f"   File path: {Path(filename).absolute()}")
                
                print(f"📊 Final lookup table saved to {filename}")
                print(f"   📈 {len(lookup_data)} scenario strategies")
                
                # Show top strategies
                print(f"\n🎯 Top Confident Strategies:")
                for i, row in enumerate(lookup_data[:5]):
                    print(f"   {i+1}. {row['player']} {row['scenario_key']}: {row['best_action']} ({row['confidence']:.1%})")
                
                return filename
                
            else:
                self.logger.warning("No strategy data available for lookup table")
                print("❌ No strategy data available for lookup table")
                return None
                
        except Exception as e:
            error_msg = f"Failed to create final lookup table: {e}"
            self.logger.error(error_msg)
            from logging_config import log_exception
            log_exception(self.logger, "Failed to create final lookup table")
            print(f"❌ Failed to create lookup table: {e}")
            return None
    
    def export_unified_scenario_lookup_csv(self, filename="scenario_lookup_table.csv"):
        """
        Export enhanced unified scenario lookup table with comprehensive metrics.
        
        Enhanced CSV contains:
        - scenario_key: 7-column identifier (hand_category|position|stack_category|blinds_level|villain_stack_category|opponent_action|is_3bet)
        - hand_category: Type of poker hand (premium_pairs, medium_aces, etc.)
        - position: Player position (BTN, BB)
        - stack_category: Stack depth category (ultra_short, short, medium, deep, very_deep)
        - blinds_level: Blinds level (low, medium, high)
        - villain_stack_category: Opponent stack category
        - opponent_action: Last opponent action context
        - is_3bet: 3-bet scenario indicator (True/False)
        - training_games: Number of games played for this scenario
        - total_visits: Total state-action visits
        - confidence_level: Statistical confidence based on sample size
        - expected_value: Expected payoff for this scenario
        - primary_strategy: Dominant strategy (FOLD/CALL/RAISE)
        - strategy_confidence: Confidence in primary strategy
        - Action frequencies (fold_pct, check_pct, call_low_pct, etc.)
        - regret_sum: Current regret accumulation
        - exploration_rate: Recent exploration frequency
        """
        self.logger.info(f"📊 Exporting enhanced unified scenario lookup table to {filename}...")
        
        export_data = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get all unique scenarios from both hero and villain strategies
        all_scenario_keys = set()
        all_scenario_keys.update(self.strategy_sum.keys())
        all_scenario_keys.update(self.villain_strategy_sum.keys())
        all_scenario_keys.update(self.natural_scenario_counter.keys())
        
        for scenario_key in all_scenario_keys:
            # Parse enhanced 7-column scenario key
            parts = scenario_key.split("|")
            if len(parts) != 7:
                continue  # Skip old format or malformed keys
                
            hand_category, position, stack_category, blinds_level, villain_stack_category, opponent_action, is_3bet_str = parts
            is_3bet = is_3bet_str == 'True'
            
            # Calculate enhanced metrics for hero
            hero_data = self._calculate_enhanced_scenario_metrics(scenario_key, is_hero=True)
            if hero_data:
                hero_data.update({
                    'player': 'HERO',
                    'scenario_key': scenario_key,
                    'hand_category': hand_category,
                    'position': position,
                    'stack_category': stack_category,
                    'blinds_level': blinds_level,
                    'villain_stack_category': villain_stack_category,
                    'opponent_action': opponent_action,
                    'is_3bet': is_3bet,
                    'last_updated': current_time
                })
                export_data.append(hero_data)
            
            # Calculate enhanced metrics for villain  
            villain_data = self._calculate_enhanced_scenario_metrics(scenario_key, is_hero=False)
            if villain_data:
                villain_data.update({
                    'player': 'VILLAIN',
                    'scenario_key': scenario_key,
                    'hand_category': hand_category,
                    'position': position,
                    'stack_category': stack_category,
                    'blinds_level': blinds_level,
                    'villain_stack_category': villain_stack_category,
                    'opponent_action': opponent_action,
                    'is_3bet': is_3bet,
                    'last_updated': current_time
                })
                export_data.append(villain_data)
        
        # Export to CSV
        if export_data:
            try:
                df = pd.DataFrame(export_data)
                df.to_csv(filename, index=False)
                
                self.logger.info(f"✅ Exported enhanced scenario lookup table: {len(export_data)} entries")
                self.logger.info(f"   📊 Unique scenarios: {len(all_scenario_keys)}")
                self.logger.info(f"   🎯 Hero strategies: {len([e for e in export_data if e['player'] == 'HERO'])}")
                self.logger.info(f"   🎯 Villain strategies: {len([e for e in export_data if e['player'] == 'VILLAIN'])}")
                self.logger.info(f"   🔥 3-bet scenarios: {len([e for e in export_data if e['is_3bet']])}")
                
                # Calculate average confidence and expected value
                avg_confidence = sum(e['confidence_level'] for e in export_data) / len(export_data)
                avg_expected_value = sum(e['expected_value'] for e in export_data) / len(export_data)
                
                self.logger.info(f"   📈 Average confidence: {avg_confidence:.1f}%")
                self.logger.info(f"   💰 Average expected value: {avg_expected_value:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to export enhanced lookup table: {e}")
        else:
            self.logger.info("📊 No scenario data available yet for enhanced lookup table export")
    
    def export_scenario_lookup_table_csv(self, filename="scenario_lookup_table.csv"):
        """
        Export CSV in the exact format specified in the requirements:
        scenario_key,hand_cat,position,stack_cat,blinds_level,villain_stack_cat,preflop_context,visits,pct_fold,pct_call_small,pct_raise_small,pct_raise_mid,pct_raise_high
        
        - All pct_* values in [0,1], rounded to 4dp
        - Atomic writes (.tmp then os.replace)
        - Export from main process only (workers send updates via IPC)
        - Unique per scenario, not duplicated by role
        """
        self.logger.info(f"📊 Exporting scenario lookup table to {filename}...")
        
        if not self.scenario_aggregation:
            self.logger.warning("No scenario aggregation data available for export")
            return False
        
        try:
            # Use atomic write pattern: write to .tmp file then replace
            tmp_filename = filename + ".tmp"
            
            export_data = []
            
            for scenario_key, agg_data in self.scenario_aggregation.items():
                # Parse scenario key: '{hand_cat}|{position}|{stack_cat}|{blinds_level}|{villain_stack_cat}|{preflop_context}'
                parts = scenario_key.split('|')
                if len(parts) != 6:
                    self.logger.warning(f"Skipping malformed scenario key: {scenario_key}")
                    continue
                
                hand_cat, position, stack_cat, blinds_level, villain_stack_cat, preflop_context = parts
                
                visits = agg_data['visits']
                if visits == 0:
                    continue  # Skip scenarios with no visits
                
                # Calculate percentages in [0,1] range, rounded to 4dp
                pct_fold = round(agg_data['sum_fold'] / visits, 4)
                pct_call_small = round(agg_data['sum_call_small'] / visits, 4)
                pct_raise_small = round(agg_data['sum_raise_small'] / visits, 4)
                pct_raise_mid = round(agg_data['sum_raise_mid'] / visits, 4)
                pct_raise_high = round(agg_data['sum_raise_high'] / visits, 4)
                
                # Ensure percentages sum to 1.0 (handle rounding errors)
                total_pct = pct_fold + pct_call_small + pct_raise_small + pct_raise_mid + pct_raise_high
                if total_pct > 0:
                    # Normalize to ensure sum = 1.0
                    pct_fold = round(pct_fold / total_pct, 4)
                    pct_call_small = round(pct_call_small / total_pct, 4)
                    pct_raise_small = round(pct_raise_small / total_pct, 4)
                    pct_raise_mid = round(pct_raise_mid / total_pct, 4)
                    pct_raise_high = round(pct_raise_high / total_pct, 4)
                
                row_data = {
                    'scenario_key': scenario_key,
                    'hand_cat': hand_cat,
                    'position': position,
                    'stack_cat': stack_cat,
                    'blinds_level': blinds_level,
                    'villain_stack_cat': villain_stack_cat,
                    'preflop_context': preflop_context,
                    'visits': visits,
                    'pct_fold': pct_fold,
                    'pct_call_small': pct_call_small,
                    'pct_raise_small': pct_raise_small,
                    'pct_raise_mid': pct_raise_mid,
                    'pct_raise_high': pct_raise_high
                }
                export_data.append(row_data)
            
            if not export_data:
                self.logger.warning("No valid scenario data to export")
                return False
            
            # Write to temporary file first (atomic write)
            df = pd.DataFrame(export_data)
            df.to_csv(tmp_filename, index=False)
            
            # Atomic replace operation
            import os
            os.replace(tmp_filename, filename)
            
            self.logger.info(f"✅ Exported {len(export_data)} scenarios to {filename}")
            self.logger.info(f"   Unique scenarios (role-agnostic): {len(self.scenario_aggregation)}")
            
            # Log variety statistics
            categories_found = len(set(row['hand_cat'] for row in export_data))
            contexts_found = len(set(row['preflop_context'] for row in export_data))
            self.logger.info(f"   Hand categories: {categories_found}, Preflop contexts: {contexts_found}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export scenario lookup table: {e}")
            # Clean up tmp file if it exists
            try:
                import os
                if os.path.exists(tmp_filename):
                    os.remove(tmp_filename)
            except:
                pass
            return False
    
    def _calculate_enhanced_scenario_metrics(self, scenario_key, is_hero=True):
        """
        Calculate comprehensive metrics for a scenario.
        
        Args:
            scenario_key: Scenario identifier
            is_hero: Whether calculating for hero or villain
            
        Returns:
            dict: Enhanced metrics or None if insufficient data
        """
        # Select appropriate strategy data
        if is_hero:
            strategy_sum = self.strategy_sum.get(scenario_key, {})
            regret_sum = self.regret_sum.get(scenario_key, {})
        else:
            strategy_sum = self.villain_strategy_sum.get(scenario_key, {})
            regret_sum = self.villain_regret_sum.get(scenario_key, {})
        
        if not strategy_sum:
            return None
        
        # Calculate basic metrics
        training_games = self.natural_scenario_counter.get(scenario_key, 0)
        total_strategy_sum = sum(strategy_sum.values())
        
        if total_strategy_sum == 0:
            return None
        
        # Calculate action frequencies
        action_frequencies = {}
        for action in ['fold', 'check', 'call_low', 'call_mid', 'call_high', 'raise_low', 'raise_mid', 'raise_high', 'shove']:
            action_frequencies[f'{action}_pct'] = (strategy_sum.get(action, 0) / total_strategy_sum) * 100
        
        # Determine primary strategy
        fold_pct = action_frequencies['fold_pct']
        call_pct = sum(action_frequencies[f'{action}_pct'] for action in ['check', 'call_low', 'call_mid', 'call_high'])
        raise_pct = sum(action_frequencies[f'{action}_pct'] for action in ['raise_low', 'raise_mid', 'raise_high', 'shove'])
        
        if fold_pct >= max(call_pct, raise_pct):
            primary_strategy = 'FOLD'
            strategy_confidence = fold_pct
        elif call_pct >= raise_pct:
            primary_strategy = 'CALL'
            strategy_confidence = call_pct
        else:
            primary_strategy = 'RAISE'
            strategy_confidence = raise_pct
        
        # Calculate confidence level based on sample size
        confidence_level = min(100, (training_games / 50) * 100)  # 100% confidence at 50+ games
        
        # Calculate expected value from recent scenarios
        expected_value = self._calculate_expected_value(scenario_key, is_hero)
        
        # Calculate exploration rate
        total_visits = sum(self.state_action_visits[scenario_key].values())
        exploration_rate = min(1.0, total_visits / max(1, training_games * 2))  # Estimate exploration frequency
        
        # Calculate regret sum
        total_regret = sum(regret_sum.values()) if regret_sum else 0.0
        
        return {
            'training_games': training_games,
            'total_visits': total_visits,
            'confidence_level': round(confidence_level, 1),
            'expected_value': round(expected_value, 4),
            'primary_strategy': primary_strategy,
            'strategy_confidence': round(strategy_confidence, 1),
            'regret_sum': round(total_regret, 2),
            'exploration_rate': round(exploration_rate, 3),
            **{k: round(v, 2) for k, v in action_frequencies.items()}
        }
    
    def _calculate_expected_value(self, scenario_key, is_hero):
        """Calculate expected value for a scenario based on recent outcomes."""
        # Find recent scenarios matching this key
        matching_scenarios = [s for s in self.natural_scenarios[-100:] if s['scenario_key'] == scenario_key]
        
        if not matching_scenarios:
            return 0.0
        
        # Calculate average payoff
        total_payoff = sum(s['hero_payoff'] for s in matching_scenarios)
        return total_payoff / len(matching_scenarios)
    
    def _cleanup_memory(self):
        """Perform periodic memory cleanup to prevent memory bloat."""
        # Clear caches if they get too large
        if len(self.scenario_key_cache) > 5000:
            # Keep only the most recent 2500 entries
            keys_to_remove = list(self.scenario_key_cache.keys())[:-2500]
            for key in keys_to_remove:
                del self.scenario_key_cache[key]
        
        if len(self.action_cache) > 3000:
            # Keep only the most recent 1500 entries  
            keys_to_remove = list(self.action_cache.keys())[:-1500]
            for key in keys_to_remove:
                del self.action_cache[key]
        
        if len(self.realistic_scenario_cache) > 10000:
            # Keep only the most recent 5000 entries
            keys_to_remove = list(self.realistic_scenario_cache.keys())[:-5000]
            for key in keys_to_remove:
                del self.realistic_scenario_cache[key]
        
        # Limit natural scenarios history to prevent memory bloat
        if len(self.natural_scenarios) > 10000:
            self.natural_scenarios = self.natural_scenarios[-5000:]  # Keep recent 5000
        
        self.logger.debug("🧹 Memory cleanup completed")
    
    def archive_old_files(self):
        """
        Archive old files and folders that are no longer used by the new simulation model.
        Moves them to 'archived_files' folder.
        
        Returns:
            list: List of archived files/folders
        """
        try:
            # Create archive directory if it doesn't exist
            archive_dir = Path("archivedfileslocation")
            archive_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Archiving old files to {archive_dir.absolute()}...")
            
            archived_items = []
            
            # Define patterns and specific files to archive
            # These are files from the old fixed-scenario system or no longer used files
            archive_patterns = [
                # Old scenario files (not natural_*)
                "demo_scenarios*.csv",
                "test_scenarios*.csv",
                "scenarios_*.csv",
                # Old strategy files (not natural_*)
                "demo_strategies*.csv", 
                "test_strategies*.csv",
                "strategies_*.csv",
                # Old checkpoint files (not natural_cfr_*)
                "cfr_checkpoint*.pkl",
                "checkpoint*.pkl",
                # Old performance files
                "performance*.csv",
                "metrics*.csv",
                # GCP/Cloud specific files (if not being used)
                "gcp_*.csv",
                "cloud_*.csv",
                # Legacy training files
                "enhanced_cfr_preflop_generator.py",  # v1 files (if v2 exists)
                "enhanced_cfr_trainer.py",  # v1 files (if v2 exists)
                # Old analysis files (not from current run)
                "analysis_scenarios_*.csv",
                "analysis_strategies_*.csv"
            ]
            
            # Also check for specific folders to archive
            archive_folders = [
                "archivefolder",  # Existing archive folder can be moved
                "archiveworks",   # Existing archive folder can be moved
                "old_checkpoints",
                "legacy_files",
                "deprecated"
            ]
            
            current_dir = Path(".")
            
            # Archive matching files
            for pattern in archive_patterns:
                for file_path in current_dir.glob(pattern):
                    if file_path.is_file():
                        try:
                            # Check if it's a recent natural CFR file - don't archive those
                            if ("natural_" in file_path.name and 
                                any(recent in file_path.name for recent in [
                                    datetime.now().strftime("%Y%m%d"),
                                    (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                                ])):
                                continue
                            
                            dest_path = archive_dir / file_path.name
                            if not dest_path.exists():
                                file_path.rename(dest_path)
                                archived_items.append(str(file_path))
                                self.logger.info(f"   📦 Archived file: {file_path} -> {dest_path}")
                        except Exception as e:
                            self.logger.warning(f"Could not archive {file_path}: {e}")
            
            # Archive matching folders
            for folder_name in archive_folders:
                folder_path = current_dir / folder_name
                if folder_path.exists() and folder_path.is_dir():
                    try:
                        dest_path = archive_dir / folder_name
                        if not dest_path.exists():
                            folder_path.rename(dest_path)
                            archived_items.append(str(folder_path))
                            self.logger.info(f"   📦 Archived folder: {folder_path} -> {dest_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not archive folder {folder_path}: {e}")
            
            # Archive specific old files if they exist
            specific_old_files = [
                "cfr_lookup_table.csv",
                "cfr_performance.csv", 
                "preflop_cfr.py",
                "old_trainer.py",
                "legacy_scenarios.csv"
            ]
            
            for old_file in specific_old_files:
                old_path = current_dir / old_file
                if old_path.exists():
                    try:
                        dest_path = archive_dir / old_file
                        if not dest_path.exists():
                            old_path.rename(dest_path)
                            archived_items.append(str(old_path))
                            self.logger.info(f"   📦 Archived legacy file: {old_path} -> {dest_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not archive {old_path}: {e}")
            
            if archived_items:
                self.logger.info(f"✅ Archiving completed successfully")
                self.logger.info(f"   Archive location: {archive_dir.absolute()}")
                self.logger.info(f"   Items archived: {len(archived_items)}")
                for item in archived_items:
                    self.logger.info(f"     - {item}")
                
                print(f"📦 Archived {len(archived_items)} old files/folders to {archive_dir}")
                print(f"   📁 Archive location: {archive_dir.absolute()}")
                
                if len(archived_items) <= 10:  # Show all if not too many
                    for item in archived_items:
                        print(f"     - {Path(item).name}")
                else:
                    for item in archived_items[:5]:
                        print(f"     - {Path(item).name}")
                    print(f"     ... and {len(archived_items) - 5} more items")
            else:
                self.logger.info("No old files found to archive")
                print("📦 No old files found to archive")
            
            return archived_items
            
        except Exception as e:
            error_msg = f"Failed to archive old files: {e}"
            self.logger.error(error_msg)
            from logging_config import log_exception
            log_exception(self.logger, "Failed to archive old files")
            print(f"❌ Failed to archive old files: {e}")
            return []


if __name__ == "__main__":
    
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