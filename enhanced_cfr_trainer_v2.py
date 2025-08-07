# enhanced_cfr_trainer_v2.py - CFR training with dynamic opponent betting

"""
Enhanced CFR Trainer for Preflop Poker with Sequential Training Support

This module provides advanced CFR training capabilities including:
1. Enhanced CFR training with dynamic opponent betting
2. Sequential scenario training with multiple rollouts per visit
3. Configurable stopping conditions and convergence checking
4. Comprehensive performance tracking and statistics

Key Features:
- Scenario keys: hand_category|position|stack_category|blinds_level (no bet_size_category)
- Dynamic opponent betting: Randomized bet sizes during training
- Action mapping: call/raise buckets determined by actual bet size vs stack ratio
- Robust learning: Model learns strategies across varied opponent bet distributions
- Balanced sampling: Ensures proportional coverage across hand categories
- Sequential training: Process scenarios in order with multiple rollouts per visit
- Advanced stopping conditions: Minimum rollouts requirement + regret stabilization

Training Process (Enhanced CFR):
1. Select scenario from deterministic set of 330 combinations
2. Generate dynamic betting context (opponent action, bet size)
3. Map available actions based on actual bet size vs stack
4. Apply CFR regret matching for strategy selection
5. Update regrets for counterfactual actions

Sequential Training Process:
1. Process scenarios sequentially (not randomly)
2. For each scenario iteration, perform multiple rollouts with different random contexts
3. Average returns from multiple rollouts for regret updates
4. Check convergence using moving average regret change over configurable window
5. Allow early stopping only after minimum rollouts completed
6. Comprehensive logging with rollout distribution statistics

Classes:
- EnhancedCFRTrainer: Base CFR trainer with balanced sampling and dynamic betting
- SequentialScenarioTrainer: Enhanced trainer with sequential processing and multiple rollouts
"""

from enhanced_cfr_preflop_generator_v2 import (
    generate_enhanced_scenarios, simulate_enhanced_showdown, cards_to_str,
    ACTIONS, get_available_actions
)
from treys import Card, Deck, Evaluator
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import time
import pickle

class EnhancedCFRTrainer:
    """
    Enhanced CFR with tournament survival, stack awareness, and bet sizing.
    
    Now includes advanced pruning techniques:
    1. Regret-based pruning: Skip actions with regret below threshold
    2. Strategy pruning: Exclude low-probability actions from exports
    3. Action space pruning: Context-aware action filtering
    """
    
    def __init__(self, scenarios=None, enable_pruning=True, regret_pruning_threshold=-300.0, 
                 strategy_pruning_threshold=0.001):
        """
        Initialize Enhanced CFR Trainer with optional pruning capabilities.
        
        Args:
            scenarios: Pre-generated scenarios (optional)
            enable_pruning: Enable all pruning techniques (default True)
            regret_pruning_threshold: Threshold for regret-based pruning (default -300.0)
            strategy_pruning_threshold: Threshold for strategy pruning (default 0.001)
        """
        # CFR data structures - now with variable action counts
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.scenario_counter = Counter()
        self.iterations = 0

        # Pruning parameters
        self.enable_pruning = enable_pruning
        self.regret_pruning_threshold = regret_pruning_threshold
        self.strategy_pruning_threshold = strategy_pruning_threshold
        
        # Pruning tracking
        self.pruned_actions = defaultdict(set)  # Track pruned actions per scenario
        self.pruning_statistics = {
            'regret_pruned_count': 0,
            'strategy_pruned_count': 0,
            'actions_restored_count': 0,
            'total_pruning_events': 0
        }
        self.restoration_chances = defaultdict(int)  # Track restoration attempts per scenario

        # Enhanced tracking
        self.stack_survival_rate = []
        self.tournament_results = []
        self.equity_by_stack = defaultdict(list)
        
        # Performance metrics tracking
        self.performance_metrics = []
        self.start_time = None
        self.last_iteration_time = None
        
        # Generate enhanced scenarios, or use provided ones
        if scenarios is not None:
            self.scenarios = scenarios
            print(f"🚀 Using provided {len(scenarios)} scenarios...")
        else:
            from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
            self.scenarios = generate_enhanced_scenarios()
            print(f"🚀 Generated {len(self.scenarios)} scenarios...")

        # Balanced hand category sampling (after scenarios are loaded)
        self.hand_category_visits = defaultdict(int)
        self.scenarios_by_category = self._group_scenarios_by_category()
            
        print(f"🏆 Enhanced CFR Trainer Initialized!")
        print(f"📊 Training scenarios: {len(self.scenarios):,}")
        print(f"📈 Hand categories: {len(self.scenarios_by_category)} balanced groups")
        if self.enable_pruning:
            print(f"✂️ Pruning enabled: regret_threshold={self.regret_pruning_threshold}, strategy_threshold={self.strategy_pruning_threshold}")

    def get_relevant_actions(self, scenario, betting_context):
        """
        Action Space Pruning: Filter actions based on contextual relevance.
        
        Args:
            scenario: Current scenario dictionary
            betting_context: Dictionary with betting information
            
        Returns:
            List of contextually relevant actions
        """
        if not self.enable_pruning:
            # Return all standard actions if pruning disabled
            return ["fold", "call_small", "call_mid", "call_high", "raise_small", "raise_mid", "raise_high"]
        
        relevant_actions = []
        hand_category = scenario.get('hand_category', 'unknown')
        stack_category = scenario.get('stack_category', 'medium')
        opponent_bet = betting_context.get('bet_size', 0)
        
        # Always include fold as an option
        relevant_actions.append("fold")
        
        # Contextual filtering logic
        if hand_category in ['trash', 'weak_aces']:
            # Trash hands: mostly fold/call only, limited raising
            if opponent_bet == 0:  # No bet to call
                relevant_actions.extend(["raise_small"])
            else:
                relevant_actions.extend(["call_small"])
        elif hand_category in ['premium_pairs', 'premium_aces']:
            # Premium hands: all actions available
            if opponent_bet == 0:
                relevant_actions.extend(["raise_small", "raise_mid", "raise_high"])
            else:
                relevant_actions.extend(["call_small", "call_mid", "call_high", "raise_small", "raise_mid", "raise_high"])
        else:
            # Medium hands: moderate action selection
            if opponent_bet == 0:
                relevant_actions.extend(["call_small", "raise_small", "raise_mid"])
            else:
                relevant_actions.extend(["call_small", "call_mid", "raise_small"])
        
        # Stack-based filtering
        if stack_category in ['ultra_short', 'short']:
            # Short stacks: remove high calls/raises, prefer all-in equivalent actions
            relevant_actions = [action for action in relevant_actions 
                              if action not in ['call_high', 'raise_high']]
        
        # Log pruning event
        if len(relevant_actions) < 7:  # Fewer than all possible actions
            self.pruning_statistics['total_pruning_events'] += 1
            
        return list(set(relevant_actions))  # Remove duplicates

    def get_strategy(self, scenario_key, available_actions):
        """
        Get strategy using regret matching for available actions with optional regret-based pruning.
        
        Args:
            scenario_key: Unique identifier for the scenario
            available_actions: List of available actions
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        regrets = self.regret_sum[scenario_key]
        
        # Apply regret-based pruning if enabled
        if self.enable_pruning:
            # Check for 1% restoration chance for pruned actions
            if random.random() < 0.01:
                self._restore_pruned_actions(scenario_key)
            
            # Filter out actions with regret below threshold 
            filtered_actions = []
            for action in available_actions:
                action_regret = regrets.get(action, 0)
                
                # Include action if:
                # 1. Regret is above threshold, OR
                # 2. Action was previously pruned but not in current pruned set (restored)
                if action_regret >= self.regret_pruning_threshold:
                    filtered_actions.append(action)
                    # Remove from pruned set if it was there (regret improved)
                    self.pruned_actions[scenario_key].discard(action)
                else:
                    # Action is below threshold - add to pruned set
                    if action not in self.pruned_actions[scenario_key]:
                        self.pruned_actions[scenario_key].add(action)
                        self.pruning_statistics['regret_pruned_count'] += 1
            
            # Ensure at least one action remains
            if not filtered_actions:
                # Keep the action with the highest regret, even if below threshold
                best_action = max(available_actions, key=lambda a: regrets.get(a, 0))
                filtered_actions = [best_action]
                # Remove it from pruned set since we're keeping it
                self.pruned_actions[scenario_key].discard(best_action)
                
            available_actions = filtered_actions
        
        # Only consider regrets for available (non-pruned) actions
        action_regrets = [regrets.get(action, 0) for action in available_actions]
        positive_regrets = np.maximum(action_regrets, 0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            strategy_probs = positive_regrets / regret_sum
        else:
            # Uniform over available actions
            strategy_probs = np.ones(len(available_actions)) / len(available_actions)
        
        # Create strategy dict
        strategy = {}
        for i, action in enumerate(available_actions):
            strategy[action] = strategy_probs[i]
            
        return strategy
    
    def _restore_pruned_actions(self, scenario_key):
        """
        Restore pruned actions with 1% chance for exploration.
        """
        if scenario_key in self.pruned_actions and self.pruned_actions[scenario_key]:
            # Restore one random pruned action
            action_to_restore = random.choice(list(self.pruned_actions[scenario_key]))
            self.pruned_actions[scenario_key].discard(action_to_restore)
            self.pruning_statistics['actions_restored_count'] += 1
            self.restoration_chances[scenario_key] += 1

    def _group_scenarios_by_category(self):
        """Group scenarios by hand category for balanced sampling"""
        scenarios_by_category = defaultdict(list)
        for scenario in self.scenarios:
            category = scenario['hand_category']
            scenarios_by_category[category].append(scenario)
        return dict(scenarios_by_category)
    
    def select_balanced_scenario(self):
        """
        Select scenario using stratified sampling to ensure balanced hand category coverage.
        Prioritizes categories with fewer visits to maintain proportional training.
        """
        if not self.hand_category_visits:
            # First iteration - select random category
            category = random.choice(list(self.scenarios_by_category.keys()))
        else:
            # Calculate weights based on visit counts (inverse weighting)
            max_visits = max(self.hand_category_visits.values())
            category_weights = {}
            
            for category in self.scenarios_by_category.keys():
                current_visits = self.hand_category_visits.get(category, 0)
                # Inverse weighting - less visited categories get higher weight
                weight = max_visits + 1 - current_visits
                category_weights[category] = max(weight, 1)  # Minimum weight of 1
            
            # Select category based on weights
            categories = list(category_weights.keys())
            weights = list(category_weights.values())
            weight_probs = np.array(weights) / sum(weights)
            category = np.random.choice(categories, p=weight_probs)
        
        # Select random scenario from chosen category
        scenario = random.choice(self.scenarios_by_category[category])
        
        # Update category visit count
        self.hand_category_visits[category] += 1
        
        return scenario

    def sample_action(self, strategy, available_actions):
        """Sample action from strategy probabilities"""
        probs = [strategy[action] for action in available_actions]
        chosen_idx = np.random.choice(len(available_actions), p=probs)
        return available_actions[chosen_idx]

    def play_enhanced_scenario(self, scenario):
        """
        Play enhanced scenario with dynamic opponent betting and integrated pruning techniques.
        
        Now includes:
        1. Action space pruning before strategy calculation
        2. Dynamic betting context generation
        3. Enhanced CFR strategy calculation
        """
        scenario_key = self.get_scenario_key(scenario)
        
        # Generate dynamic betting context (replaces static bet_size_category)
        from enhanced_cfr_preflop_generator_v2 import generate_dynamic_betting_context
        betting_context = generate_dynamic_betting_context(scenario)
        
        # Get initially available actions based on the dynamic betting context
        initial_available_actions = betting_context["available_actions"]
        
        # Apply action space pruning to filter contextually relevant actions
        if self.enable_pruning:
            relevant_actions = self.get_relevant_actions(scenario, betting_context)
            # Take intersection of initially available and contextually relevant actions
            available_actions = [action for action in initial_available_actions if action in relevant_actions]
            # Ensure at least one action remains
            if not available_actions:
                available_actions = initial_available_actions[:1]  # Keep at least one action
        else:
            available_actions = initial_available_actions
        
        # Hero's decision using CFR strategy (now with pruned action space)
        hero_strategy = self.get_strategy(scenario_key, available_actions)
        hero_action = self.sample_action(hero_strategy, available_actions)
        
        # Generate villain and their action
        villain_cards = self.generate_villain_hand(scenario['hero_cards_int'])
        villain_action = self.get_enhanced_villain_action(villain_cards, scenario, betting_context)
        
        # Calculate enhanced payoff with dynamic betting context
        payoff_result = self.calculate_enhanced_payoff(scenario, hero_action, villain_action, villain_cards, betting_context)
        
        # Update regrets for all available (potentially pruned) actions
        self.update_enhanced_regrets(scenario_key, hero_action, hero_strategy, 
                                   payoff_result, available_actions)
        
        # Update strategy sum
        for action in available_actions:
            self.strategy_sum[scenario_key][action] += hero_strategy[action]
        
        return {
            'scenario_key': scenario_key,
            'hero_action': hero_action,
            'villain_action': villain_action,
            'payoff': payoff_result['payoff'],
            'hero_stack_after': payoff_result['hero_stack_after'],
            'busted': payoff_result['busted'],
            'available_actions': available_actions,
            'initial_actions': initial_available_actions,  # Track original vs pruned
            'betting_context': betting_context
        }

    def calculate_enhanced_payoff(self, scenario, hero_action, villain_action, villain_cards, betting_context):
        """Calculate payoff with dynamic betting context and stack survival considerations"""
        hero_stack_before = scenario['hero_stack_bb']
        bet_amount = betting_context['bet_to_call_bb']
        
        # Determine bet amounts based on actions
        hero_bet = self.get_bet_amount(hero_action, hero_stack_before, bet_amount)
        villain_bet = self.get_bet_amount(villain_action, hero_stack_before, bet_amount)
        
        # Simulate showdown
        showdown_result = simulate_enhanced_showdown(
            scenario['hero_cards_int'], villain_cards, 
            hero_action, villain_action,
            hero_stack_before, scenario['villain_stack_bb'], 
            max(hero_bet, villain_bet)
        )
        
        # Calculate new stack size
        stack_change = showdown_result['hero_stack_change']
        hero_stack_after = max(0, hero_stack_before + stack_change)
        busted = (hero_stack_after == 0)
        
        # Base payoff
        base_payoff = stack_change / hero_stack_before  # Normalize by stack size
        
        # Stack-based adjustments
        stack_payoff = self.apply_stack_adjustments(
            base_payoff, hero_stack_before, hero_stack_after, busted
        )
        
        return {
            'payoff': stack_payoff,
            'hero_stack_after': hero_stack_after,
            'busted': busted,
            'stack_change': stack_change
        }

    def get_bet_amount(self, action, stack_size, current_bet):
        """Convert action to bet amount"""
        if action == "fold":
            return 0
        elif action in ["call_small", "call_large"]:
            return current_bet
        elif action == "raise_small":
            return current_bet * 2.5
        elif action == "raise_large":
            return current_bet * 4
        elif action == "all_in":
            return stack_size
        else:
            return current_bet

    def apply_stack_adjustments(self, base_payoff, stack_before, stack_after, busted):
        """Apply stack-based payoff adjustments (replacing tournament logic)"""
        stack_payoff = base_payoff
        
        # Massive penalty for busting (severity based on stack size)
        if busted:
            if stack_before <= 15:  # Short stack bust
                stack_payoff = -8.0   # Very bad to bust when short
            elif stack_before <= 30:  # Medium stack bust  
                stack_payoff = -5.0   # Bad to bust with medium stack
            else:
                stack_payoff = -3.0   # Standard bust penalty
        
        # Survival bonus for short stacks
        if stack_before <= 15 and stack_after > stack_before:
            survival_bonus = 2.0 * (stack_after - stack_before) / stack_before
            stack_payoff += survival_bonus
        
        # Stack preservation bonus for short stacks
        if stack_before <= 30 and not busted:
            if stack_after >= stack_before * 0.8:  # Didn't lose much
                preservation_bonus = 0.5
                stack_payoff += preservation_bonus
        
        # Chip accumulation bonus for deeper stacks
        if stack_before > 50 and stack_after > stack_before * 1.5:
            accumulation_bonus = 1.0
            stack_payoff += accumulation_bonus
        
        return stack_payoff

    def generate_villain_hand(self, hero_cards):
        """Generate random villain hand"""
        deck = Deck()
        for card in hero_cards:
            if card in deck.cards:
                deck.cards.remove(card)
        return deck.draw(2)

    def get_enhanced_villain_action(self, villain_cards, scenario, betting_context):
        """Enhanced villain action based on stack context and dynamic betting"""
        try:
            villain_equity = self.estimate_villain_equity(villain_cards)
            villain_stack = scenario['villain_stack_bb']
            
            # Adjust strategy based on stack size
            if villain_stack <= 15:  # Short stack - push/fold
                if villain_equity > 0.45:
                    return "all_in"
                else:
                    return "fold"
            elif villain_stack <= 30:  # Medium short stack - tighter play
                if villain_equity > 0.6:
                    return "all_in"
                elif villain_equity > 0.5:
                    return "call_small"
                else:
                    return "fold"
            
            else:  # Normal stack - standard play
                if villain_equity > 0.65:
                    return random.choice(["raise_large", "raise_small"])
                elif villain_equity > 0.45:
                    return random.choice(["call_small", "raise_small"])
                else:
                    return random.choice(["fold", "call_small"]) if random.random() < 0.8 else "fold"
                    
        except:
            return random.choice(["fold", "call_small", "raise_small"])

    def estimate_villain_equity(self, villain_cards, simulations=50):
        """Quick equity estimation for villain"""
        wins = 0
        for _ in range(simulations):
            deck = Deck()
            for card in villain_cards:
                if card in deck.cards:
                    deck.cards.remove(card)
            
            # Random opponent
            hero_cards = deck.draw(2)
            board = deck.draw(5)
            
            evaluator = Evaluator()
            villain_score = evaluator.evaluate(villain_cards, board)
            hero_score = evaluator.evaluate(hero_cards, board)
            
            if villain_score < hero_score:
                wins += 1
        
        return wins / simulations

    def update_enhanced_regrets(self, scenario_key, action_taken, strategy, 
                              payoff_result, available_actions):
        """
        Enhanced regret update considering all available actions with regret-based pruning support.
        
        Args:
            scenario_key: Unique identifier for the scenario
            action_taken: Action that was actually taken
            strategy: Strategy probabilities used
            payoff_result: Result of the action
            available_actions: Actions that were available
        """
        actual_payoff = payoff_result['payoff']
        
        # Estimate counterfactual payoffs for other actions
        for action in available_actions:
            # Skip regret updates for actions below pruning threshold (if pruning enabled)
            if (self.enable_pruning and 
                action in self.pruned_actions[scenario_key] and 
                self.regret_sum[scenario_key][action] < self.regret_pruning_threshold):
                continue  # Skip updating regret for pruned actions
            
            if action == action_taken:
                # Actual result
                self.regret_sum[scenario_key][action] += 0  # No regret for chosen action
            else:
                # Estimate what would have happened
                estimated_payoff = self.estimate_counterfactual_payoff(
                    action, payoff_result, available_actions
                )
                
                # Regret = what I could have got - what I actually got
                regret = estimated_payoff - actual_payoff
                self.regret_sum[scenario_key][action] += regret

    def estimate_counterfactual_payoff(self, alternative_action, actual_result, available_actions):
        """Estimate payoff if we had taken a different action"""
        actual_payoff = actual_result['payoff']
        
        # Enhanced heuristic estimates for new action set
        if alternative_action == "fold":
            return -0.1  # Small loss from folding
        elif alternative_action == "call_small":
            return actual_payoff * 0.7  # Conservative, small risk
        elif alternative_action == "call_mid":
            return actual_payoff * 0.8  # Moderate risk
        elif alternative_action == "call_high":
            return actual_payoff * 0.9  # Higher risk call
        elif alternative_action == "raise_small":
            return actual_payoff * 1.1  # Modest aggression
        elif alternative_action == "raise_mid":
            return actual_payoff * 1.2  # Standard aggression
        elif alternative_action == "raise_high":
            return actual_payoff * 1.4  # High aggression/risk
        else:
            return actual_payoff

    def get_scenario_key(self, scenario):
        """Enhanced scenario key without bet_size_category"""
        return (f"{scenario['hand_category']}|{scenario['hero_position']}|"
                f"{scenario['stack_category']}|{scenario['blinds_level']}")

    def export_strategies_to_csv(self, filename="enhanced_cfr_strategies.csv"):
        """
        Export all learned strategies to CSV with comprehensive scenario details.
        Includes probabilities for each action, scenario details (hole cards, position, 
        stack depth, blinds level), and best action as determined by model.
        
        CSV columns include:
        - scenario_key: Unique identifier for the scenario
        - hand_category: Broad category of hole cards (premium_pairs, medium_aces, etc.)
        - example_hands: Sample hands from this category  
        - position: Hero's position (BTN/BB)
        - stack_depth: Stack size category (ultra_short, short, medium, deep, very_deep)
        - blinds_level: Blinds level (low, medium, high)
        - training_games: Number of training iterations for this scenario
        - best_action: Recommended action (FOLD, CALL_SMALL, CALL_MID, CALL_HIGH, RAISE_SMALL, RAISE_MID, RAISE_HIGH)
        - confidence: Probability of best action (0-1)
        - fold_prob through raise_high_prob: Probability of each action
        """
        import pandas as pd
        from enhanced_cfr_preflop_generator_v2 import ACTIONS, PREFLOP_HAND_RANGES
        
        print(f"📊 Exporting strategies to {filename}...")
        
        # Collect all strategy data
        export_data = []
        
        for scenario_key, strategy_counts in self.strategy_sum.items():
            if sum(strategy_counts.values()) > 0:  # Only export scenarios with data
                
                # Parse scenario key (bet_size_category removed, blinds_level added)
                parts = scenario_key.split("|")
                if len(parts) >= 4:
                    hand_category = parts[0]
                    position = parts[1] 
                    stack_category = parts[2]
                    blinds_level = parts[3]
                else:
                    continue  # Skip malformed keys
                
                # Get example hands for this category
                example_hands = ""
                if hand_category in PREFLOP_HAND_RANGES:
                    examples = PREFLOP_HAND_RANGES[hand_category][:3]  # First 3 examples
                    example_hands = ", ".join(examples)
                
    def export_strategies_to_csv(self, filename="enhanced_cfr_strategies.csv"):
        """
        Export all learned strategies to CSV with comprehensive scenario details and strategy pruning.
        Includes probabilities for each action, scenario details (hole cards, position, 
        stack depth, blinds level), and best action as determined by model.
        
        Strategy Pruning: Excludes actions below strategy_pruning_threshold and renormalizes probabilities.
        
        CSV columns include:
        - scenario_key: Unique identifier for the scenario
        - hand_category: Broad category of hole cards (premium_pairs, medium_aces, etc.)
        - example_hands: Sample hands from this category  
        - position: Hero's position (BTN/BB)
        - stack_depth: Stack size category (ultra_short, short, medium, deep, very_deep)
        - blinds_level: Blinds level (low, medium, high)
        - training_games: Number of training iterations for this scenario
        - actions_pruned_count: Number of actions pruned due to low probability
        - best_action: Recommended action (FOLD, CALL_SMALL, CALL_MID, CALL_HIGH, RAISE_SMALL, RAISE_MID, RAISE_HIGH)
        - confidence: Probability of best action (0-1)
        - fold_prob through raise_high_prob: Probability of each action
        """
        import pandas as pd
        from enhanced_cfr_preflop_generator_v2 import ACTIONS, PREFLOP_HAND_RANGES
        
        print(f"📊 Exporting strategies to {filename}...")
        
        # Collect all strategy data
        export_data = []
        total_actions_pruned = 0
        
        for scenario_key, strategy_counts in self.strategy_sum.items():
            if sum(strategy_counts.values()) > 0:  # Only export scenarios with data
                
                # Parse scenario key (bet_size_category removed, blinds_level added)
                parts = scenario_key.split("|")
                if len(parts) >= 4:
                    hand_category = parts[0]
                    position = parts[1] 
                    stack_category = parts[2]
                    blinds_level = parts[3]
                else:
                    continue  # Skip malformed keys
                
                # Get example hands for this category
                example_hands = ""
                if hand_category in PREFLOP_HAND_RANGES:
                    examples = PREFLOP_HAND_RANGES[hand_category][:3]  # First 3 examples
                    example_hands = ", ".join(examples)
                
                # Calculate average strategy (normalized probabilities)
                total_count = sum(strategy_counts.values())
                raw_action_probs = {}
                
                # Calculate raw probabilities first
                for action, count in strategy_counts.items():
                    if action in ACTIONS:
                        raw_action_probs[action] = count / total_count
                
                # Apply strategy pruning if enabled
                actions_pruned_count = 0
                filtered_probs = {}
                
                if self.enable_pruning:
                    # Filter out low-probability actions
                    for action, prob in raw_action_probs.items():
                        if prob >= self.strategy_pruning_threshold:
                            filtered_probs[action] = prob
                        else:
                            actions_pruned_count += 1
                            self.pruning_statistics['strategy_pruned_count'] += 1
                    
                    # Ensure at least one action remains
                    if not filtered_probs:
                        # Keep the highest probability action
                        best_action = max(raw_action_probs.items(), key=lambda x: x[1])[0]
                        filtered_probs[best_action] = raw_action_probs[best_action]
                        actions_pruned_count -= 1  # Don't count the kept action as pruned
                    
                    # Renormalize probabilities
                    if filtered_probs:
                        total_filtered = sum(filtered_probs.values())
                        for action in filtered_probs:
                            filtered_probs[action] /= total_filtered
                else:
                    filtered_probs = raw_action_probs
                
                total_actions_pruned += actions_pruned_count
                
                # Initialize all action probabilities to 0
                action_probs = {}
                for action_name in ACTIONS.keys():
                    action_probs[f"{action_name}_prob"] = 0.0
                
                # Fill in filtered probabilities
                for action, prob in filtered_probs.items():
                    action_probs[f"{action}_prob"] = prob
                
                # Determine best action from filtered probabilities
                if filtered_probs:
                    best_action = max(filtered_probs.items(), key=lambda x: x[1])[0]
                    best_action_confidence = max(filtered_probs.values())
                else:
                    best_action = "fold"  # Default fallback
                    best_action_confidence = 1.0
                
                # Get training count for this scenario
                training_games = self.scenario_counter.get(scenario_key, 0)
                
                # Build export row (bet_size_category removed, blinds_level added)
                row = {
                    'scenario_key': scenario_key,
                    'hand_category': hand_category,
                    'example_hands': example_hands,
                    'position': position,
                    'stack_depth': stack_category,
                    'blinds_level': blinds_level,
                    'training_games': training_games,
                    'actions_pruned_count': actions_pruned_count,
                    'best_action': best_action.upper(),
                    'confidence': round(best_action_confidence, 3),
                    **{k: round(v, 3) for k, v in action_probs.items()}
                }
                
                export_data.append(row)
        
        # Create DataFrame and export
        if export_data:
            df = pd.DataFrame(export_data)
            
            # Sort by confidence descending, then by training games
            df = df.sort_values(['confidence', 'training_games'], ascending=[False, False])
            
            # Export to CSV
            df.to_csv(filename, index=False)
            
            print(f"✅ Exported {len(export_data)} scenarios to {filename}")
            print(f"📊 Columns: {list(df.columns)}")
            
            # Show summary stats with pruning information
            print(f"\n📈 EXPORT SUMMARY:")
            print(f"Total scenarios: {len(export_data)}")
            print(f"Avg training games per scenario: {df['training_games'].mean():.1f}")
            print(f"Avg confidence: {df['confidence'].mean():.3f}")
            if self.enable_pruning:
                print(f"✂️ Total actions pruned: {total_actions_pruned}")
                print(f"✂️ Avg actions pruned per scenario: {df['actions_pruned_count'].mean():.1f}")
            
            # Show action distribution
            print(f"\nBest Action Distribution:")
            action_dist = df['best_action'].value_counts()
            for action, count in action_dist.items():
                print(f"  {action}: {count} scenarios ({count/len(export_data)*100:.1f}%)")
            
            return df
        else:
            print("❌ No strategy data to export")
            return None
    
    def get_pruning_statistics(self):
        """
        Get comprehensive pruning effectiveness metrics.
        
        Returns:
            Dictionary with pruning statistics and effectiveness metrics
        """
        stats = self.pruning_statistics.copy()
        
        # Calculate additional metrics
        total_scenarios = len(self.regret_sum)
        if total_scenarios > 0:
            stats['scenarios_with_pruning'] = len([k for k, v in self.pruned_actions.items() if v])
            stats['avg_pruned_actions_per_scenario'] = stats['regret_pruned_count'] / total_scenarios if total_scenarios > 0 else 0
            stats['pruning_effectiveness'] = (stats['regret_pruned_count'] + stats['strategy_pruned_count']) / max(1, stats['total_pruning_events'])
            stats['restoration_rate'] = stats['actions_restored_count'] / max(1, stats['regret_pruned_count'])
        else:
            stats['scenarios_with_pruning'] = 0
            stats['avg_pruned_actions_per_scenario'] = 0
            stats['pruning_effectiveness'] = 0
            stats['restoration_rate'] = 0
        
        # Add restoration statistics per scenario
        stats['scenarios_with_restorations'] = len([k for k, v in self.restoration_chances.items() if v > 0])
        stats['total_restoration_attempts'] = sum(self.restoration_chances.values())
        
        return stats

    def start_performance_tracking(self):
        """Initialize performance tracking for the training session"""
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.performance_metrics = []
        print("📊 Performance tracking started")

    def calculate_regret_statistics(self):
        """Calculate current regret statistics efficiently"""
        if not self.regret_sum:
            return {'avg_regret': 0.0, 'max_regret': 0.0}
        
        all_regrets = []
        max_regret = 0.0
        
        for scenario_regrets in self.regret_sum.values():
            for regret_value in scenario_regrets.values():
                all_regrets.append(abs(regret_value))
                max_regret = max(max_regret, abs(regret_value))
        
        avg_regret = np.mean(all_regrets) if all_regrets else 0.0
        return {'avg_regret': avg_regret, 'max_regret': max_regret}

    def get_scenario_coverage_histogram(self):
        """Create a histogram of scenario visit counts"""
        if not self.scenario_counter:
            return {}
        
        visit_counts = list(self.scenario_counter.values())
        # Create bins for histogram: 0-10, 11-25, 26-50, 51-100, 100+
        bins = [0, 10, 25, 50, 100, float('inf')]
        bin_labels = ['0-10', '11-25', '26-50', '51-100', '100+']
        
        histogram = {}
        for i, label in enumerate(bin_labels):
            count = sum(1 for visits in visit_counts 
                       if bins[i] < visits <= bins[i+1])
            histogram[label] = count
        
        return histogram

    def calculate_scenario_space_coverage(self):
        """Calculate scenario space coverage statistics"""
        from enhanced_cfr_preflop_generator_v2 import PREFLOP_HAND_RANGES, STACK_CATEGORIES
        
        # Calculate theoretical maximum scenarios (bet_size_category removed)
        hand_categories = len(PREFLOP_HAND_RANGES)
        positions = 2  # BTN, BB
        stack_categories = len(STACK_CATEGORIES)
        blinds_levels = 3  # low, medium, high
        
        total_possible = hand_categories * positions * stack_categories * blinds_levels
        
        # Calculate coverage percentage
        unique_scenarios_visited = len(self.scenario_counter)
        coverage_percentage = (unique_scenarios_visited / total_possible) * 100 if total_possible > 0 else 0
        
        return {
            'total_possible': total_possible,
            'coverage_percentage': round(coverage_percentage, 2)
        }

    def record_iteration_metrics(self, iteration):
        """Record performance metrics for this iteration with enhanced coverage tracking"""
        current_time = time.time()
        
        # Calculate timing metrics
        total_elapsed = current_time - self.start_time
        time_per_iteration = current_time - self.last_iteration_time
        
        # Calculate regret statistics (efficiently)
        regret_stats = self.calculate_regret_statistics()
        
        # Get scenario coverage
        unique_scenarios = len(self.scenario_counter)
        coverage_histogram = self.get_scenario_coverage_histogram()
        
        # Calculate scenario space analysis
        scenario_space_stats = self.calculate_scenario_space_coverage()
        
        # Record metrics with enhanced coverage data
        metrics = {
            'iteration': iteration,
            'time_per_iteration': time_per_iteration,
            'total_elapsed_time': total_elapsed,
            'average_regret': regret_stats['avg_regret'],
            'max_regret': regret_stats['max_regret'],
            'unique_scenarios_visited': unique_scenarios,
            'scenario_coverage_0_10': coverage_histogram.get('0-10', 0),
            'scenario_coverage_11_25': coverage_histogram.get('11-25', 0),
            'scenario_coverage_26_50': coverage_histogram.get('26-50', 0),
            'scenario_coverage_51_100': coverage_histogram.get('51-100', 0),
            'scenario_coverage_100_plus': coverage_histogram.get('100+', 0),
            'total_possible_scenarios': scenario_space_stats['total_possible'],
            'scenario_coverage_percentage': scenario_space_stats['coverage_percentage'],
            **{f'hand_category_{cat}_visits': visits for cat, visits in self.hand_category_visits.items()}
        }
        
        self.performance_metrics.append(metrics)
        self.last_iteration_time = current_time
        
        return metrics

    def export_performance_metrics(self, filename="model_performance.csv"):
        """Export performance metrics to CSV file"""
        if not self.performance_metrics:
            print("❌ No performance metrics to export")
            return None
        
        import pandas as pd
        
        print(f"📊 Exporting performance metrics to {filename}...")
        
        df = pd.DataFrame(self.performance_metrics)
        df.to_csv(filename, index=False)
        
        # Show summary
        print(f"\n📈 PERFORMANCE METRICS SUMMARY:")
        print(f"Total iterations tracked: {len(df)}")
        print(f"Total training time: {df['total_elapsed_time'].iloc[-1]:.2f}s")
        print(f"Average time per iteration: {df['time_per_iteration'].mean():.4f}s")
        print(f"Final average regret: {df['average_regret'].iloc[-1]:.6f}")
        print(f"Final max regret: {df['max_regret'].iloc[-1]:.6f}")
        print(f"Unique scenarios visited: {df['unique_scenarios_visited'].iloc[-1]}")
        
        # Show convergence trend
        if len(df) >= 10:
            recent_avg_regret = df['average_regret'].tail(10).mean()
            early_avg_regret = df['average_regret'].head(10).mean()
            regret_change = (recent_avg_regret - early_avg_regret) / early_avg_regret * 100 if early_avg_regret > 0 else 0
            print(f"Regret convergence: {regret_change:+.1f}% change from early to recent iterations")
        
        return df

class SequentialScenarioTrainer(EnhancedCFRTrainer):
    """
    Sequential Scenario Trainer - processes scenarios sequentially with stopping conditions
    and multiple rollouts per visit
    
    Key differences from EnhancedCFRTrainer:
    1. Processes scenarios sequentially from a pre-generated list
    2. Runs fixed iterations (X) per scenario until stopping condition (Y) is met
    3. Stopping condition based on average regret stabilization
    4. Multiple random rollouts per scenario visit with averaged returns
    5. Enhanced logging with time estimates and rollout statistics
    """
    
    def __init__(self, scenarios=None, rollouts_per_visit=1, iterations_per_scenario=1000, 
                 stopping_condition_window=20, regret_stability_threshold=0.05,
                 min_rollouts_before_convergence=100,
                 enable_min_rollouts_stopping=True, enable_regret_stability_stopping=True,
                 enable_max_iterations_stopping=True, stopping_condition_mode='flexible',
                 enable_pruning=True, regret_pruning_threshold=-300.0, 
                 strategy_pruning_threshold=0.001):
        """
        Initialize Sequential Scenario Trainer with modular stopping conditions and pruning.
        
        Args:
            scenarios: Pre-generated scenarios (optional)
            rollouts_per_visit: Number of rollouts per scenario visit
            iterations_per_scenario: Maximum iterations per scenario
            stopping_condition_window: Window size for regret stability check
            regret_stability_threshold: Threshold for regret stability
            min_rollouts_before_convergence: Minimum rollouts before convergence check
            enable_min_rollouts_stopping: Enable minimum rollouts stopping condition
            enable_regret_stability_stopping: Enable regret stability stopping condition
            enable_max_iterations_stopping: Enable maximum iterations stopping condition
            stopping_condition_mode: 'strict' (all conditions must be met), 
                                     'flexible' (any condition can stop), 
                                     'custom' (user-defined logic)
            enable_pruning: Enable all pruning techniques
            regret_pruning_threshold: Threshold for regret-based pruning
            strategy_pruning_threshold: Threshold for strategy pruning
        """
        super().__init__(scenarios, enable_pruning, regret_pruning_threshold, strategy_pruning_threshold)
        
        # Sequential training parameters
        self.rollouts_per_visit = rollouts_per_visit
        self.iterations_per_scenario = iterations_per_scenario
        self.stopping_condition_window = stopping_condition_window
        self.regret_stability_threshold = regret_stability_threshold
        self.min_rollouts_before_convergence = min_rollouts_before_convergence
        
        # Modular stopping conditions
        self.enable_min_rollouts_stopping = enable_min_rollouts_stopping
        self.enable_regret_stability_stopping = enable_regret_stability_stopping
        self.enable_max_iterations_stopping = enable_max_iterations_stopping
        self.stopping_condition_mode = stopping_condition_mode
        
        # Stopping condition tracking
        self.stopping_reasons = defaultdict(list)  # Track why each scenario stopped
        self.stopping_statistics = {
            'min_rollouts_stops': 0,
            'regret_stability_stops': 0,
            'max_iterations_stops': 0,
            'custom_stops': 0
        }
        
        # Create scenario list for sequential processing
        self.scenario_list = list(self.scenarios)
        self.current_scenario_index = 0
        self.completed_scenarios = []
        
        # Enhanced tracking for sequential training
        self.scenario_regret_history = defaultdict(list)  # Track regret history per scenario
        self.scenario_completion_times = {}
        self.scenario_iteration_counts = defaultdict(int)
        self.scenario_rollout_counts = defaultdict(int)  # Track total rollouts per scenario
        self.rollout_distribution_stats = defaultdict(list)  # Track rollout outcomes
        
        print(f"🎯 Sequential Scenario Trainer Initialized!")
        print(f"📊 Total scenarios to process: {len(self.scenario_list)}")
        print(f"🎲 Rollouts per visit: {self.rollouts_per_visit}")
        print(f"🔄 Iterations per scenario: {self.iterations_per_scenario}")
        print(f"🛑 Stopping condition window: {self.stopping_condition_window} iterations")
        print(f"📈 Regret stability threshold: {self.regret_stability_threshold}")
        print(f"🎯 Min rollouts before convergence: {self.min_rollouts_before_convergence}")
        print(f"🔄 Stopping conditions: min_rollouts={self.enable_min_rollouts_stopping}, "
              f"regret_stability={self.enable_regret_stability_stopping}, "
              f"max_iterations={self.enable_max_iterations_stopping}")
        print(f"🎚️ Stopping mode: {self.stopping_condition_mode}")
        if self.enable_pruning:
            print(f"✂️ Pruning enabled: regret_threshold={self.regret_pruning_threshold}, "
                  f"strategy_threshold={self.strategy_pruning_threshold}")
    
    def check_stopping_condition(self, scenario_key, current_iteration=None):
        """
        Modular stopping condition checker supporting multiple stopping criteria.
        
        Args:
            scenario_key: Unique scenario identifier
            current_iteration: Current iteration number (optional)
            
        Returns:
            tuple: (should_stop: bool, stopping_reason: str)
        """
        regret_history = self.scenario_regret_history[scenario_key]
        total_rollouts = self.scenario_rollout_counts[scenario_key]
        current_iter = current_iteration or self.scenario_iteration_counts[scenario_key]
        
        stopping_reasons = []
        conditions_met = []
        
        # Check minimum rollouts condition
        if self.enable_min_rollouts_stopping:
            min_rollouts_met = total_rollouts >= self.min_rollouts_before_convergence
            conditions_met.append(('min_rollouts', min_rollouts_met))
            if min_rollouts_met:
                stopping_reasons.append("min_rollouts_met")
        
        # Check regret stability condition
        if self.enable_regret_stability_stopping:
            regret_stable = self._check_regret_stability(regret_history)
            conditions_met.append(('regret_stability', regret_stable))
            if regret_stable:
                stopping_reasons.append("regret_stabilized")
        
        # Check maximum iterations condition
        if self.enable_max_iterations_stopping:
            max_iter_reached = current_iter >= self.iterations_per_scenario
            conditions_met.append(('max_iterations', max_iter_reached))
            if max_iter_reached:
                stopping_reasons.append("max_iterations_reached")
        
        # Apply stopping condition mode logic
        should_stop = False
        final_reason = "unknown"
        
        if self.stopping_condition_mode == 'strict':
            # All enabled conditions must be met
            enabled_conditions = [cond for name, cond in conditions_met]
            should_stop = all(enabled_conditions) if enabled_conditions else False
            if should_stop:
                final_reason = "strict_all_conditions_met"
                self.stopping_statistics['min_rollouts_stops'] += (
                    1 if 'min_rollouts_met' in stopping_reasons else 0)
        
        elif self.stopping_condition_mode == 'flexible':
            # Any enabled condition can trigger stop (but min_rollouts is prerequisite)
            min_rollouts_ok = (not self.enable_min_rollouts_stopping or 
                             total_rollouts >= self.min_rollouts_before_convergence)
            
            other_conditions = [cond for name, cond in conditions_met if name != 'min_rollouts']
            should_stop = min_rollouts_ok and any(other_conditions) if other_conditions else False
            
            if should_stop:
                if 'regret_stabilized' in stopping_reasons:
                    final_reason = "regret_stabilized"
                    self.stopping_statistics['regret_stability_stops'] += 1
                elif 'max_iterations_reached' in stopping_reasons:
                    final_reason = "max_iterations_reached"
                    self.stopping_statistics['max_iterations_stops'] += 1
                else:
                    final_reason = "flexible_condition_met"
        
        elif self.stopping_condition_mode == 'custom':
            # User can override this method for custom logic
            should_stop, final_reason = self._custom_stopping_logic(scenario_key, conditions_met, current_iter)
            self.stopping_statistics['custom_stops'] += 1 if should_stop else 0
        
        # Record stopping reason for this scenario
        if should_stop:
            self.stopping_reasons[scenario_key].append(final_reason)
        
        return should_stop, final_reason
    
    def _check_regret_stability(self, regret_history):
        """Check if regret has stabilized over the window."""
        # Need at least window size + buffer for meaningful comparison
        if len(regret_history) < self.stopping_condition_window + 20:
            return False
        
        # Compare recent window average vs earlier window average
        recent_window = regret_history[-self.stopping_condition_window:]
        earlier_window = regret_history[-self.stopping_condition_window*2:-self.stopping_condition_window]
        
        if not earlier_window:  # Not enough history
            return False
        
        recent_avg = np.mean(recent_window)
        earlier_avg = np.mean(earlier_window)
        
        # Calculate relative change in average regret
        if earlier_avg == 0:
            relative_change = abs(recent_avg)
        else:
            relative_change = abs(recent_avg - earlier_avg) / abs(earlier_avg)
        
        return relative_change < self.regret_stability_threshold
    
    def _custom_stopping_logic(self, scenario_key, conditions_met, current_iteration):
        """
        Override this method to implement custom stopping logic.
        
        Args:
            scenario_key: Current scenario key
            conditions_met: List of (condition_name, met) tuples
            current_iteration: Current iteration number
            
        Returns:
            tuple: (should_stop: bool, reason: str)
        """
        # Default implementation - can be overridden by subclasses
        return False, "custom_not_implemented"
    
    def reconfigure_stopping_conditions(self, **kwargs):
        """
        Runtime reconfiguration of stopping conditions.
        
        Args:
            **kwargs: Stopping condition parameters to update
        """
        valid_params = {
            'enable_min_rollouts_stopping', 'enable_regret_stability_stopping',
            'enable_max_iterations_stopping', 'stopping_condition_mode',
            'min_rollouts_before_convergence', 'regret_stability_threshold',
            'iterations_per_scenario', 'stopping_condition_window'
        }
        
        for param, value in kwargs.items():
            if param in valid_params and hasattr(self, param):
                setattr(self, param, value)
                print(f"🔄 Updated stopping condition: {param} = {value}")
            else:
                print(f"⚠️ Invalid stopping condition parameter: {param}")
    
    def get_stopping_statistics(self):
        """Get comprehensive stopping condition statistics."""
        stats = self.stopping_statistics.copy()
        stats['total_scenarios_completed'] = len(self.completed_scenarios)
        
        # Calculate stopping reason distribution
        reason_counts = {}
        for scenario_reasons in self.stopping_reasons.values():
            for reason in scenario_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        stats['stopping_reason_distribution'] = reason_counts
        return stats
    
    def get_current_scenario_regret(self, scenario_key):
        """Calculate current average regret for a scenario"""
        if scenario_key not in self.regret_sum:
            return 0.0
        
        scenario_regrets = self.regret_sum[scenario_key]
        if not scenario_regrets:
            return 0.0
        
        # Calculate average absolute regret for this scenario
        total_regret = sum(abs(regret) for regret in scenario_regrets.values())
        avg_regret = total_regret / len(scenario_regrets)
        return avg_regret
    
    def play_scenario_multiple_rollouts(self, scenario):
        """
        Play scenario with multiple rollouts per visit and average the results
        Returns aggregated result from all rollouts for regret updates
        """
        scenario_key = self.get_scenario_key(scenario)
        rollout_results = []
        
        # Perform multiple rollouts with different random contexts
        for rollout_idx in range(self.rollouts_per_visit):
            # Each rollout uses different random seed contexts for:
            # - Dynamic betting context generation
            # - Villain hand generation  
            # - Action sampling
            result = self.play_enhanced_scenario(scenario)
            rollout_results.append(result)
            
            # Track individual rollout for statistics
            self.scenario_rollout_counts[scenario_key] += 1
            self.rollout_distribution_stats[scenario_key].append(result['payoff'])
        
        # Average the payoffs and other results across rollouts
        avg_payoff = np.mean([r['payoff'] for r in rollout_results])
        avg_hero_stack_after = np.mean([r['hero_stack_after'] for r in rollout_results])
        total_busted = sum([r['busted'] for r in rollout_results])
        
        # Use the last rollout's context info but averaged payoffs
        aggregated_result = rollout_results[-1].copy()  
        aggregated_result.update({
            'payoff': avg_payoff,
            'hero_stack_after': avg_hero_stack_after,
            'busted': total_busted > 0,  # True if any rollout resulted in bust
            'rollout_count': self.rollouts_per_visit,
            'rollout_payoffs': [r['payoff'] for r in rollout_results],
            'rollout_variance': np.var([r['payoff'] for r in rollout_results]) if len(rollout_results) > 1 else 0.0
        })
        
        return aggregated_result
    
    def process_single_scenario(self, scenario, max_iterations=None):
        """
        Process a single scenario with multiple rollouts per visit until stopping condition
        Returns dict with processing results including rollout statistics
        """
        scenario_key = self.get_scenario_key(scenario)
        start_time = time.time()
        iteration_count = 0
        max_iter = max_iterations or self.iterations_per_scenario * 10  # Safety limit
        
        print(f"\n🎯 Processing scenario: {scenario_key}")
        print(f"   Hand: {scenario['hand_category']}, Position: {scenario['hero_position']}, "
              f"Stack: {scenario['stack_category']}, Blinds: {scenario['blinds_level']}")
        print(f"   Rollouts per visit: {self.rollouts_per_visit}")
        
        while iteration_count < max_iter:
            # Run multiple rollouts for this scenario iteration
            if self.rollouts_per_visit == 1:
                # Use single rollout for backward compatibility and performance
                result = self.play_enhanced_scenario(scenario)
                self.scenario_rollout_counts[scenario_key] += 1
            else:
                # Use multiple rollouts per visit
                result = self.play_scenario_multiple_rollouts(scenario)
            
            self.scenario_counter[scenario_key] += 1
            iteration_count += 1
            
            # Record regret for stopping condition check
            current_regret = self.get_current_scenario_regret(scenario_key)
            self.scenario_regret_history[scenario_key].append(current_regret)
            
            # Check stopping condition every batch of iterations
            if iteration_count >= self.stopping_condition_window and iteration_count % 50 == 0:
                should_stop, stop_reason = self.check_stopping_condition(scenario_key, iteration_count)
                if should_stop:
                    total_rollouts = self.scenario_rollout_counts[scenario_key]
                    print(f"✅ Stopping condition met after {iteration_count} iterations "
                          f"({total_rollouts} total rollouts, reason: {stop_reason}, regret: {current_regret:.6f})")
                    break
            
            # Progress logging every certain iterations
            if iteration_count % max(self.iterations_per_scenario // 4, 100) == 0:
                total_rollouts = self.scenario_rollout_counts[scenario_key]
                rollout_variance = np.var(self.rollout_distribution_stats[scenario_key]) if len(self.rollout_distribution_stats[scenario_key]) > 1 else 0.0
                print(f"   Iteration {iteration_count:4d}: regret={current_regret:.6f}, "
                      f"rollouts={total_rollouts}, payoff_var={rollout_variance:.4f}")
        
        # Record completion
        end_time = time.time()
        processing_time = end_time - start_time
        self.scenario_completion_times[scenario_key] = processing_time
        self.scenario_iteration_counts[scenario_key] = iteration_count
        
        # Determine why processing stopped
        total_rollouts = self.scenario_rollout_counts[scenario_key]
        if iteration_count >= max_iter:
            stop_reason = "max_iterations_reached"
        elif total_rollouts < self.min_rollouts_before_convergence:
            stop_reason = f"min_rollouts_not_met ({total_rollouts}/{self.min_rollouts_before_convergence})"
        else:
            should_stop, reason = self.check_stopping_condition(scenario_key, iteration_count)
            stop_reason = reason if should_stop else "completed_normally"
        
        final_regret = self.get_current_scenario_regret(scenario_key)
        
        # Calculate rollout statistics
        rollout_payoffs = self.rollout_distribution_stats[scenario_key]
        rollout_stats = {
            'total_rollouts': total_rollouts,
            'avg_payoff': np.mean(rollout_payoffs) if rollout_payoffs else 0.0,
            'payoff_variance': np.var(rollout_payoffs) if len(rollout_payoffs) > 1 else 0.0,
            'payoff_min': min(rollout_payoffs) if rollout_payoffs else 0.0,
            'payoff_max': max(rollout_payoffs) if rollout_payoffs else 0.0
        }
        
        result_summary = {
            'scenario_key': scenario_key,
            'iterations_completed': iteration_count,
            'processing_time_seconds': processing_time,
            'final_regret': final_regret,
            'stop_reason': stop_reason,
            'regret_history_length': len(self.scenario_regret_history[scenario_key]),
            **rollout_stats
        }
        
        print(f"✅ Completed scenario {scenario_key}: {iteration_count} iterations, {total_rollouts} rollouts, "
              f"{processing_time:.2f}s, final_regret={final_regret:.6f}")
        
        return result_summary
    
    def calculate_remaining_time_estimate(self):
        """Calculate estimated remaining training time"""
        completed_scenarios = len(self.completed_scenarios)
        
        if completed_scenarios == 0:
            return {"estimated_remaining_seconds": None, "estimated_total_seconds": None}
        
        # Average time per completed scenario
        total_completed_time = sum(self.scenario_completion_times.values())
        avg_time_per_scenario = total_completed_time / completed_scenarios
        
        # Estimate remaining time
        remaining_scenarios = len(self.scenario_list) - self.current_scenario_index
        estimated_remaining_seconds = remaining_scenarios * avg_time_per_scenario
        estimated_total_seconds = len(self.scenario_list) * avg_time_per_scenario
        
        return {
            "avg_time_per_scenario": avg_time_per_scenario,
            "remaining_scenarios": remaining_scenarios,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "estimated_total_seconds": estimated_total_seconds,
            "completed_scenarios": completed_scenarios
        }
    
    def log_progress_with_estimates(self):
        """Enhanced logging with time estimates and rollout statistics"""
        time_estimates = self.calculate_remaining_time_estimate()
        completed = len(self.completed_scenarios)
        total = len(self.scenario_list)
        progress_percent = (completed / total) * 100 if total > 0 else 0
        
        print(f"\n📊 SEQUENTIAL TRAINING PROGRESS:")
        print(f"   Completed scenarios: {completed}/{total} ({progress_percent:.1f}%)")
        
        if time_estimates["estimated_remaining_seconds"]:
            remaining_hours = time_estimates["estimated_remaining_seconds"] / 3600
            total_hours = time_estimates["estimated_total_seconds"] / 3600
            avg_minutes = time_estimates["avg_time_per_scenario"] / 60
            
            print(f"   Average time per scenario: {avg_minutes:.2f} minutes")
            print(f"   Estimated remaining time: {remaining_hours:.2f} hours")
            print(f"   Estimated total training time: {total_hours:.2f} hours")
        
        # Show iteration and rollout distribution
        if self.scenario_iteration_counts:
            iterations_list = list(self.scenario_iteration_counts.values())
            avg_iterations = np.mean(iterations_list)
            total_iterations = sum(iterations_list)
            print(f"   Average iterations per scenario: {avg_iterations:.1f}")
            print(f"   Total iterations completed: {total_iterations:,}")
            
        # Show rollout statistics
        if self.scenario_rollout_counts:
            rollouts_list = list(self.scenario_rollout_counts.values())
            total_rollouts = sum(rollouts_list)
            avg_rollouts = np.mean(rollouts_list)
            print(f"   Total rollouts completed: {total_rollouts:,}")
            print(f"   Average rollouts per scenario: {avg_rollouts:.1f}")
            
            # Show rollout payoff variance statistics
            if self.rollout_distribution_stats:
                all_payoffs = []
                scenario_variances = []
                for scenario_payoffs in self.rollout_distribution_stats.values():
                    if len(scenario_payoffs) > 1:
                        scenario_variances.append(np.var(scenario_payoffs))
                        all_payoffs.extend(scenario_payoffs)
                
                if scenario_variances:
                    avg_variance = np.mean(scenario_variances)
                    print(f"   Average payoff variance per scenario: {avg_variance:.4f}")
    
    def run_sequential_training(self):
        """
        Run the complete sequential training process
        Process all scenarios in order until each meets its stopping condition
        """
        print(f"🚀 Starting Sequential Scenario Training")
        print(f"📊 Total scenarios to process: {len(self.scenario_list)}")
        print("=" * 70)
        
        # Start performance tracking
        self.start_performance_tracking()
        training_start_time = time.time()
        
        for idx, scenario in enumerate(self.scenario_list):
            self.current_scenario_index = idx
            
            # Process this scenario until stopping condition
            result = self.process_single_scenario(scenario)
            self.completed_scenarios.append(result)
            
            # Log progress with estimates
            if (idx + 1) % max(len(self.scenario_list) // 20, 1) == 0:  # Log every 5% or more frequently
                self.log_progress_with_estimates()
        
        # Final summary
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print(f"\n🎉 SEQUENTIAL TRAINING COMPLETE!")
        print(f"⏱️  Total training time: {total_training_time/3600:.2f} hours")
        print(f"📊 Scenarios processed: {len(self.completed_scenarios)}")
        
        # Training completion statistics
        total_iterations = sum(self.scenario_iteration_counts.values())
        total_rollouts = sum(self.scenario_rollout_counts.values()) if self.scenario_rollout_counts else total_iterations
        avg_iterations = np.mean(list(self.scenario_iteration_counts.values())) if self.scenario_iteration_counts else 0
        avg_rollouts_per_scenario = np.mean(list(self.scenario_rollout_counts.values())) if self.scenario_rollout_counts else 0
        
        print(f"🔢 Total iterations: {total_iterations:,}")
        print(f"🎲 Total rollouts: {total_rollouts:,}")
        print(f"🎯 Average iterations per scenario: {avg_iterations:.1f}")
        print(f"🎰 Average rollouts per scenario: {avg_rollouts_per_scenario:.1f}")
        print(f"📊 Rollouts per visit configuration: {self.rollouts_per_visit}")
        
        # Rollout variance analysis
        if self.rollout_distribution_stats and self.rollouts_per_visit > 1:
            print(f"\n🎲 ROLLOUT VARIANCE ANALYSIS:")
            scenario_variances = []
            for scenario_key, payoffs in self.rollout_distribution_stats.items():
                if len(payoffs) > 1:
                    variance = np.var(payoffs)
                    scenario_variances.append(variance)
            
            if scenario_variances:
                print(f"   Average payoff variance: {np.mean(scenario_variances):.4f}")
                print(f"   Min variance: {min(scenario_variances):.4f}")
                print(f"   Max variance: {max(scenario_variances):.4f}")
        
        # Stop reasons distribution
        stop_reasons = [result['stop_reason'] for result in self.completed_scenarios]
        stop_reason_counts = Counter(stop_reasons)
        print(f"\n📈 Stopping condition analysis:")
        for reason, count in stop_reason_counts.items():
            percentage = (count / len(stop_reasons)) * 100
            print(f"   {reason}: {count} scenarios ({percentage:.1f}%)")
        
        # Export results
        self.export_strategies_to_csv("sequential_cfr_strategies.csv")
        self.export_performance_metrics("sequential_performance_metrics.csv")
        self.export_scenario_completion_report("scenario_completion_report.csv")
        
        return self.completed_scenarios
    
    def export_scenario_completion_report(self, filename="scenario_completion_report.csv"):
        """Export detailed report of scenario completion"""
        if not self.completed_scenarios:
            print("❌ No scenario completion data to export")
            return None
        
        import pandas as pd
        
        print(f"📊 Exporting scenario completion report to {filename}...")
        
        # Prepare detailed completion data
        report_data = []
        for result in self.completed_scenarios:
            scenario_key = result['scenario_key']
            parts = scenario_key.split("|")
            
            row = {
                'scenario_key': scenario_key,
                'hand_category': parts[0] if len(parts) > 0 else 'unknown',
                'position': parts[1] if len(parts) > 1 else 'unknown',
                'stack_category': parts[2] if len(parts) > 2 else 'unknown',
                'blinds_level': parts[3] if len(parts) > 3 else 'unknown',
                'iterations_completed': result['iterations_completed'],
                'processing_time_seconds': result['processing_time_seconds'],
                'processing_time_minutes': result['processing_time_seconds'] / 60,
                'final_regret': result['final_regret'],
                'stop_reason': result['stop_reason'],
                'regret_history_length': result['regret_history_length'],
                # Add rollout statistics
                'total_rollouts': result.get('total_rollouts', result['iterations_completed']),
                'avg_payoff': result.get('avg_payoff', 0.0),
                'payoff_variance': result.get('payoff_variance', 0.0),
                'payoff_min': result.get('payoff_min', 0.0),
                'payoff_max': result.get('payoff_max', 0.0)
            }
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Sort by processing time descending to see which took longest
        df = df.sort_values('processing_time_seconds', ascending=False)
        
        # Export to CSV
        df.to_csv(filename, index=False)
        
        print(f"✅ Exported completion report for {len(report_data)} scenarios")
        
        # Show summary statistics
        print(f"\n📈 SCENARIO COMPLETION SUMMARY:")
        print(f"Avg processing time: {df['processing_time_minutes'].mean():.2f} minutes")
        print(f"Max processing time: {df['processing_time_minutes'].max():.2f} minutes")
        print(f"Min processing time: {df['processing_time_minutes'].min():.2f} minutes")
        print(f"Avg iterations per scenario: {df['iterations_completed'].mean():.1f}")
        print(f"Total training iterations: {df['iterations_completed'].sum():,}")
        print(f"Avg rollouts per scenario: {df['total_rollouts'].mean():.1f}")
        print(f"Total training rollouts: {df['total_rollouts'].sum():,}")
        
        if self.rollouts_per_visit > 1:
            print(f"Avg payoff variance: {df['payoff_variance'].mean():.4f}")
        
        return df


if __name__ == "__main__":
    print("Enhanced CFR Trainer v2 - Ready for training!")
    
    # Add a simple training function for testing
    def run_enhanced_training(n_iterations=200000, metrics_interval=1000):
        """Run enhanced CFR training with balanced sampling and performance tracking"""
        print(f"🚀 Running Enhanced CFR Training with Balanced Hand Category Coverage")
        print(f"Iterations: {n_iterations}")
        print(f"Metrics interval: every {metrics_interval} iterations")
        print(f"Action set: FOLD, CALL_SMALL, CALL_MID, CALL_HIGH, RAISE_SMALL, RAISE_MID, RAISE_HIGH")
        print(f"🎯 Using STRATIFIED SAMPLING for balanced hand category coverage")
        print("=" * 70)
        
        # Generate all possible scenarios (no manual n_scenarios parameter)
        from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
        scenarios = generate_enhanced_scenarios()
        
        # Initialize trainer
        trainer = EnhancedCFRTrainer(scenarios=scenarios)
        
        # Start performance tracking
        trainer.start_performance_tracking()
        
        # Train with balanced sampling and performance tracking
        print(f"\n🎯 Starting CFR training with balanced sampling...")
        for iteration in range(n_iterations):
            # Use balanced scenario selection instead of random
            scenario = trainer.select_balanced_scenario()
            trainer.play_enhanced_scenario(scenario)
            trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1
            
            # Record metrics at regular intervals
            if iteration % metrics_interval == 0:
                metrics = trainer.record_iteration_metrics(iteration)
                if iteration > 0:  # Skip first iteration for meaningful timing
                    print(f"Iteration {iteration:6d}: {metrics['unique_scenarios_visited']:3d} scenarios, "
                          f"coverage={metrics['scenario_coverage_percentage']:5.1f}%, "
                          f"avg_regret={metrics['average_regret']:.6f}")
        
        # Record final metrics
        final_metrics = trainer.record_iteration_metrics(n_iterations - 1)
        
        print(f"✅ Training complete after {n_iterations:,} iterations")
        print(f"📊 Learned strategies for {len(trainer.strategy_sum)} scenarios")
        print(f"🎯 Hand category coverage balance:")
        for category, visits in trainer.hand_category_visits.items():
            percentage = (visits / n_iterations) * 100
            print(f"   {category:15s}: {visits:6d} visits ({percentage:5.1f}%)")
        
        # Export both strategy results and performance metrics
        trainer.export_strategies_to_csv("enhanced_cfr_results.csv")
        trainer.export_performance_metrics("model_performance.csv")
        
        return trainer
    
    def run_sequential_training_demo(iterations_per_scenario=500, stopping_window=50, 
                                   regret_threshold=0.01, max_scenarios=5, rollouts_per_visit=1,
                                   min_rollouts_before_convergence=100):
        """Run sequential training demo with configurable parameters"""
        print(f"🚀 Sequential Training Demo")
        print(f"Iterations per scenario: {iterations_per_scenario}")
        print(f"Rollouts per visit: {rollouts_per_visit}")
        print(f"Stopping window: {stopping_window}")
        print(f"Regret threshold: {regret_threshold}")
        print(f"Min rollouts before convergence: {min_rollouts_before_convergence}")
        print(f"Max scenarios: {max_scenarios}")
        
        # Generate scenarios
        from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
        all_scenarios = generate_enhanced_scenarios()
        
        # Use only first few scenarios for demo
        demo_scenarios = all_scenarios[:max_scenarios]
        
        # Initialize sequential trainer
        trainer = SequentialScenarioTrainer(
            scenarios=demo_scenarios,
            rollouts_per_visit=rollouts_per_visit,
            iterations_per_scenario=iterations_per_scenario,
            stopping_condition_window=stopping_window,
            regret_stability_threshold=regret_threshold,
            min_rollouts_before_convergence=min_rollouts_before_convergence
        )
        
        # Run training
        results = trainer.run_sequential_training()
        
        return trainer, results
    
    # Uncomment to run training:
    # trainer = run_enhanced_training()
    # trainer, results = run_sequential_training_demo()
