# enhanced_cfr_trainer_v2.py - Tournament-aware CFR with stack survival

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
    Enhanced CFR with tournament survival, stack awareness, and bet sizing
    """
    
    def __init__(self, scenarios=None, n_scenarios=1000):
        # CFR data structures - now with variable action counts
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.scenario_counter = Counter()
        self.iterations = 0

        # Enhanced tracking
        self.stack_survival_rate = []
        self.tournament_results = []
        self.equity_by_stack = defaultdict(list)

        # Generate enhanced scenarios, or use provided ones
        if scenarios is not None:
            self.scenarios = scenarios
            print(f"🚀 Using provided {len(scenarios)} scenarios...")
        else:
            self.scenarios = generate_enhanced_scenarios(n_scenarios)
            print(f"🚀 Generated {n_scenarios} scenarios...")
            
        print(f"🏆 Enhanced CFR Trainer Initialized!")
        print(f"📊 Training scenarios: {len(self.scenarios):,}")

    def get_strategy(self, scenario_key, available_actions):
        """Get strategy using regret matching for available actions only"""
        regrets = self.regret_sum[scenario_key]
        
        # Only consider regrets for available actions
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

    def sample_action(self, strategy, available_actions):
        """Sample action from strategy probabilities"""
        probs = [strategy[action] for action in available_actions]
        chosen_idx = np.random.choice(len(available_actions), p=probs)
        return available_actions[chosen_idx]

    def play_enhanced_scenario(self, scenario):
        """Play enhanced scenario with stack and tournament considerations"""
        scenario_key = self.get_scenario_key(scenario)
        available_actions = scenario["available_actions"]
        
        # Hero's decision using CFR strategy
        hero_strategy = self.get_strategy(scenario_key, available_actions)
        hero_action = self.sample_action(hero_strategy, available_actions)
        
        # Generate villain and their action
        villain_cards = self.generate_villain_hand(scenario['hero_cards_int'])
        villain_action = self.get_enhanced_villain_action(villain_cards, scenario)
        
        # Calculate enhanced payoff with tournament considerations
        payoff_result = self.calculate_enhanced_payoff(scenario, hero_action, villain_action, villain_cards)
        
        # Update regrets for all available actions
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
            'available_actions': available_actions
        }

    def calculate_enhanced_payoff(self, scenario, hero_action, villain_action, villain_cards):
        """Calculate payoff with tournament survival considerations"""
        hero_stack_before = scenario['hero_stack_bb']
        bet_amount = scenario['bet_to_call_bb']
        tournament_stage = scenario['tournament_stage']
        
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
        
        # Tournament survival adjustments
        tournament_payoff = self.apply_tournament_adjustments(
            base_payoff, hero_stack_before, hero_stack_after, 
            tournament_stage, busted
        )
        
        return {
            'payoff': tournament_payoff,
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

    def apply_tournament_adjustments(self, base_payoff, stack_before, stack_after, 
                                   tournament_stage, busted):
        """Apply tournament-specific payoff adjustments"""
        tournament_payoff = base_payoff
        
        # Massive penalty for busting
        if busted:
            if tournament_stage == "bubble":
                tournament_payoff = -10.0  # Extremely bad to bust on bubble
            elif tournament_stage == "late":
                tournament_payoff = -5.0   # Very bad to bust late
            else:
                tournament_payoff = -3.0   # Bad to bust anytime
        
        # Survival bonus for short stacks
        if stack_before <= 15 and stack_after > stack_before:
            survival_bonus = 2.0 * (stack_after - stack_before) / stack_before
            tournament_payoff += survival_bonus
        
        # Stack preservation bonus in late stages
        if tournament_stage in ["late", "bubble"] and not busted:
            if stack_after >= stack_before * 0.8:  # Didn't lose much
                preservation_bonus = 0.5
                tournament_payoff += preservation_bonus
        
        # Chip accumulation bonus in early stages
        if tournament_stage == "early" and stack_after > stack_before * 1.5:
            accumulation_bonus = 1.0
            tournament_payoff += accumulation_bonus
        
        return tournament_payoff

    def generate_villain_hand(self, hero_cards):
        """Generate random villain hand"""
        deck = Deck()
        for card in hero_cards:
            if card in deck.cards:
                deck.cards.remove(card)
        return deck.draw(2)

    def get_enhanced_villain_action(self, villain_cards, scenario):
        """Enhanced villain action based on stack and tournament context"""
        try:
            villain_equity = self.estimate_villain_equity(villain_cards)
            villain_stack = scenario['villain_stack_bb']
            tournament_stage = scenario['tournament_stage']
            
            # Adjust strategy based on tournament stage and stack
            if tournament_stage == "bubble" and villain_stack <= 20:
                # Very tight on bubble with short stack
                if villain_equity > 0.7:
                    return "all_in"
                elif villain_equity > 0.5:
                    return "call_small"
                else:
                    return "fold"
            
            elif villain_stack <= 15:  # Short stack - push/fold
                if villain_equity > 0.45:
                    return "all_in"
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
        """Enhanced regret update considering all available actions"""
        actual_payoff = payoff_result['payoff']
        
        # Estimate counterfactual payoffs for other actions
        for action in available_actions:
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
        
        # Simple heuristic estimates
        if alternative_action == "fold":
            return -0.1  # Small loss from folding
        elif alternative_action == "all_in":
            if actual_result['busted']:
                return -2.0  # Would have busted anyway
            else:
                return actual_payoff * 1.5  # Higher risk, higher reward
        elif "call" in alternative_action:
            return actual_payoff * 0.8  # More conservative
        elif "raise" in alternative_action:
            return actual_payoff * 1.2  # More aggressive
        else:
            return actual_payoff

    def get_scenario_key(self, scenario):
        """Enhanced scenario key with stack and tournament context"""
        return (f"{scenario['hand_category']}_{scenario['hero_position']}_"
                f"{scenario['stack_category']}_{scenario['bet_size_category']}_"
                f"{scenario['tournament_stage']}")

if __name__ == "__main__":
    print("Enhanced CFR Trainer v2 - Ready for training!")
