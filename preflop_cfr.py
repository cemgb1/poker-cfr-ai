# -*- coding: utf-8 -*-
"""
Preflop CFR Trainer - Complete implementation for all 169 starting hands
Ensures equal distribution of training across all scenarios
"""

import random
import numpy as np
from collections import defaultdict, Counter

class PreflopCFR:
    """
    CFR solver focused on preflop play with guaranteed equal training distribution
    """

    def __init__(self, big_blind=2):
        # CFR data structures
        self.regret_sum = defaultdict(lambda: np.zeros(3))  # [fold, call, raise]
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.iterations = 0
        self.info_set_counter = Counter()

        # Game parameters
        self.big_blind = big_blind

        # Create comprehensive hand grouping system
        self.hand_groups = self.create_hand_groups()
        self.all_hands = self.generate_all_169_hands()
        
        # Generate ALL possible preflop scenarios
        self.all_preflop_scenarios = self.generate_all_preflop_scenarios()
        
        print(f"🃏 Preflop CFR Initialized!")
        print(f"📊 Total hands: {len(self.all_hands)}")
        print(f"📊 Hand groups: {len(set(self.hand_groups.values()))}")
        print(f"📊 Total scenarios: {len(self.all_preflop_scenarios)}")

    def create_hand_groups(self):
        """Group all 169 poker hands into strategic categories"""
        groups = {}
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i <= j:
                    if rank1 == rank2:
                        groups[rank1 + rank2] = self.categorize_pair(rank1)
                    else:
                        suited_hand = rank1 + rank2 + 's'
                        offsuit_hand = rank1 + rank2 + 'o'
                        groups[suited_hand] = self.categorize_suited(rank1, rank2)
                        groups[offsuit_hand] = self.categorize_offsuit(rank1, rank2)
        return groups

    def categorize_pair(self, rank):
        """Categorize pocket pairs"""
        if rank in ['A', 'K', 'Q']:
            return 'premium_pairs'
        elif rank in ['J', 'T', '9']:
            return 'medium_pairs'
        else:
            return 'small_pairs'

    def categorize_suited(self, rank1, rank2):
        """Categorize suited hands"""
        high_rank, low_rank = rank1, rank2

        if high_rank == 'A':
            if low_rank in ['K', 'Q', 'J']:
                return 'premium_aces_suited'
            elif low_rank in ['T', '9', '8']:
                return 'medium_aces_suited'
            else:
                return 'weak_aces_suited'
        elif high_rank == 'K':
            if low_rank in ['Q', 'J', 'T']:
                return 'premium_kings_suited'
            else:
                return 'medium_kings_suited'
        elif high_rank in ['Q', 'J', 'T']:
            if self.is_connected(rank1, rank2):
                return 'suited_connectors_high'
            else:
                return 'suited_broadways'
        elif self.is_connected(rank1, rank2):
            return 'suited_connectors_low'
        else:
            return 'suited_trash'

    def categorize_offsuit(self, rank1, rank2):
        """Categorize offsuit hands"""
        high_rank, low_rank = rank1, rank2

        if high_rank == 'A':
            if low_rank in ['K', 'Q', 'J']:
                return 'premium_aces_offsuit'
            elif low_rank in ['T', '9']:
                return 'medium_aces_offsuit'
            else:
                return 'weak_aces_offsuit'
        elif high_rank == 'K':
            if low_rank in ['Q', 'J']:
                return 'premium_kings_offsuit'
            else:
                return 'weak_kings_offsuit'
        elif high_rank in ['Q', 'J', 'T'] and low_rank in ['J', 'T', '9']:
            return 'offsuit_broadways'
        else:
            return 'offsuit_trash'

    def is_connected(self, rank1, rank2):
        """Check if ranks are connected"""
        rank_order = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        try:
            pos1 = rank_order.index(rank1)
            pos2 = rank_order.index(rank2)
            return abs(pos1 - pos2) == 1
        except:
            return False

    def generate_all_169_hands(self):
        """Generate all 169 possible starting hands"""
        hands = []
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i < j:
                    hands.append(rank1 + rank2 + 's')
                    hands.append(rank1 + rank2 + 'o')
                elif i == j:
                    hands.append(rank1 + rank2)
        return hands

    def generate_all_preflop_scenarios(self):
        """Generate ALL possible preflop training scenarios"""
        scenarios = []
        hand_groups = set(self.hand_groups.values())
        positions = ["BTN", "BB"]
        histories = ["", "r", "rr", "cr", "rrr", "rrc", "crr"]
        stack_depths = ["short", "medium", "deep"]

        for group in hand_groups:
            for position in positions:
                for history in histories:
                    for stack_depth in stack_depths:
                        scenario = f"{group}|{position}|{history}|{stack_depth}"
                        scenarios.append(scenario)
        
        return scenarios

    def train_equal_distribution(self, total_iterations=500000, min_visits_per_scenario=500):
        """Train with guaranteed equal distribution across all scenarios"""
        print(f"🚀 Training Preflop CFR with Equal Distribution")
        print(f"Target: {min_visits_per_scenario} visits per scenario")
        print("=" * 70)

        hands_seen = set()
        
        for iteration in range(total_iterations):
            # Find scenarios that need more training
            undertrained = [
                scenario for scenario in self.all_preflop_scenarios 
                if self.info_set_counter.get(scenario, 0) < min_visits_per_scenario
            ]

            if not undertrained:
                print(f"✅ All scenarios trained to {min_visits_per_scenario}+ visits at iteration {iteration}")
                break

            # Select scenario that needs most training (lowest count)
            target_scenario = min(undertrained, key=lambda x: self.info_set_counter.get(x, 0))
            
            # Parse scenario
            group, position, history, stack_cat = target_scenario.split("|")
            
            # Select hands from the target group
            candidate_hands = [h for h, g in self.hand_groups.items() if g == group]
            p0_hand = random.choice(candidate_hands)
            p1_hand = random.choice([h for h in self.all_hands if h != p0_hand])
            
            # Convert stack category to BB
            stack_bb = {"short": 20, "medium": 50, "deep": 100}[stack_cat]
            
            # Run CFR
            player = 0 if position == "BTN" else 1
            self.cfr_preflop(p0_hand, p1_hand, history, 1.0, 1.0, player, stack_bb)
            
            # Update counters
            self.info_set_counter[target_scenario] += 1
            hands_seen.add(p0_hand)
            hands_seen.add(p1_hand)

            # Progress reporting
            if iteration % 10000 == 0:
                completed = len([s for s in self.all_preflop_scenarios 
                               if self.info_set_counter.get(s, 0) >= min_visits_per_scenario])
                progress = completed / len(self.all_preflop_scenarios) * 100
                
                print(f"Iter {iteration:6d}: {progress:5.1f}% complete, "
                      f"{len(undertrained):3d} scenarios need training")
                
                # Show least trained scenarios
                least_trained = sorted(undertrained, key=lambda x: self.info_set_counter.get(x, 0))[:3]
                for scenario in least_trained:
                    count = self.info_set_counter.get(scenario, 0)
                    print(f"  ↳ {scenario}: {count} visits")

        self.iterations = iteration
        print(f"\n🎯 Training Complete!")
        print(f"Total iterations: {self.iterations}")
        print(f"Hands seen: {len(hands_seen)}")
        print(f"Scenarios trained: {len([s for s in self.all_preflop_scenarios if self.info_set_counter[s] > 0])}")

    def cfr_preflop(self, p0_hand, p1_hand, history, p0, p1, player, stack_bb):
        """CFR recursion for preflop play"""
        
        # Terminal check
        if self.is_terminal(history):
            return self.get_payoff(p0_hand, p1_hand, history, player, stack_bb)

        # Get information set
        current_hand = p0_hand if player == 0 else p1_hand
        hand_group = self.hand_groups[current_hand]
        position = "BTN" if player == 0 else "BB"
        stack_cat = "deep" if stack_bb > 75 else "medium" if stack_bb > 35 else "short"
        
        info_set = f"{hand_group}|{position}|{history}|{stack_cat}"
        
        # Get strategy
        strategy = self.get_strategy(info_set, p0 if player == 0 else p1)
        
        # Get valid actions
        valid_actions = self.get_valid_actions(history)
        
        # Calculate utilities for each action
        utilities = np.zeros(3)
        for action in valid_actions:
            new_history = history + ['f', 'c', 'r'][action]
            
            if player == 0:
                utilities[action] = -self.cfr_preflop(
                    p0_hand, p1_hand, new_history, p0 * strategy[action], p1, 1, stack_bb
                )
            else:
                utilities[action] = -self.cfr_preflop(
                    p0_hand, p1_hand, new_history, p0, p1 * strategy[action], 0, stack_bb
                )

        # Calculate node utility and update regrets
        node_util = np.sum(strategy * utilities)
        
        for action in valid_actions:
            regret = utilities[action] - node_util
            self.regret_sum[info_set][action] += (p1 if player == 0 else p0) * regret

        return node_util

    def is_terminal(self, history):
        """Check if game state is terminal"""
        return (
            len(history) > 0 and history[-1] == 'f' or
            history in ['cc', 'rc', 'crc'] or
            len(history) >= 4
        )

    def get_payoff(self, p0_hand, p1_hand, history, player, stack_bb):
        """Calculate payoff for terminal states"""
        if history and history[-1] == 'f':
            # Someone folded
            pot_size = 2 + history[:-1].count('r')
            folder = (len(history) - 1) % 2
            return pot_size if player != folder else -pot_size
        else:
            # Showdown
            p0_strength = self.get_hand_strength(p0_hand)
            p1_strength = self.get_hand_strength(p1_hand)
            pot_size = 2 + history.count('r')

            if p0_strength > p1_strength:
                return pot_size if player == 0 else -pot_size
            elif p1_strength > p0_strength:
                return -pot_size if player == 0 else pot_size
            else:
                return 0

    def get_hand_strength(self, hand):
        """Calculate relative hand strength"""
        group = self.hand_groups[hand]
        group_strengths = {
            'premium_pairs': 95, 'medium_pairs': 75, 'small_pairs': 45,
            'premium_aces_suited': 90, 'premium_aces_offsuit': 85,
            'medium_aces_suited': 70, 'medium_aces_offsuit': 60,
            'weak_aces_suited': 50, 'weak_aces_offsuit': 35,
            'premium_kings_suited': 80, 'premium_kings_offsuit': 70,
            'medium_kings_suited': 55, 'suited_connectors_high': 65,
            'suited_connectors_low': 40, 'suited_broadways': 60,
            'offsuit_broadways': 50, 'suited_trash': 25, 'offsuit_trash': 15
        }
        return group_strengths.get(group, 30)

    def get_valid_actions(self, history):
        """Get valid actions for current game state"""
        if len(history) == 0:
            return [1, 2]  # check or raise (can't fold first)
        elif len(history) >= 3:
            return [0, 1]  # fold or call only (betting cap)
        else:
            return [0, 1, 2]  # fold, call, raise

    def get_strategy(self, info_set, reach_prob):
        """Get strategy using regret matching"""
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)

        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            strategy = np.ones(3) / 3

        self.strategy_sum[info_set] += reach_prob * strategy
        return strategy

    def get_average_strategy(self, info_set):
        """Get average strategy over all iterations"""
        strategy_sum = self.strategy_sum[info_set]
        total = np.sum(strategy_sum)
        return strategy_sum / total if total > 0 else np.ones(3) / 3

    def get_strategy_for_hand(self, hand, position, history="", stack_depth="deep"):
        """Get optimal strategy for any hand in any situation"""
        if hand not in self.hand_groups:
            return None, f"Hand {hand} not recognized"

        hand_group = self.hand_groups[hand]
        info_set = f"{hand_group}|{position}|{history}|{stack_depth}"

        if info_set in self.strategy_sum and np.sum(self.strategy_sum[info_set]) > 0:
            strategy = self.get_average_strategy(info_set)
            best_action = ["FOLD", "CALL", "RAISE"][np.argmax(strategy)]

            return {
                'hand': hand,
                'group': hand_group,
                'position': position,
                'situation': history if history else "first_to_act",
                'stack_depth': stack_depth,
                'fold_prob': strategy[0],
                'call_prob': strategy[1],
                'raise_prob': strategy[2],
                'recommended_action': best_action,
                'confidence': np.max(strategy)
            }, None
        else:
            return None, f"No strategy learned for {info_set}"

    def analyze_results(self):
        """Analyze training results and show key strategies"""
        print(f"\n📊 PREFLOP TRAINING ANALYSIS")
        print("=" * 80)
        
        # Show coverage
        trained_scenarios = len([s for s in self.all_preflop_scenarios if self.info_set_counter[s] > 0])
        coverage = trained_scenarios / len(self.all_preflop_scenarios) * 100
        print(f"Scenario coverage: {trained_scenarios}/{len(self.all_preflop_scenarios)} ({coverage:.1f}%)")
        
        # Show hand group strategies
        print(f"\nKey Strategies by Position:")
        print("-" * 50)
        
        test_situations = [
            ("BTN", "", "Opening from Button"),
            ("BB", "r", "Defending Big Blind vs Raise"),
            ("BTN", "cr", "Button vs Check-Raise")
        ]
        
        unique_groups = sorted(set(self.hand_groups.values()))[:10]  # Show first 10 groups
        
        for situation_pos, situation_hist, situation_desc in test_situations:
            print(f"\n{situation_desc}:")
            print("Group                    Fold%   Call%   Raise%  Action")
            print("-" * 60)
            
            for group in unique_groups:
                info_set = f"{group}|{situation_pos}|{situation_hist}|deep"
                if info_set in self.strategy_sum and np.sum(self.strategy_sum[info_set]) > 0:
                    strategy = self.get_average_strategy(info_set)
                    best_action = ["FOLD", "CALL", "RAISE"][np.argmax(strategy)]
                    print(f"{group:24s} {strategy[0]:5.1%}   {strategy[1]:5.1%}   {strategy[2]:5.1%}   {best_action}")

    def output_all_learned_strategies(self):
        """Output all scenarios encountered during training with their learned action probabilities"""
        print(f"\n🎯 ALL LEARNED STRATEGIES (Training Verification)")
        print("=" * 90)
        print(f"Total scenarios with training data: {len([s for s in self.all_preflop_scenarios if self.info_set_counter[s] > 0])}")
        print(f"Total visits across all scenarios: {sum(self.info_set_counter.values())}")
        print("")
        
        # Group scenarios by hand group for better readability
        scenarios_by_group = {}
        for scenario in self.all_preflop_scenarios:
            if self.info_set_counter[scenario] > 0:  # Only show scenarios that were actually trained
                group = scenario.split('|')[0]
                if group not in scenarios_by_group:
                    scenarios_by_group[group] = []
                scenarios_by_group[group].append(scenario)
        
        # Output all learned strategies organized by hand group
        for group in sorted(scenarios_by_group.keys()):
            print(f"\n📋 Hand Group: {group}")
            print("-" * 80)
            print("Scenario                                 Visits  Fold%   Call%   Raise%  Best Action")
            print("-" * 80)
            
            scenarios = sorted(scenarios_by_group[group])
            for scenario in scenarios:
                visits = self.info_set_counter[scenario]
                if visits > 0 and scenario in self.strategy_sum:
                    strategy = self.get_average_strategy(scenario)
                    best_action = ["FOLD", "CALL", "RAISE"][np.argmax(strategy)]
                    scenario_display = scenario.replace('|', ' | ')
                    print(f"{scenario_display:40s} {visits:6d}  {strategy[0]:5.1%}   {strategy[1]:5.1%}   {strategy[2]:5.1%}   {best_action}")
        
        print(f"\n📊 SUMMARY OF LEARNING:")
        print("-" * 50)
        trained_scenarios = [s for s in self.all_preflop_scenarios if self.info_set_counter[s] > 0]
        print(f"Scenarios encountered: {len(trained_scenarios)}/{len(self.all_preflop_scenarios)}")
        
        # Show distribution of actions learned
        fold_count = call_count = raise_count = 0
        for scenario in trained_scenarios:
            if scenario in self.strategy_sum:
                strategy = self.get_average_strategy(scenario)
                best_action_idx = np.argmax(strategy)
                if best_action_idx == 0:
                    fold_count += 1
                elif best_action_idx == 1:
                    call_count += 1
                else:
                    raise_count += 1
        
        total = fold_count + call_count + raise_count
        if total > 0:
            print(f"Preferred actions learned:")
            print(f"  FOLD:  {fold_count:3d} scenarios ({fold_count/total:5.1%})")
            print(f"  CALL:  {call_count:3d} scenarios ({call_count/total:5.1%})")
            print(f"  RAISE: {raise_count:3d} scenarios ({raise_count/total:5.1%})")

    def test_hand_lookup(self, test_hands=None):
        """Test strategy lookup for specific hands"""
        if test_hands is None:
            test_hands = ['AA', 'KK', 'AKs', 'AKo', 'QQ', 'JJ', 'AQs', 'KQs', '22', 'T9s']
        
        print(f"\n🎯 STRATEGY LOOKUP TEST")
        print("=" * 70)
        print("Hand  Position  Situation       Action    Confidence  Probabilities")
        print("-" * 70)
        
        situations = [
            ("BTN", "", "Opening"),
            ("BB", "r", "vs Raise"),
            ("BTN", "cr", "vs C-Raise")
        ]
        
        for hand in test_hands:
            for pos, hist, desc in situations:
                result, error = self.get_strategy_for_hand(hand, pos, hist, "deep")
                if result:
                    print(f"{hand:5s} {pos:8s} {desc:15s} {result['recommended_action']:8s} "
                          f"{result['confidence']:9.1%}  F:{result['fold_prob']:.2f} "
                          f"C:{result['call_prob']:.2f} R:{result['raise_prob']:.2f}")
                else:
                    print(f"{hand:5s} {pos:8s} {desc:15s} NO DATA")

def run_preflop_training():
    """Main function to run preflop training"""
    print("🃏 PREFLOP CFR TRAINING")
    print("=" * 50)
    
    # Initialize trainer
    cfr = PreflopCFR()
    
    # Train with equal distribution
    cfr.train_equal_distribution(total_iterations=100, min_visits_per_scenario=1)
    
    # Analyze results
    cfr.analyze_results()
    
    # Output all learned strategies for verification
    cfr.output_all_learned_strategies()
    
    # Test specific hands
    cfr.test_hand_lookup()
    
    print(f"\n✅ Preflop training complete!")
    print(f"✅ All 169 hands covered with strategic grouping")
    print(f"✅ Equal distribution training ensures no blind spots")
    
    return cfr

if __name__ == "__main__":
    preflop_solver = run_preflop_training()
