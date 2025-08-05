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
        
        # Performance metrics tracking
        self.performance_metrics = []
        self.start_time = None
        self.last_iteration_time = None
        
        # Generate enhanced scenarios, or use provided ones
        if scenarios is not None:
            self.scenarios = scenarios
            print(f"🚀 Using provided {len(scenarios)} scenarios...")
        else:
            self.scenarios = generate_enhanced_scenarios(n_scenarios)
            print(f"🚀 Generated {n_scenarios} scenarios...")

        # Balanced hand category sampling (after scenarios are loaded)
        self.hand_category_visits = defaultdict(int)
        self.scenarios_by_category = self._group_scenarios_by_category()
            
        print(f"🏆 Enhanced CFR Trainer Initialized!")
        print(f"📊 Training scenarios: {len(self.scenarios):,}")
        print(f"📈 Hand categories: {len(self.scenarios_by_category)} balanced groups")

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
        """Calculate payoff with stack survival considerations (tournament_stage removed)"""
        hero_stack_before = scenario['hero_stack_bb']
        bet_amount = scenario['bet_to_call_bb']
        # tournament_stage removed - using stack-based logic instead
        
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
        
        # Stack-based adjustments (replacing tournament-specific logic)
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

    def get_enhanced_villain_action(self, villain_cards, scenario):
        """Enhanced villain action based on stack context (tournament_stage removed)"""
        try:
            villain_equity = self.estimate_villain_equity(villain_cards)
            villain_stack = scenario['villain_stack_bb']
            # tournament_stage removed - using stack-based logic instead
            
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
        """Enhanced scenario key with stack context (tournament_stage removed)"""
        return (f"{scenario['hand_category']}_{scenario['hero_position']}_"
                f"{scenario['stack_category']}_{scenario['bet_size_category']}")

    def export_strategies_to_csv(self, filename="enhanced_cfr_strategies.csv"):
        """
        Export all learned strategies to CSV with comprehensive scenario details.
        Includes probabilities for each action, scenario details (hole cards, position, 
        stack depth, bet sizing info), and best action as determined by model.
        
        CSV columns include:
        - scenario_key: Unique identifier for the scenario
        - hand_category: Broad category of hole cards (premium_pairs, medium_aces, etc.)
        - example_hands: Sample hands from this category  
        - position: Hero's position (BTN/BB)
        - stack_depth: Stack size category (ultra_short, short, medium, deep, very_deep)
        - bet_size_category: Size of bet to call (tiny, small, medium, large, no_bet)
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
                
                # Parse scenario key (tournament_stage removed)
                parts = scenario_key.split("_")
                if len(parts) >= 4:
                    hand_category = parts[0]
                    position = parts[1] 
                    stack_category = parts[2]
                    bet_size_category = parts[3]
                else:
                    continue  # Skip malformed keys
                
                # Get example hands for this category
                example_hands = ""
                if hand_category in PREFLOP_HAND_RANGES:
                    examples = PREFLOP_HAND_RANGES[hand_category][:3]  # First 3 examples
                    example_hands = ", ".join(examples)
                
                # Calculate average strategy (normalized probabilities)
                total_count = sum(strategy_counts.values())
                action_probs = {}
                
                # Initialize all action probabilities to 0
                for action_name in ACTIONS.keys():
                    action_probs[f"{action_name}_prob"] = 0.0
                
                # Fill in actual probabilities
                for action, count in strategy_counts.items():
                    if action in ACTIONS:
                        action_probs[f"{action}_prob"] = count / total_count
                
                # Determine best action
                best_action = max(strategy_counts.items(), key=lambda x: x[1])[0]
                best_action_confidence = max(action_probs.values())
                
                # Get training count for this scenario
                training_games = self.scenario_counter.get(scenario_key, 0)
                
                # Build export row (tournament_stage removed)
                row = {
                    'scenario_key': scenario_key,
                    'hand_category': hand_category,
                    'example_hands': example_hands,
                    'position': position,
                    'stack_depth': stack_category,
                    'bet_size_category': bet_size_category,
                    'training_games': training_games,
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
            
            # Show summary stats
            print(f"\n📈 EXPORT SUMMARY:")
            print(f"Total scenarios: {len(export_data)}")
            print(f"Avg training games per scenario: {df['training_games'].mean():.1f}")
            print(f"Avg confidence: {df['confidence'].mean():.3f}")
            
            # Show action distribution
            print(f"\nBest Action Distribution:")
            action_dist = df['best_action'].value_counts()
            for action, count in action_dist.items():
                print(f"  {action}: {count} scenarios ({count/len(export_data)*100:.1f}%)")
            
            return df
        else:
            print("❌ No strategy data to export")
            return None

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
        
        # Calculate theoretical maximum scenarios
        hand_categories = len(PREFLOP_HAND_RANGES)
        positions = 2  # BTN, BB
        stack_categories = len(STACK_CATEGORIES)
        bet_size_categories = 5  # no_bet, tiny, small, large, huge
        blinds_levels = 3  # low, medium, high
        
        total_possible = hand_categories * positions * stack_categories * bet_size_categories * blinds_levels
        
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

if __name__ == "__main__":
    print("Enhanced CFR Trainer v2 - Ready for training!")
    
    # Add a simple training function for testing
    def run_enhanced_training(n_scenarios=100, n_iterations=200000, metrics_interval=1000):
        """Run enhanced CFR training with balanced sampling and performance tracking"""
        print(f"🚀 Running Enhanced CFR Training with Balanced Hand Category Coverage")
        print(f"Scenarios: {n_scenarios}, Iterations: {n_iterations}")
        print(f"Metrics interval: every {metrics_interval} iterations")
        print(f"Action set: FOLD, CALL_SMALL, CALL_MID, CALL_HIGH, RAISE_SMALL, RAISE_MID, RAISE_HIGH")
        print(f"🎯 Using STRATIFIED SAMPLING for balanced hand category coverage")
        print("=" * 70)
        
        # Generate scenarios
        from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
        scenarios = generate_enhanced_scenarios(n_scenarios)
        
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
    
    # Uncomment to run training:
    # trainer = run_enhanced_training()
