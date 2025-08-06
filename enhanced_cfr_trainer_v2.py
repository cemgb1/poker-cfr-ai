# enhanced_cfr_trainer_v2.py - Simplified CFR with Monte Carlo sampling

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
    Simplified CFR with Monte Carlo sampling and heads-up play
    """
    
    def __init__(self, scenarios=None, n_scenarios=1000, monte_carlo=False, simulations_per_scenario=100):
        # CFR data structures - now with variable action counts
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.scenario_counter = Counter()
        self.iterations = 0
        self.monte_carlo = monte_carlo
        self.simulations_per_scenario = simulations_per_scenario

        # Performance metrics tracking
        self.performance_metrics = []
        self.start_time = None
        self.last_iteration_time = None
        
        if monte_carlo:
            # Monte Carlo mode - use pre-defined scenarios, run X simulations each
            if scenarios is not None:
                self.scenarios = scenarios
            else:
                from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
                self.scenarios = generate_enhanced_scenarios(n_scenarios)
            
            print(f"🚀 Monte Carlo CFR Mode - {simulations_per_scenario} simulations per scenario")
            print(f"🎯 Total target iterations: {len(self.scenarios) * simulations_per_scenario}")
        else:
            # Traditional mode - use pre-generated scenarios
            if scenarios is not None:
                self.scenarios = scenarios
                print(f"🚀 Using provided {len(scenarios)} scenarios...")
            else:
                from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
                self.scenarios = generate_enhanced_scenarios(n_scenarios)
                print(f"🚀 Generated {n_scenarios} scenarios...")

        # Balanced hand category sampling (after scenarios are loaded)
        self.hand_category_visits = defaultdict(int)
        self.scenarios_by_category = self._group_scenarios_by_category()
            
        print(f"🏆 {'Monte Carlo' if monte_carlo else 'Simplified'} CFR Trainer Initialized!")
        print(f"📊 Training scenarios: {len(self.scenarios):,}")
        if not monte_carlo:
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
        Select scenario for Monte Carlo CFR or traditional balanced sampling.
        In Monte Carlo mode, ensures each scenario gets exactly X simulations before moving to next.
        """
        if self.monte_carlo:
            return self.select_monte_carlo_scenario()
        
        # Traditional mode - select from pre-generated scenarios with balance
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

    def select_monte_carlo_scenario(self):
        """
        Select next scenario for Monte Carlo CFR.
        Ensures each scenario gets exactly simulations_per_scenario visits before moving to next.
        """
        # Find scenario with fewest visits
        scenario_visits = {self.get_scenario_key(s): self.scenario_counter.get(self.get_scenario_key(s), 0) 
                          for s in self.scenarios}
        
        # Get scenario with minimum visits (round-robin style)
        min_visits = min(scenario_visits.values()) if scenario_visits else 0
        
        # Find all scenarios with minimum visits
        candidates = [s for s in self.scenarios 
                     if self.scenario_counter.get(self.get_scenario_key(s), 0) == min_visits]
        
        # Select randomly from candidates (for variety within each round)
        selected_scenario = random.choice(candidates)
        
        return selected_scenario

    def should_continue_training(self, min_visits_per_scenario=None):
        """
        Determine if training should continue based on dynamic stopping criteria.
        
        Args:
            min_visits_per_scenario: Minimum number of times each scenario should be visited.
                                   For Monte Carlo mode, defaults to simulations_per_scenario.
            
        Returns:
            bool: True if training should continue, False if stopping criteria met
        """
        if not self.scenario_counter:
            return True  # Continue if no scenarios visited yet
        
        # Set default minimum visits
        if min_visits_per_scenario is None:
            min_visits_per_scenario = self.simulations_per_scenario if self.monte_carlo else 100
        
        # Check minimum visits per scenario
        min_visits = min(self.scenario_counter.values()) if self.scenario_counter else 0
        
        if self.monte_carlo:
            # Monte Carlo mode: stop when all scenarios have required visits
            return min_visits < min_visits_per_scenario
        else:
            # Traditional mode: stop when minimum visits reached
            return min_visits < min_visits_per_scenario

    def sample_action(self, strategy, available_actions):
        """Sample action from strategy probabilities"""
        probs = [strategy[action] for action in available_actions]
        chosen_idx = np.random.choice(len(available_actions), p=probs)
        return available_actions[chosen_idx]

    def play_enhanced_scenario(self, scenario):
        """Play simplified scenario with stack considerations"""
        scenario_key = self.get_scenario_key(scenario)
        available_actions = scenario["available_actions"]
        
        # Hero's decision using CFR strategy
        hero_strategy = self.get_strategy(scenario_key, available_actions)
        hero_action = self.sample_action(hero_strategy, available_actions)
        
        # Generate villain and their action
        villain_cards = self.generate_villain_hand(scenario['hero_cards_int'])
        villain_action = self.get_enhanced_villain_action(villain_cards, scenario)
        
        # Calculate simplified payoff 
        payoff_result = self.calculate_simplified_payoff(scenario, hero_action, villain_action, villain_cards)
        
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
            'available_actions': available_actions
        }

    def calculate_simplified_payoff(self, scenario, hero_action, villain_action, villain_cards):
        """Calculate payoff with simplified stack considerations"""
        hero_stack_before = scenario['hero_stack_bb']
        bet_amount = scenario['bet_to_call_bb']
        
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
        
        # Simple payoff - just the normalized stack change
        payoff = stack_change / hero_stack_before if hero_stack_before > 0 else 0
        
        return {
            'payoff': payoff,
            'hero_stack_after': hero_stack_after,
            'stack_change': stack_change
        }

    def get_bet_amount(self, action, stack_size, current_bet):
        """Convert simplified action to bet amount"""
        if action == "fold":
            return 0
        elif action == "call":
            return current_bet
        elif action == "raise_small":
            return current_bet * 2.5
        elif action == "shove":
            return stack_size
        else:
            return current_bet  # Default to call



    def generate_villain_hand(self, hero_cards):
        """Generate random villain hand"""
        deck = Deck()
        for card in hero_cards:
            if card in deck.cards:
                deck.cards.remove(card)
        return deck.draw(2)

    def get_enhanced_villain_action(self, villain_cards, scenario):
        """Simplified villain action based on stack context"""
        try:
            villain_equity = self.estimate_villain_equity(villain_cards)
            villain_stack = scenario['villain_stack_bb']
            
            # Adjust strategy based on stack size
            if villain_stack <= 15:  # Short stack - push/fold
                if villain_equity > 0.45:
                    return "shove"
                else:
                    return "fold"
            elif villain_stack <= 30:  # Medium short stack - tighter play
                if villain_equity > 0.6:
                    return "shove"
                elif villain_equity > 0.5:
                    return "call"
                else:
                    return "fold"
            
            else:  # Normal stack - standard play
                if villain_equity > 0.65:
                    return random.choice(["shove", "raise_small"])
                elif villain_equity > 0.45:
                    return random.choice(["call", "raise_small"])
                else:
                    return random.choice(["fold", "call"]) if random.random() < 0.8 else "fold"
                    
        except:
            return random.choice(["fold", "call", "raise_small"])

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
        """Simplified scenario key with stack context"""
        bet_situation = "bet" if scenario['bet_to_call_bb'] > 0 else "no_bet"
        return (f"{scenario['hand_category']}_{scenario['hero_position']}_"
                f"{scenario['stack_category']}_{bet_situation}")

    def export_strategies_to_csv(self, filename="simplified_cfr_strategies.csv"):
        """
        Export all learned strategies to CSV with simplified scenario details.
        Includes probabilities for each action, scenario details (hole cards, position, 
        stack depth, betting situation), and best action as determined by model.
        
        CSV columns include:
        - scenario_key: Unique identifier for the scenario
        - hand_category: Broad category of hole cards (premium_pairs, medium_aces, etc.)
        - example_hands: Sample hands from this category  
        - position: Hero's position (BTN/BB)
        - stack_depth: Stack size category (ultra_short, short, medium, deep, very_deep)
        - betting_situation: Whether there's a bet to call (bet/no_bet)
        - training_games: Number of training iterations for this scenario
        - best_action: Recommended action (FOLD, CALL, RAISE_SMALL, SHOVE)
        - confidence: Probability of best action (0-1)
        - fold_prob, call_prob, raise_small_prob, shove_prob: Probability of each action
        """
        import pandas as pd
        from enhanced_cfr_preflop_generator_v2 import ACTIONS, PREFLOP_HAND_RANGES
        
        print(f"📊 Exporting strategies to {filename}...")
        
        # Collect all strategy data
        export_data = []
        
        for scenario_key, strategy_counts in self.strategy_sum.items():
            if sum(strategy_counts.values()) > 0:  # Only export scenarios with data
                
                # Parse simplified scenario key
                parts = scenario_key.split("_")
                if len(parts) >= 4:
                    hand_category = parts[0]
                    position = parts[1] 
                    stack_category = parts[2]
                    betting_situation = parts[3]  # "bet" or "no_bet"
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
                
                # Build simplified export row
                row = {
                    'scenario_key': scenario_key,
                    'hand_category': hand_category,
                    'example_hands': example_hands,
                    'position': position,
                    'stack_depth': stack_category,
                    'betting_situation': betting_situation,
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
        
        # Calculate theoretical maximum scenarios (simplified)
        hand_categories = len(PREFLOP_HAND_RANGES)
        positions = 2  # BTN, BB
        stack_categories = len(STACK_CATEGORIES)
        betting_situations = 2  # bet, no_bet
        
        total_possible = hand_categories * positions * stack_categories * betting_situations
        
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
    
    def run_monte_carlo_training(n_scenarios=100, simulations_per_scenario=50, metrics_interval=1000):
        """Run Monte Carlo CFR training with X simulations per scenario"""
        print(f"🎯 Running Monte Carlo CFR Training")
        print(f"Scenarios: {n_scenarios}, Simulations per scenario: {simulations_per_scenario}")
        print(f"Total iterations target: {n_scenarios * simulations_per_scenario} (X × n)")
        print(f"Action set: FOLD, CALL, RAISE_SMALL, SHOVE")
        print(f"🎯 Using ROUND-ROBIN scenario coverage")
        print("=" * 70)
        
        # Generate scenarios first
        from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
        scenarios = generate_enhanced_scenarios(n_scenarios)
        
        # Initialize Monte Carlo trainer
        trainer = EnhancedCFRTrainer(scenarios=scenarios, monte_carlo=True, 
                                    simulations_per_scenario=simulations_per_scenario)
        
        # Start performance tracking
        trainer.start_performance_tracking()
        
        # Train with dynamic stopping criteria
        print(f"\n🎯 Starting Monte Carlo CFR training...")
        iteration = 0
        
        while trainer.should_continue_training():
            # Select scenario using round-robin approach
            scenario = trainer.select_balanced_scenario()
            trainer.play_enhanced_scenario(scenario)
            trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1
            iteration += 1
            
            # Record metrics at regular intervals
            if iteration % metrics_interval == 0:
                metrics = trainer.record_iteration_metrics(iteration)
                min_visits = min(trainer.scenario_counter.values()) if trainer.scenario_counter else 0
                max_visits = max(trainer.scenario_counter.values()) if trainer.scenario_counter else 0
                print(f"Iteration {iteration:6d}: {metrics['unique_scenarios_visited']:3d} scenarios, "
                      f"visits: min={min_visits}, max={max_visits}")
        
        # Record final metrics
        final_metrics = trainer.record_iteration_metrics(iteration - 1)
        
        print(f"✅ Monte Carlo training complete after {iteration:,} iterations")
        print(f"📊 All scenarios visited {simulations_per_scenario} times")
        print(f"🎯 Hand category coverage balance:")
        total_visits = sum(trainer.hand_category_visits.values())
        for category, visits in trainer.hand_category_visits.items():
            percentage = (visits / total_visits) * 100 if total_visits > 0 else 0
            print(f"   {category:15s}: {visits:6d} visits ({percentage:5.1f}%)")
        
        # Show scenario visit distribution
        min_visits = min(trainer.scenario_counter.values()) if trainer.scenario_counter else 0
        max_visits = max(trainer.scenario_counter.values()) if trainer.scenario_counter else 0
        print(f"\n📈 All scenarios have {min_visits}-{max_visits} visits (target: {simulations_per_scenario})")
        
        # Export both strategy results and performance metrics
        trainer.export_strategies_to_csv("monte_carlo_cfr_results.csv")
        trainer.export_performance_metrics("monte_carlo_performance.csv")
        
        return trainer
    
    # Add a simple training function for testing
    def run_enhanced_training(n_scenarios=100, n_iterations=200000, metrics_interval=1000):
        """Run enhanced CFR training with balanced sampling and performance tracking"""
        print(f"🚀 Running Enhanced CFR Training with Balanced Hand Category Coverage")
        print(f"Scenarios: {n_scenarios}, Iterations: {n_iterations}")
        print(f"Metrics interval: every {metrics_interval} iterations")
        print(f"Action set: FOLD, CALL, RAISE_SMALL, SHOVE")
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
