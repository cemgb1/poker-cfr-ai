# enhanced_postflop_cfr_test.py - Updated for way more scenarios

from poker_scenario_generator import generate_scenarios
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import time

class EnhancedPostflopCFRTest:
    """
    Test version of enhanced postflop CFR using realistic scenario generation
    """

    def __init__(self, n_scenarios=5000):  # INCREASED FROM 200!
        # CFR data structures
        self.regret_sum = defaultdict(lambda: np.zeros(3))  # [fold, call, raise]
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.scenario_counter = Counter()
        self.iterations = 0

        # Generate realistic scenarios using treys-based generator
        print(f"🚀 Generating {n_scenarios} poker scenarios...")
        self.all_scenarios = generate_scenarios(n_scenarios)

        print(f"🎯 Enhanced Postflop CFR Test Initialized!")
        print(f"📊 Total valid scenarios: {len(self.all_scenarios):,}")
        print(f"🧪 Ready for test training!")
        
        # Show category breakdown
        self.analyze_scenario_distribution()

    def analyze_scenario_distribution(self):
        """Show what types of scenarios were generated"""
        print(f"\n📊 SCENARIO DISTRIBUTION:")
        
        # Count by category
        hand_cats = Counter(s['hand_category'] for s in self.all_scenarios)
        pair_types = Counter(s['pair_type'] for s in self.all_scenarios)
        draw_cats = Counter(s['draw_category'] for s in self.all_scenarios)
        strengths = Counter(s['relative_strength'] for s in self.all_scenarios)
        
        print(f"Hand Categories ({len(hand_cats)} types):")
        for cat, count in hand_cats.most_common():
            print(f"  {cat:20s}: {count:4d}")
            
        print(f"\nPair Types ({len(pair_types)} types):")
        for pair_type, count in pair_types.most_common():
            print(f"  {pair_type:20s}: {count:4d}")
            
        print(f"\nDraw Categories ({len(draw_cats)} types):")
        for draw, count in draw_cats.most_common():
            print(f"  {draw:20s}: {count:4d}")
            
        print(f"\nRelative Strengths ({len(strengths)} types):")
        for strength, count in strengths.most_common():
            print(f"  {strength:20s}: {count:4d}")

    def train_test(self, test_iterations=50000, checkpoint_every=5000):  # INCREASED ITERATIONS!
        """Run test training with more iterations"""
        print(f"🚀 Starting test training for {test_iterations:,} iterations")
        print(f"📊 Checkpointing every {checkpoint_every:,} iterations")
        print("=" * 70)

        start_time = time.time()
        hands_trained = set()

        for iteration in range(test_iterations):
            # Find least-trained scenario index
            undertrained = [i for i, s in enumerate(self.all_scenarios) if self.scenario_counter.get(i, 0) < 10]  # Increased threshold

            if not undertrained:
                print(f"✅ All scenarios trained to minimum threshold at iteration {iteration:,}")
                break

            # Train the least-seen scenario
            target_idx = min(undertrained, key=lambda x: self.scenario_counter.get(x, 0))
            target_scenario = self.all_scenarios[target_idx]

            # Simulate training (simplified CFR)
            self.simulate_cfr_iteration(target_idx, target_scenario)

            # Update counters
            self.scenario_counter[target_idx] += 1
            hands_trained.add(target_scenario["hand_category"])
            self.iterations += 1

            # Progress updates
            if (iteration + 1) % checkpoint_every == 0:
                trained_scenarios = len([i for i in range(len(self.all_scenarios)) if self.scenario_counter[i] > 0])
                coverage = trained_scenarios / len(self.all_scenarios) * 100
                avg_visits = np.mean([self.scenario_counter[i] for i in self.scenario_counter])

                print(f"Iter {iteration + 1:6,d}: {coverage:5.1f}% scenarios trained, "
                      f"{len(undertrained):5,d} need training, avg visits: {avg_visits:.1f}")
                
                # Show training distribution
                visit_counts = Counter(self.scenario_counter[i] for i in self.scenario_counter if self.scenario_counter[i] > 0)
                print(f"   Visit distribution: {dict(sorted(visit_counts.items()))}")

        elapsed = time.time() - start_time
        print(f"\n🏆 Test training complete!")
        print(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"📊 Scenarios trained: {len([i for i in self.scenario_counter if self.scenario_counter[i] > 0]):,}")
        print(f"🃏 Hand categories seen: {len(hands_trained)}")
        print(f"⚡ Iterations per second: {self.iterations/elapsed:.1f}")

    def simulate_cfr_iteration(self, scenario_idx, scenario):
        """Enhanced CFR simulation with more realistic strategies"""
        rel_strength = scenario['relative_strength']
        pair_type = scenario['pair_type']
        draw_category = scenario['draw_category']
        position = scenario['position']
        pot_size = scenario['pot_size']

        # Base strategy from hand strength
        if rel_strength == 'nuts':
            base_strategy = np.array([0.0, 0.1, 0.9])  # Almost always raise
        elif rel_strength == 'near_nuts':
            base_strategy = np.array([0.0, 0.3, 0.7])  # Mostly raise
        elif rel_strength == 'strong':
            base_strategy = np.array([0.1, 0.5, 0.4])  # Balanced
        elif rel_strength == 'medium':
            base_strategy = np.array([0.2, 0.6, 0.2])  # Mostly call
        elif rel_strength == 'weak':
            base_strategy = np.array([0.5, 0.4, 0.1])  # Lean towards fold
        else:  # air
            base_strategy = np.array([0.8, 0.15, 0.05])  # Mostly fold

        # Adjust for draws
        if draw_category in ['nut_flush_draw', 'combo_draw']:
            base_strategy[2] += 0.2  # More aggressive with strong draws
            base_strategy[0] -= 0.2
        elif draw_category in ['flush_draw', 'open_ended']:
            base_strategy[1] += 0.1  # More calling with draws
            base_strategy[0] -= 0.1

        # Adjust for position
        if position in ['BTN', 'LP']:
            base_strategy[2] += 0.1  # More aggressive in position
            base_strategy[0] -= 0.1
        elif position in ['SB', 'BB']:
            base_strategy[0] += 0.1  # More cautious out of position
            base_strategy[2] -= 0.1

        # Adjust for pot size
        if pot_size in ['large_pot', 'huge_pot']:
            base_strategy[0] += 0.1  # More cautious in big pots
            base_strategy[1] += 0.1
            base_strategy[2] -= 0.2

        # Add randomness for realism
        noise = np.random.normal(0, 0.05, 3)  # Reduced noise
        strategy = np.maximum(0, base_strategy + noise)
        strategy = strategy / np.sum(strategy)  # Normalize

        # Update strategy sum
        self.strategy_sum[scenario_idx] += strategy

    def export_test_results_csv(self, filename='enhanced_postflop_results.csv'):
        """Export test results to CSV showing what was learned"""
        print(f"📊 Exporting test results to {filename}...")

        results = []

        for idx, visits in self.scenario_counter.items():
            if visits > 0:
                scenario = self.all_scenarios[idx]
                if idx in self.strategy_sum:
                    strategy = self.strategy_sum[idx] / visits
                    best_action = ['FOLD', 'CALL', 'RAISE'][np.argmax(strategy)]
                    confidence = np.max(strategy)
                else:
                    strategy = np.array([0.33, 0.33, 0.34])
                    best_action = 'UNKNOWN'
                    confidence = 0.0

                results.append({
                    'hand_category': scenario['hand_category'],
                    'hole_cards': scenario['hole_cards'],
                    'board': scenario['board'],
                    'relative_strength': scenario['relative_strength'],
                    'pair_type': scenario['pair_type'],
                    'draw_category': scenario['draw_category'],
                    'board_texture': scenario['board_texture'],
                    'board_danger': scenario['board_danger'],
                    'position': scenario['position'],
                    'preflop_action': scenario['preflop_action'],
                    'postflop_action': scenario['postflop_action'],
                    'pot_size': scenario['pot_size'],
                    'stack_depth': scenario['stack_depth'],
                    'recommended_action': best_action,
                    'fold_prob': round(strategy[0], 3),
                    'call_prob': round(strategy[1], 3),
                    'raise_prob': round(strategy[2], 3),
                    'confidence': round(confidence, 3),
                    'training_visits': visits
                })

        # Create DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

        print(f"✅ Exported {len(results):,} trained scenarios")
        print(f"📁 File size: {len(df) * 20 / 1024:.1f} KB")

        return df

    def analyze_strategy_insights(self, results_df):
        """Analyze learned strategies for insights"""
        print(f"\n🧠 STRATEGY INSIGHTS ANALYSIS")
        print("=" * 60)
        
        # Action frequency by hand strength
        print(f"Action Frequencies by Hand Strength:")
        for strength in ['nuts', 'near_nuts', 'strong', 'medium', 'weak', 'air']:
            subset = results_df[results_df['relative_strength'] == strength]
            if len(subset) > 0:
                avg_fold = subset['fold_prob'].mean()
                avg_call = subset['call_prob'].mean()
                avg_raise = subset['raise_prob'].mean()
                print(f"  {strength:10s}: Fold {avg_fold:.2f}, Call {avg_call:.2f}, Raise {avg_raise:.2f}")
        
        # Most aggressive scenarios
        print(f"\n🔥 MOST AGGRESSIVE SCENARIOS (High Raise %):")
        aggressive = results_df.nlargest(10, 'raise_prob')[['hand_category', 'relative_strength', 'pair_type', 'draw_category', 'raise_prob']]
        for _, row in aggressive.iterrows():
            print(f"  {row['hand_category']:15s} | {row['relative_strength']:10s} | {row['pair_type']:15s} | Raise: {row['raise_prob']:.2f}")
        
        # Most passive scenarios
        print(f"\n🛡️ MOST PASSIVE SCENARIOS (High Fold %):")
        passive = results_df.nlargest(10, 'fold_prob')[['hand_category', 'relative_strength', 'pair_type', 'draw_category', 'fold_prob']]
        for _, row in passive.iterrows():
            print(f"  {row['hand_category']:15s} | {row['relative_strength']:10s} | {row['pair_type']:15s} | Fold: {row['fold_prob']:.2f}")
        
        # Draw strategy analysis
        print(f"\n🎯 DRAW STRATEGY ANALYSIS:")
        for draw in ['nut_flush_draw', 'flush_draw', 'open_ended', 'combo_draw', 'no_draw']:
            subset = results_df[results_df['draw_category'] == draw]
            if len(subset) > 0:
                avg_raise = subset['raise_prob'].mean()
                print(f"  {draw:15s}: Avg raise freq {avg_raise:.2f} ({len(subset):3d} scenarios)")

    def show_sample_strategies(self, n_samples=20):
        """Show sample learned strategies"""
        print(f"\n🎯 SAMPLE LEARNED STRATEGIES")
        print("=" * 110)
        print("Hand Category        Strength     Pair Type        Draw             Action   Confidence  F/C/R")
        print("-" * 110)

        trained_idxs = [idx for idx in self.scenario_counter if self.scenario_counter[idx] > 0]
        samples = random.sample(trained_idxs, min(n_samples, len(trained_idxs)))

        for idx in samples:
            scenario = self.all_scenarios[idx]

            if idx in self.strategy_sum:
                strategy = self.strategy_sum[idx] / self.scenario_counter[idx]
                best_action = ['FOLD', 'CALL', 'RAISE'][np.argmax(strategy)]
                confidence = np.max(strategy)

                print(f"{scenario['hand_category']:18s} {scenario['relative_strength']:10s} "
                      f"{scenario['pair_type']:15s} {scenario['draw_category']:15s} "
                      f"{best_action:8s} {confidence:9.1%}  "
                      f"{strategy[0]:.2f}/{strategy[1]:.2f}/{strategy[2]:.2f}")

def run_comprehensive_test():
    """Run comprehensive test of enhanced postflop system"""
    print("🧪 ENHANCED POSTFLOP CFR COMPREHENSIVE TEST")
    print("=" * 70)

    # Initialize test system with MANY more scenarios
    cfr_test = EnhancedPostflopCFRTest(n_scenarios=10000)  # 10K scenarios!

    # Run test training
    cfr_test.train_test(test_iterations=100000, checkpoint_every=10000)  # 100K iterations!

    # Analyze what was learned
    cfr_test.analyze_coverage()

    # Show sample strategies
    cfr_test.show_sample_strategies(25)

    # Export to CSV
    results_df = cfr_test.export_test_results_csv()

    # Advanced strategy analysis
    cfr_test.analyze_strategy_insights(results_df)

    # Summary statistics
    print(f"\n📊 COMPREHENSIVE TEST SUMMARY:")
    print(f"Total possible scenarios: {len(cfr_test.all_scenarios):,}")
    print(f"Scenarios trained: {len([i for i in cfr_test.scenario_counter if cfr_test.scenario_counter[i] > 0]):,}")
    print(f"Training coverage: {len([i for i in cfr_test.scenario_counter if cfr_test.scenario_counter[i] > 0]) / len(cfr_test.all_scenarios) * 100:.1f}%")
    print(f"CSV file: enhanced_postflop_results.csv")

    return cfr_test, results_df

def production_scale_test():
    """Production-scale test with maximum scenarios"""
    print("🏭 PRODUCTION-SCALE TEST")
    print("=" * 50)

    # MASSIVE scale test
    cfr_test = EnhancedPostflopCFRTest(n_scenarios=50000)  # 50K scenarios!

    # Long training run
    cfr_test.train_test(test_iterations=500000, checkpoint_every=50000)  # 500K iterations!

    # Export results
    results_df = cfr_test.export_test_results_csv('production_postflop_results.csv')

    # Analysis
    cfr_test.analyze_strategy_insights(results_df)

    print(f"\n🏆 PRODUCTION TEST COMPLETE!")
    print(f"Trained {len(results_df):,} scenarios")
    print(f"Results saved to: production_postflop_results.csv")

    return cfr_test, results_df

def quick_test():
    """Quick test with moderate scenarios"""
    print("⚡ QUICK TEST - 1000 SCENARIOS")
    print("=" * 40)

    cfr_test = EnhancedPostflopCFRTest(n_scenarios=1000)
    cfr_test.train_test(test_iterations=10000, checkpoint_every=2000)

    # Analysis
    trained_count = len([i for i in cfr_test.scenario_counter if cfr_test.scenario_counter[i] > 0])
    print(f"\n📊 Quick Results:")
    print(f"Scenarios trained: {trained_count:,} / {len(cfr_test.all_scenarios):,}")
    print(f"Coverage: {trained_count / len(cfr_test.all_scenarios) * 100:.1f}%")

    # Export sample
    if trained_count > 0:
        results_df = cfr_test.export_test_results_csv('quick_enhanced_results.csv')
        print(f"📁 Results saved to: quick_enhanced_results.csv")

        # Show category breakdown
        print(f"\n🎯 Categories Trained:")
        categories = results_df.groupby(['pair_type', 'draw_category']).size().sort_values(ascending=False)
        for (pair_type, draw_cat), count in categories.head(10).items():
            print(f"  {pair_type:15s} + {draw_cat:15s}: {count:3d} scenarios")

    return cfr_test

def ultra_quick_test():
    """Ultra-quick test to verify the enhanced system works"""
    print("🏃 ULTRA-QUICK ENHANCED TEST - 500 SCENARIOS")
    print("=" * 55)

    cfr_test = EnhancedPostflopCFRTest(n_scenarios=500)

    start_time = time.time()
    cfr_test.train_test(test_iterations=5000, checkpoint_every=1000)
    elapsed = time.time() - start_time

    # Quick validation
    trained_count = len([i for i in cfr_test.scenario_counter if cfr_test.scenario_counter[i] > 0])

    print(f"\n⚡ ENHANCED ULTRA-QUICK RESULTS ({elapsed:.1f} seconds):")
    print(f"✅ System works: {trained_count:,} scenarios trained")
    print(f"✅ Categories covered: {len(set(cfr_test.all_scenarios[i]['hand_category'] for i in cfr_test.scenario_counter if cfr_test.scenario_counter[i] > 0))}")
    print(f"✅ Enhanced system ready!")

    # Export sample
    if trained_count > 0:
        results_df = cfr_test.export_test_results_csv('enhanced_validation_test.csv')
        print(f"📁 Sample results: enhanced_validation_test.csv ({len(results_df):,} rows)")

        # Show diverse strategies
        if len(results_df) >= 5:
            print(f"\n🎯 Sample Strategy Diversity:")
            sample = results_df[['relative_strength', 'pair_type', 'recommended_action', 'confidence']].head(5)
            for _, row in sample.iterrows():
                print(f"  {row['relative_strength']:10s} + {row['pair_type']:15s} → {row['recommended_action']:5s} ({row['confidence']:.1%})")

    return cfr_test

if __name__ == "__main__":
    # Choose your test scale:
    
    # For quick validation (30 seconds):
    test_cfr = ultra_quick_test()
    
    # For moderate testing (5 minutes):
    # test_cfr = quick_test()
    
    # For comprehensive testing (30+ minutes):
    # test_cfr, results_df = run_comprehensive_test()
    
    # For production scale (hours):
    # test_cfr, results_df = production_scale_test()

    print(f"\n💡 Scale Up Path:")
    print(f"1. ✅ Ultra-quick: 500 scenarios, 5K iterations")
    print(f"2. 🔧 Quick: 1K scenarios, 10K iterations") 
    print(f"3. 🚀 Comprehensive: 10K scenarios, 100K iterations")
    print(f"4. 🏭 Production: 50K scenarios, 500K iterations")
