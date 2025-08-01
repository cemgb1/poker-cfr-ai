# -*- coding: utf-8 -*-
"""
Complete CFR with Checkpointing System
Saves training state periodically and can resume from checkpoints
"""

import pickle
import json
import os
import time
import random
import numpy as np
from collections import defaultdict, Counter

class CheckpointManager:
    """
    Manages saving and loading CFR training checkpoints
    """
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, cfr_solver, iteration, checkpoint_name=None):
        """Save complete CFR state to checkpoint file"""
        if checkpoint_name is None:
            checkpoint_name = f"cfr_checkpoint_iter_{iteration}.pkl"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        print(f"💾 Saving checkpoint at iteration {iteration}...")
        
        # Prepare checkpoint data
        checkpoint_data = {
            'iteration': iteration,
            'timestamp': time.time(),
            'regret_sum': dict(cfr_solver.regret_sum),
            'strategy_sum': dict(cfr_solver.strategy_sum),
            'info_set_counter': dict(cfr_solver.info_set_counter),
            'hand_groups': cfr_solver.hand_groups,
            'all_hands': cfr_solver.all_hands,
            'metadata': {
                'total_scenarios': len(getattr(cfr_solver, 'all_preflop_scenarios', [])) + 
                                 len(getattr(cfr_solver, 'all_postflop_scenarios', [])),
                'solver_type': type(cfr_solver).__name__,
                'save_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save with compression
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save human-readable summary
        summary_path = checkpoint_path.replace('.pkl', '_summary.json')
        summary = {
            'iteration': iteration,
            'save_time': checkpoint_data['metadata']['save_time'],
            'scenarios_trained': len([k for k, v in cfr_solver.info_set_counter.items() if v > 0]),
            'total_training_examples': sum(cfr_solver.info_set_counter.values()),
            'file_size_mb': round(os.path.getsize(checkpoint_path) / 1024 / 1024, 2),
            'top_trained_scenarios': sorted(cfr_solver.info_set_counter.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Checkpoint saved: {checkpoint_path} ({summary['file_size_mb']} MB)")
        print(f"📊 Scenarios trained: {summary['scenarios_trained']}")
        print(f"📊 Total examples: {summary['total_training_examples']:,}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load CFR state from checkpoint file"""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"✅ Loaded checkpoint from iteration {checkpoint_data['iteration']}")
        print(f"📅 Saved: {checkpoint_data['metadata']['save_time']}")
        print(f"🎯 Solver type: {checkpoint_data['metadata']['solver_type']}")
        
        return checkpoint_data
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(self.checkpoint_dir, file)
                try:
                    # Quick peek at checkpoint info
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    checkpoints.append({
                        'filename': file,
                        'path': file_path,
                        'iteration': data['iteration'],
                        'timestamp': data['timestamp'],
                        'size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2),
                        'scenarios_trained': len([k for k, v in data['info_set_counter'].items() if v > 0])
                    })
                except:
                    continue
        
        return sorted(checkpoints, key=lambda x: x['iteration'])
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Keep only the most recent N checkpoints to save disk space"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        to_delete = checkpoints[:-keep_last_n]
        
        for checkpoint in to_delete:
            os.remove(checkpoint['path'])
            # Also remove summary file
            summary_path = checkpoint['path'].replace('.pkl', '_summary.json')
            if os.path.exists(summary_path):
                os.remove(summary_path)
            print(f"🗑️ Deleted old checkpoint: {checkpoint['filename']}")


class ResumablePreflopCFR:
    """
    Preflop CFR with checkpoint support
    """
    
    def __init__(self, big_blind=2, checkpoint_manager=None):
        self.regret_sum = defaultdict(lambda: np.zeros(3))
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.info_set_counter = Counter()
        self.iterations = 0
        self.big_blind = big_blind
        
        # Checkpoint management
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Initialize poker-specific data
        self.hand_groups = self.create_hand_groups()
        self.all_hands = self.generate_all_169_hands()
        self.all_preflop_scenarios = self.generate_all_preflop_scenarios()
        
        print(f"🎯 ResumablePreflopCFR initialized with {len(self.all_preflop_scenarios)} scenarios")
    
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

    def create_hand_groups(self):
        """Create hand grouping system"""
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

    def train_with_checkpoints(self, total_iterations=500000, min_visits_per_scenario=300, 
                              checkpoint_every=5000, resume_from=None):
        """
        Train with automatic checkpointing and resume capability
        """
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_from_checkpoint(resume_from)
            print(f"🔄 Resuming from iteration {self.iterations}")
        
        print(f"🚀 Training with checkpoints every {checkpoint_every} iterations")
        print(f"Target: {min_visits_per_scenario} visits per scenario")
        
        start_time = time.time()
        start_iteration = self.iterations
        hands_seen = set()
        
        while self.iterations < total_iterations:
            # Find undertrained scenarios
            undertrained = [
                scenario for scenario in self.all_preflop_scenarios 
                if self.info_set_counter.get(scenario, 0) < min_visits_per_scenario
            ]

            if not undertrained:
                print(f"✅ All scenarios trained to {min_visits_per_scenario}+ visits at iteration {self.iterations}")
                break

            # Train least-seen scenario
            target_scenario = min(undertrained, key=lambda x: self.info_set_counter.get(x, 0))
            
            # Parse and train
            group, position, history, stack_cat = target_scenario.split("|")
            candidate_hands = [h for h, g in self.hand_groups.items() if g == group]
            p0_hand = random.choice(candidate_hands)
            p1_hand = random.choice([h for h in self.all_hands if h != p0_hand])
            stack_bb = {"short": 20, "medium": 50, "deep": 100}[stack_cat]
            
            player = 0 if position == "BTN" else 1
            self.cfr_preflop(p0_hand, p1_hand, history, 1.0, 1.0, player, stack_bb)
            
            # Update counters
            self.info_set_counter[target_scenario] += 1
            hands_seen.add(p0_hand)
            hands_seen.add(p1_hand)
            self.iterations += 1

            # Checkpoint periodically
            if self.iterations % checkpoint_every == 0:
                self.checkpoint_manager.save_checkpoint(self, self.iterations)
                self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)
                
                # Progress report
                completed = len([s for s in self.all_preflop_scenarios 
                               if self.info_set_counter.get(s, 0) >= min_visits_per_scenario])
                progress = completed / len(self.all_preflop_scenarios) * 100
                elapsed_hours = (time.time() - start_time) / 3600
                rate = (self.iterations - start_iteration) / elapsed_hours if elapsed_hours > 0 else 0
                
                print(f"📊 Checkpoint at iteration {self.iterations}")
                print(f"   Progress: {progress:.1f}% scenarios complete")
                print(f"   Undertrained: {len(undertrained)} scenarios")
                print(f"   Training rate: {rate:.0f} iter/hour")

            # Regular progress updates
            if self.iterations % (checkpoint_every // 5) == 0:
                completed = len([s for s in self.all_preflop_scenarios 
                               if self.info_set_counter.get(s, 0) >= min_visits_per_scenario])
                progress = completed / len(self.all_preflop_scenarios) * 100
                print(f"Iter {self.iterations:6d}: {progress:5.1f}% complete, "
                      f"{len(undertrained):3d} scenarios need training")

        # Final checkpoint
        final_checkpoint = self.checkpoint_manager.save_checkpoint(self, self.iterations, 
                                                                  "preflop_final_checkpoint.pkl")
        
        print(f"🏆 Preflop training complete!")
        print(f"📁 Final checkpoint: {final_checkpoint}")
        
        return self
    
    def load_from_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint"""
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Restore CFR state
        self.regret_sum = defaultdict(lambda: np.zeros(3))
        for k, v in checkpoint_data['regret_sum'].items():
            self.regret_sum[k] = np.array(v)
            
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        for k, v in checkpoint_data['strategy_sum'].items():
            self.strategy_sum[k] = np.array(v)
            
        self.info_set_counter = Counter(checkpoint_data['info_set_counter'])
        self.iterations = checkpoint_data['iteration']
        
        # Restore poker data
        self.hand_groups = checkpoint_data['hand_groups']
        self.all_hands = checkpoint_data['all_hands']
        
        return self

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
            pot_size = 2 + history[:-1].count('r')
            folder = (len(history) - 1) % 2
            return pot_size if player != folder else -pot_size
        else:
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


class ResumableTrainingOrchestrator:
    """
    Orchestrates training of both preflop and postflop with checkpointing
    """
    
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def run_full_training_with_checkpoints(self, checkpoint_every=5000):
        """Run complete training with checkpointing"""
        
        print("🚀 RESUMABLE CFR TRAINING WITH CHECKPOINTS")
        print("=" * 60)
        
        # Check for existing checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if checkpoints:
            print("📂 Found existing checkpoints:")
            for cp in checkpoints[-3:]:  # Show last 3
                print(f"   {cp['filename']}: iteration {cp['iteration']}, {cp['size_mb']} MB")
            
            print("\nTo resume from latest checkpoint, use resume_from parameter")
        
        # Train preflop with checkpoints
        print("\n1️⃣ Training Preflop with Checkpoints...")
        preflop_solver = ResumablePreflopCFR(checkpoint_manager=self.checkpoint_manager)
        preflop_solver.train_with_checkpoints(
            total_iterations=300000,
            checkpoint_every=checkpoint_every
        )
        
        # Train postflop (use existing implementation for now)
        print("\n2️⃣ Training Postflop...")
        try:
            from postflop_cfr import run_postflop_training
            postflop_solver = run_postflop_training(preflop_solver)
        except ImportError:
            print("⚠️ Postflop module not available, skipping...")
            postflop_solver = None
        
        print("✅ Complete training with checkpoints finished!")
        
        return preflop_solver, postflop_solver


def run_preflop_training_with_checkpoints():
    """Main function to run preflop training with checkpoints"""
    print("🃏 PREFLOP CFR TRAINING WITH CHECKPOINTS")
    print("=" * 50)
    
    # Initialize trainer
    cfr = ResumablePreflopCFR()
    
    # Train with checkpoints
    cfr.train_with_checkpoints(total_iterations=300000, min_visits_per_scenario=300, checkpoint_every=2500)
    
    # Analyze results
    cfr.analyze_results()
    
    # Test specific hands
    cfr.test_hand_lookup()
    
    print(f"\n✅ Preflop training with checkpoints complete!")
    print(f"✅ All 169 hands covered with strategic grouping")
    print(f"✅ Checkpoints saved for resume capability")
    
    return cfr


def quick_checkpoint_demo():
    """Demonstrate checkpoint functionality"""
    
    print("🎯 CHECKPOINT SYSTEM DEMO")
    print("=" * 40)
    
    # Create checkpoint manager
    cm = CheckpointManager("demo_checkpoints")
    
    # Create simple CFR solver
    cfr = ResumablePreflopCFR(checkpoint_manager=cm)
    
    # Train for a bit
    cfr.train_with_checkpoints(total_iterations=1000, checkpoint_every=250)
    
    # List checkpoints
    print("\n📂 Available checkpoints:")
    checkpoints = cm.list_checkpoints()
    for cp in checkpoints:
        print(f"   {cp['filename']}: iter {cp['iteration']}, {cp['scenarios_trained']} scenarios")
    
    # Demonstrate resume
    if checkpoints:
        print(f"\n🔄 Demonstrating resume from {checkpoints[-1]['filename']}")
        new_cfr = ResumablePreflopCFR(checkpoint_manager=cm)
        new_cfr.load_from_checkpoint(checkpoints[-1]['path'])
        print(f"✅ Resumed at iteration {new_cfr.iterations}")


if __name__ == "__main__":
    # Run checkpoint demo
    # quick_checkpoint_demo()
    
    # Or run full training with checkpoints
    run_preflop_training_with_checkpoints()
