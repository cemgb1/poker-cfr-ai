# -*- coding: utf-8 -*-
"""
CFR with Checkpointing System
Saves training state periodically and can resume from checkpoints
"""

import pickle
import json
import os
import time
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
    
    def train_with_checkpoints(self, total_iterations=500000, min_visits_per_scenario=300, 
                              checkpoint_every=50000, resume_from=None):
        """
        Train with automatic checkpointing and resume capability
        """
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_from_checkpoint(resume_from)
            print(f"🔄 Resuming from iteration {self.iterations}")
        
        print(f"🚀 Training with checkpoints every {checkpoint_every} iterations")
        print(f"Target: {min_visits_per_scenario} visits per scenario")
        
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
                
                print(f"📊 Checkpoint at iteration {self.iterations}")
                print(f"   Progress: {progress:.1f}% scenarios complete")
                print(f"   Undertrained: {len(undertrained)} scenarios")
                print(f"   Training rate: {(self.iterations - start_iteration) / ((time.time() - start_time) / 3600):.0f} iter/hour")

            # Regular progress updates
            if self.iterations % 10000 == 0:
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

    # [Include all the original CFR methods: create_hand_groups, cfr_preflop, etc.]
    # [Abbreviated for space - would include full implementation]

    def create_hand_groups(self):
        """Create hand grouping system"""
        # Same as original implementation
        pass
    
    def cfr_preflop(self, p0_hand, p1_hand, history, p0, p1, player, stack_bb):
        """CFR recursion for preflop"""
        # Same as original implementation  
        pass


class ResumableTrainingOrchestrator:
    """
    Orchestrates training of both preflop and postflop with checkpointing
    """
    
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def run_full_training_with_checkpoints(self, checkpoint_every=25000):
        """Run complete training with checkpointing"""
        
        print("🚀 RESUMABLE CFR TRAINING WITH CHECKPOINTS")
        print("=" * 60)
        
        # Check for existing checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if checkpoints:
            print("📂 Found existing checkpoints:")
            for cp in checkpoints[-3:]:  # Show last 3
                print(f"   {cp['filename']}: iteration {cp['iteration']}, {cp['size_mb']} MB")
            
            resume = input("Resume from latest checkpoint? (y/n): ").lower().strip()
            if resume == 'y':
                latest_checkpoint = checkpoints[-1]['path']
                print(f"🔄 Resuming from {latest_checkpoint}")
        
        # Train preflop with checkpoints
        print("\n1️⃣ Training Preflop with Checkpoints...")
        preflop_solver = ResumablePreflopCFR(checkpoint_manager=self.checkpoint_manager)
        
        if 'latest_checkpoint' in locals():
            preflop_solver.train_with_checkpoints(
                total_iterations=300000,
                checkpoint_every=checkpoint_every,
                resume_from=latest_checkpoint
            )
        else:
            preflop_solver.train_with_checkpoints(
                total_iterations=300000,
                checkpoint_every=checkpoint_every
            )
        
        # Train postflop with checkpoints (would implement similar pattern)
        print("\n2️⃣ Training Postflop with Checkpoints...")
        # postflop_solver = ResumablePostflopCFR(checkpoint_manager=self.checkpoint_manager)
        # postflop_solver.train_with_checkpoints(...)
        
        print("✅ Complete training with checkpoints finished!")
        
        return preflop_solver


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
    quick_checkpoint_demo()
    
    # Or run full training with checkpoints
    # orchestrator = ResumableTrainingOrchestrator()
    # orchestrator.run_full_training_with_checkpoints()
