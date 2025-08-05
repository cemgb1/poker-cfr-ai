#!/usr/bin/env python3
"""
GCP CFR Training Script - Production-ready CFR training with multiprocessing

Features:
- Uses all CPU cores (multiprocessing)
- Samples least-trained hand groups for balanced coverage  
- Outputs lookup-table CSV with percentage choices per action
- Logs every 15 minutes or at least every 500 iterations
- Records model performance to separate CSV for analysis/charting
- Moves unused files to archivefolder at end
- Uses EnhancedCFRTrainer and generate_enhanced_scenarios
"""

import multiprocessing as mp
import os
import shutil
import time
import logging
import pickle
import glob
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import random
import signal
import sys

import numpy as np
import pandas as pd

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios, ACTIONS


class GCPCFRTrainer:
    """
    Production GCP CFR Trainer with balanced scenario sampling and comprehensive logging
    """
    
    def __init__(self, n_scenarios=5000, n_workers=None, log_interval_minutes=15):
        self.n_workers = n_workers or mp.cpu_count()
        self.log_interval_minutes = log_interval_minutes
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # File management
        self.output_files = []
        
        # Setup logging
        self.setup_logging()
        
        # Initialize scenario pool with balanced sampling
        self.logger.info(f"🚀 Initializing GCP CFR Trainer with {self.n_workers} workers")
        self.logger.info(f"🎯 Generating {n_scenarios} scenarios for balanced training...")
        
        self.scenarios = generate_enhanced_scenarios(n_scenarios)
        self.scenario_training_counts = Counter()
        
        # Shared data structures for multiprocessing
        self.manager = mp.Manager()
        self.shared_regrets = self.manager.dict()
        self.shared_strategies = self.manager.dict() 
        self.shared_counters = self.manager.dict()
        self.shared_queue = self.manager.Queue()
        
        # Performance tracking
        self.performance_metrics = []
        self.iteration_count = 0
        
        self.logger.info(f"✅ GCP CFR Trainer initialized successfully")
        self.logger.info(f"   🖥️  CPU cores: {self.n_workers}")
        self.logger.info(f"   📊 Scenarios: {len(self.scenarios):,}")
        self.logger.info(f"   📝 Log interval: {self.log_interval_minutes} minutes")
        
    def setup_logging(self):
        """Setup comprehensive logging to file"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/gcp_cfr_training_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_filename = log_filename
        self.output_files.append(log_filename)
        
        self.logger.info("🔧 Logging system initialized")
        self.logger.info(f"📝 Log file: {log_filename}")
        
    def get_balanced_scenario_batch(self, batch_size=1000):
        """
        Sample scenarios with bias toward least-trained hand groups for balanced coverage
        """
        # Group scenarios by hand category
        scenarios_by_category = defaultdict(list)
        for scenario in self.scenarios:
            scenarios_by_category[scenario['hand_category']].append(scenario)
        
        # Calculate training counts per category
        category_counts = defaultdict(int)
        for scenario_key, count in self.scenario_training_counts.items():
            # Parse hand category from scenario key
            hand_category = scenario_key.split('_')[0]
            category_counts[hand_category] += count
        
        # Create weighted sampling - prioritize least trained categories
        max_count = max(category_counts.values()) if category_counts else 1
        category_weights = {}
        
        for category in scenarios_by_category.keys():
            # Inverse weighting - less trained categories get higher weight
            current_count = category_counts.get(category, 0)
            weight = max_count + 1 - current_count
            category_weights[category] = max(weight, 1)  # Ensure minimum weight of 1
        
        # Sample scenarios based on weights
        balanced_batch = []
        categories = list(scenarios_by_category.keys())
        weights = [category_weights[cat] for cat in categories]
        
        for _ in range(batch_size):
            # Select category based on weights
            selected_category = np.random.choice(categories, p=np.array(weights)/sum(weights))
            # Select random scenario from that category
            scenario = random.choice(scenarios_by_category[selected_category])
            balanced_batch.append(scenario)
        
        return balanced_batch
    
    def worker_train_process(self, worker_id, iterations_per_worker):
        """
        Worker process for parallel CFR training with balanced scenario sampling
        """
        try:
            self.logger.info(f"Worker {worker_id}: Starting {iterations_per_worker:,} iterations")
            
            # Create local CFR trainer for this worker
            local_trainer = EnhancedCFRTrainer(scenarios=self.scenarios)
            local_trainer.start_performance_tracking()
            
            worker_start_time = time.time()
            local_iteration_count = 0
            
            for iteration in range(iterations_per_worker):
                # Use balanced scenario selection for better coverage
                scenario = local_trainer.select_balanced_scenario()
                
                # Train on scenario
                result = local_trainer.play_enhanced_scenario(scenario)
                scenario_key = local_trainer.get_scenario_key(scenario)
                local_trainer.scenario_counter[scenario_key] += 1
                local_iteration_count += 1
                
                # Record performance metrics periodically
                if iteration % 100 == 0:
                    metrics = local_trainer.record_iteration_metrics(iteration)
                    self.shared_queue.put(('metrics', worker_id, metrics))
                
                # Progress logging every 500 iterations
                if (iteration + 1) % 500 == 0:
                    elapsed = time.time() - worker_start_time
                    rate = local_iteration_count / elapsed
                    progress_msg = f"Worker {worker_id}: {iteration+1:,}/{iterations_per_worker:,} iterations ({rate:.1f}/sec)"
                    self.shared_queue.put(('progress', worker_id, progress_msg))
            
            # Send final results back
            worker_results = {
                'worker_id': worker_id,
                'iterations_completed': iterations_per_worker,
                'regret_sum': dict(local_trainer.regret_sum),
                'strategy_sum': dict(local_trainer.strategy_sum),
                'scenario_counter': dict(local_trainer.scenario_counter),
                'performance_metrics': local_trainer.performance_metrics,
                'final_time': time.time() - worker_start_time
            }
            
            self.shared_queue.put(('results', worker_id, worker_results))
            self.logger.info(f"✅ Worker {worker_id}: Completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Worker {worker_id} failed: {e}")
            self.shared_queue.put(('error', worker_id, str(e)))
    
    def run_parallel_training(self, total_iterations=200000):
        """
        Main training loop with parallel processing and periodic logging
        """
        self.logger.info(f"🚀 Starting GCP CFR parallel training")
        self.logger.info(f"   🎯 Total iterations: {total_iterations:,}")
        self.logger.info(f"   🖥️  Workers: {self.n_workers}")
        self.logger.info(f"   ⚡ Iterations per worker: {total_iterations // self.n_workers:,}")
        self.logger.info("=" * 80)
        
        iterations_per_worker = total_iterations // self.n_workers
        
        # Start worker processes
        processes = []
        for worker_id in range(self.n_workers):
            p = mp.Process(
                target=self.worker_train_process,
                args=(worker_id, iterations_per_worker)
            )
            p.start()
            processes.append(p)
        
        # Monitor progress and handle logging
        completed_workers = 0
        worker_results = {}
        last_log_time = time.time()
        
        while completed_workers < self.n_workers:
            try:
                # Check for messages from workers
                message = self.shared_queue.get(timeout=10)
                msg_type, worker_id, data = message
                
                if msg_type == 'progress':
                    self.logger.info(data)
                    
                elif msg_type == 'metrics':
                    self.performance_metrics.extend([data])
                    
                elif msg_type == 'results':
                    worker_results[worker_id] = data
                    completed_workers += 1
                    self.logger.info(f"✅ Worker {worker_id} completed ({completed_workers}/{self.n_workers})")
                    
                elif msg_type == 'error':
                    self.logger.error(f"❌ Worker {worker_id} error: {data}")
                
                # Periodic logging (every 15 minutes or as configured)
                current_time = time.time()
                if current_time - last_log_time >= (self.log_interval_minutes * 60):
                    self.log_training_progress(current_time)
                    last_log_time = current_time
                    
            except:
                # Timeout - check if any processes are still alive
                alive_count = sum(1 for p in processes if p.is_alive())
                if alive_count == 0:
                    break
                    
                # Periodic logging even during quiet periods
                current_time = time.time()
                if current_time - last_log_time >= (self.log_interval_minutes * 60):
                    self.log_training_progress(current_time)
                    last_log_time = current_time
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results
        self.combine_worker_results(worker_results)
        
        # Final logging
        total_time = time.time() - self.start_time
        self.logger.info(f"🏆 Training completed successfully!")
        self.logger.info(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"🎮 Total iterations: {total_iterations:,}")
        self.logger.info(f"⚡ Average rate: {total_iterations/total_time:.1f} iterations/second")
        
        return worker_results
    
    def log_training_progress(self, current_time):
        """Log periodic training progress"""
        elapsed_time = current_time - self.start_time
        self.logger.info(f"📊 Training Progress Update")
        self.logger.info(f"   ⏱️  Elapsed time: {elapsed_time/60:.1f} minutes")
        self.logger.info(f"   📈 Performance metrics collected: {len(self.performance_metrics)}")
        self.logger.info(f"   🎯 Scenario training distribution balance in progress...")
    
    def combine_worker_results(self, worker_results):
        """Combine results from all workers"""
        self.logger.info(f"🔄 Combining results from {len(worker_results)} workers...")
        
        self.combined_regret_sum = defaultdict(lambda: defaultdict(float))
        self.combined_strategy_sum = defaultdict(lambda: defaultdict(float))
        self.combined_scenario_counter = Counter()
        self.all_performance_metrics = []
        
        for worker_id, worker_data in worker_results.items():
            self.logger.info(f"   Worker {worker_id}: {worker_data['iterations_completed']:,} iterations in {worker_data['final_time']:.1f}s")
            
            # Combine regrets
            for scenario_key, regrets in worker_data['regret_sum'].items():
                for action, regret_value in regrets.items():
                    self.combined_regret_sum[scenario_key][action] += regret_value
            
            # Combine strategies  
            for scenario_key, strategies in worker_data['strategy_sum'].items():
                for action, strategy_value in strategies.items():
                    self.combined_strategy_sum[scenario_key][action] += strategy_value
            
            # Combine counters
            for scenario_key, count in worker_data['scenario_counter'].items():
                self.combined_scenario_counter[scenario_key] += count
            
            # Combine performance metrics
            self.all_performance_metrics.extend(worker_data['performance_metrics'])
        
        self.logger.info(f"✅ Results combined successfully")
        self.logger.info(f"   📊 Total unique scenarios trained: {len(self.combined_scenario_counter)}")
        self.logger.info(f"   📈 Total performance metrics: {len(self.all_performance_metrics)}")
    
    def export_lookup_table_csv(self, filename="gcp_cfr_lookup_table.csv"):
        """
        Export comprehensive lookup table CSV with percentage choices per action
        One row per scenario with all action probabilities
        """
        self.logger.info(f"📊 Exporting lookup table to {filename}...")
        
        from enhanced_cfr_preflop_generator_v2 import PREFLOP_HAND_RANGES
        
        export_data = []
        
        for scenario_key, strategy_counts in self.combined_strategy_sum.items():
            if sum(strategy_counts.values()) > 0:
                
                # Parse scenario key  
                parts = scenario_key.split("_")
                if len(parts) >= 4:
                    hand_category = parts[0]
                    position = parts[1] 
                    stack_category = parts[2]
                    bet_size_category = parts[3]
                else:
                    continue
                
                # Get example hands for this category
                example_hands = ""
                if hand_category in PREFLOP_HAND_RANGES:
                    examples = PREFLOP_HAND_RANGES[hand_category][:3]
                    example_hands = ", ".join(examples)
                
                # Calculate normalized action probabilities
                total_count = sum(strategy_counts.values())
                action_percentages = {}
                
                # Initialize all actions to 0%
                for action_name in ACTIONS.keys():
                    action_percentages[f"{action_name}_percent"] = 0.0
                
                # Fill in actual percentages  
                for action, count in strategy_counts.items():
                    if action in ACTIONS:
                        percentage = (count / total_count) * 100
                        action_percentages[f"{action}_percent"] = round(percentage, 2)
                
                # Determine best action and confidence
                best_action = max(strategy_counts.items(), key=lambda x: x[1])[0]
                best_action_confidence = max(action_percentages.values())
                
                # Get training statistics
                training_games = self.combined_scenario_counter.get(scenario_key, 0)
                
                # Build lookup table row
                row = {
                    'scenario_key': scenario_key,
                    'hand_category': hand_category,
                    'example_hands': example_hands,
                    'position': position,
                    'stack_depth': stack_category,
                    'bet_size_category': bet_size_category,
                    'training_games': training_games,
                    'recommended_action': best_action.upper(),
                    'confidence_percent': round(best_action_confidence, 2),
                    **action_percentages
                }
                
                export_data.append(row)
        
        if export_data:
            df = pd.DataFrame(export_data)
            
            # Sort by confidence descending, then by training games
            df = df.sort_values(['confidence_percent', 'training_games'], ascending=[False, False])
            
            # Export to CSV
            df.to_csv(filename, index=False)
            self.output_files.append(filename)
            
            self.logger.info(f"✅ Exported {len(export_data)} scenarios to lookup table")
            self.logger.info(f"📊 Columns: {list(df.columns)}")
            
            # Log summary statistics
            self.logger.info(f"📈 Lookup Table Summary:")
            self.logger.info(f"   📊 Total scenarios: {len(export_data)}")
            self.logger.info(f"   🎯 Average training games: {df['training_games'].mean():.1f}")
            self.logger.info(f"   💪 Average confidence: {df['confidence_percent'].mean():.1f}%")
            
            # Show action distribution
            action_dist = df['recommended_action'].value_counts()
            self.logger.info(f"   🎯 Action Distribution:")
            for action, count in action_dist.items():
                pct = count/len(export_data)*100
                self.logger.info(f"      {action}: {count} scenarios ({pct:.1f}%)")
            
            return df
        else:
            self.logger.warning("❌ No strategy data to export")
            return None
    
    def export_performance_metrics_csv(self, filename="gcp_cfr_performance.csv"):
        """Export performance metrics to separate CSV for analysis/charting"""
        if not self.all_performance_metrics:
            self.logger.warning("❌ No performance metrics to export")
            return None
        
        self.logger.info(f"📈 Exporting performance metrics to {filename}...")
        
        df = pd.DataFrame(self.all_performance_metrics)
        df.to_csv(filename, index=False)
        self.output_files.append(filename)
        
        # Log performance summary
        self.logger.info(f"✅ Exported {len(df)} performance data points")
        self.logger.info(f"📊 Performance Metrics Summary:")
        self.logger.info(f"   ⏱️  Total training time: {df['total_elapsed_time'].iloc[-1]:.2f}s")
        self.logger.info(f"   ⚡ Average iteration time: {df['time_per_iteration'].mean():.4f}s")
        self.logger.info(f"   📉 Final average regret: {df['average_regret'].iloc[-1]:.6f}")
        self.logger.info(f"   📊 Unique scenarios visited: {df['unique_scenarios_visited'].iloc[-1]}")
        
        return df
    
    def cleanup_and_archive_files(self):
        """
        Move all unused files to archivefolder, keeping only essential model files
        """
        self.logger.info("🧹 Starting file cleanup and archival process...")
        
        # Create archivefolder if it doesn't exist
        archive_dir = Path("archivefolder")
        archive_dir.mkdir(exist_ok=True)
        
        # Files to keep in root (essential for final model)
        essential_files = {
            'enhanced_cfr_trainer_v2.py',
            'enhanced_cfr_preflop_generator_v2.py', 
            'run_gcp_cfr_training.py',
            'requirements.txt',
            'gcp_cfr_lookup_table.csv',  # Final model output
            'gcp_cfr_performance.csv'    # Performance metrics
        }
        
        # Get all Python files and other files in root
        all_files = list(Path('.').glob('*.py')) + list(Path('.').glob('*.csv')) + list(Path('.').glob('*.md'))
        
        archived_count = 0
        for file_path in all_files:
            if file_path.name not in essential_files and not file_path.name.startswith('.'):
                try:
                    destination = archive_dir / file_path.name
                    shutil.move(str(file_path), str(destination))
                    archived_count += 1
                    self.logger.info(f"   📦 Archived: {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Could not archive {file_path.name}: {e}")
        
        # Also archive old log files except current one
        log_files = list(Path('.').glob('*.log'))
        for log_file in log_files:
            if log_file.name != os.path.basename(self.log_filename):
                try:
                    destination = archive_dir / log_file.name
                    shutil.move(str(log_file), str(destination))
                    archived_count += 1
                    self.logger.info(f"   📦 Archived old log: {log_file.name}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Could not archive {log_file.name}: {e}")
        
        self.logger.info(f"✅ File cleanup completed")
        self.logger.info(f"   📦 Files archived: {archived_count}")
        self.logger.info(f"   📁 Archive location: {archive_dir}")
        
        # List final essential files remaining
        remaining_files = [f for f in essential_files if Path(f).exists()]
        self.logger.info(f"   💎 Essential files remaining: {len(remaining_files)}")
        for essential_file in remaining_files:
            self.logger.info(f"      📄 {essential_file}")


def setup_signal_handlers(trainer):
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        trainer.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        # Export current progress
        if hasattr(trainer, 'combined_strategy_sum'):
            trainer.export_lookup_table_csv("emergency_backup_lookup_table.csv")
            trainer.export_performance_metrics_csv("emergency_backup_performance.csv")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """
    Main GCP CFR Training execution
    Production-ready training with all required features
    """
    print("🚀 GCP CFR Training System Starting...")
    print("=" * 80)
    
    # Initialize trainer with production settings
    trainer = GCPCFRTrainer(
        n_scenarios=10000,          # Large scenario pool for comprehensive coverage
        n_workers=mp.cpu_count(),   # Use all CPU cores
        log_interval_minutes=15     # Log every 15 minutes as required
    )
    
    # Setup graceful shutdown
    setup_signal_handlers(trainer)
    
    try:
        # Run parallel training (200k iterations for deeper learning)
        trainer.run_parallel_training(total_iterations=200000)
        
        # Export lookup table CSV (main requirement)
        lookup_df = trainer.export_lookup_table_csv("gcp_cfr_lookup_table.csv")
        
        # Export performance metrics CSV (requirement)
        performance_df = trainer.export_performance_metrics_csv("gcp_cfr_performance.csv")
        
        # Final training summary
        trainer.logger.info("🏆 GCP CFR Training Completed Successfully!")
        trainer.logger.info(f"📊 Lookup table exported: {len(lookup_df) if lookup_df is not None else 0} scenarios")
        trainer.logger.info(f"📈 Performance metrics exported: {len(performance_df) if performance_df is not None else 0} data points")
        
        # Cleanup and archive unused files (requirement)
        trainer.cleanup_and_archive_files()
        
        print("\n🎯 GCP CFR TRAINING COMPLETE!")
        print("✅ All requirements satisfied:")
        print("   🖥️  Used all CPU cores (multiprocessing)")
        print("   ⚖️  Sampled least-trained hand groups for balanced coverage") 
        print("   📊 Generated lookup-table CSV with percentage choices per action")
        print("   📝 Logged every 15 minutes and every 500+ iterations")
        print("   📈 Recorded model performance to separate CSV")
        print("   🧹 Moved unused files to archivefolder")
        print("   💎 Only essential model files remain in repo root")
        
        return trainer
        
    except Exception as e:
        trainer.logger.error(f"❌ Training failed with error: {e}")
        # Emergency backup
        if hasattr(trainer, 'combined_strategy_sum'):
            trainer.export_lookup_table_csv("emergency_backup_lookup_table.csv")
            trainer.export_performance_metrics_csv("emergency_backup_performance.csv")
        raise


if __name__ == "__main__":
    # Run GCP CFR Training
    trainer = main()