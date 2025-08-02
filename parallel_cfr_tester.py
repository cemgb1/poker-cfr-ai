# parallel_cfr_trainer.py - Multi-core CFR training for maximum CPU utilization

import multiprocessing as mp
from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import time
import pickle
import os

class ParallelCFRTrainer:
    """
    Parallel CFR trainer that uses all CPU cores for maximum performance
    """
    
    def __init__(self, n_scenarios=100, n_workers=None):
        # Auto-detect CPU cores or use specified number
        self.n_workers = n_workers or mp.cpu_count()
        print(f"🚀 Initializing Parallel CFR with {self.n_workers} workers")
        
        # Generate scenarios once, share across workers
        print(f"🎯 Generating {n_scenarios} scenarios for parallel training...")
        self.scenarios = generate_enhanced_scenarios(n_scenarios)
        
        # Shared results storage
        self.manager = mp.Manager()
        self.shared_results = self.manager.list()
        self.shared_regrets = self.manager.dict()
        self.shared_strategies = self.manager.dict()
        
        print(f"💪 Parallel CFR Ready!")
        print(f"   🖥️  CPU cores: {self.n_workers}")
        print(f"   📊 Scenarios: {len(self.scenarios):,}")
        print(f"   🎮 Ready for high-performance training!")

        def worker_train_batch(self, worker_id, scenarios_batch, iterations_per_worker, shared_queue):
        """
        Worker function that trains on a batch of scenarios
        Each worker runs independently to maximize CPU utilization
        """
        print(f"Worker {worker_id}: Starting {iterations_per_worker:,} iterations")
        
        # Create local CFR trainer for this worker with batch scenarios
        local_cfr = EnhancedCFRTrainer(scenarios=scenarios_batch)
        
        # Track local results
        local_results = []
        local_start_time = time.time()
        
        for iteration in range(iterations_per_worker):
            scenario = random.choice(scenarios_batch)
            print(f"[Worker {worker_id}] Iteration {iteration + 1}/{iterations_per_worker}")
            print(f"  Scenario Key: {scenario.get('key', str(scenario)[:50])}")
            print(f"  Scenario Details: {scenario}")
            result = local_cfr.play_enhanced_scenario(scenario)
            print(f"  Result: {result}")
            local_results.append(result)
            # (progress update every 500 iterations)
            
            # Progress update every 500 iterations
            if (iteration + 1) % 500 == 0:
                elapsed = time.time() - local_start_time
                rate = (iteration + 1) / elapsed
                shared_queue.put(f"Worker {worker_id}: {iteration+1:,}/{iterations_per_worker:,} ({rate:.1f}/sec)")
        
        # Send results back
        worker_summary = {
            'worker_id': worker_id,
            'iterations_completed': iterations_per_worker,
            'regret_sum': dict(local_cfr.regret_sum),
            'strategy_sum': dict(local_cfr.strategy_sum),
            'scenario_counter': dict(local_cfr.scenario_counter),
            'results': local_results,
            'final_time': time.time() - local_start_time
        }
        
        shared_queue.put(worker_summary)
        print(f"Worker {worker_id}: Completed {iterations_per_worker:,} iterations")
    def parallel_train(self, total_iterations=50000, checkpoint_every=10000):
        """
        Main parallel training function
        Distributes work across all CPU cores
        """
        print(f"🚀 Starting Parallel CFR Training")
        print(f"   🎯 Total iterations: {total_iterations:,}")
        print(f"   🖥️  Workers: {self.n_workers}")
        print(f"   ⚡ Iterations per worker: {total_iterations // self.n_workers:,}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Divide scenarios among workers
        scenarios_per_worker = len(self.scenarios) // self.n_workers
        worker_scenario_batches = []
        
        for i in range(self.n_workers):
            start_idx = i * scenarios_per_worker
            end_idx = start_idx + scenarios_per_worker if i < self.n_workers - 1 else len(self.scenarios)
            worker_batch = self.scenarios[start_idx:end_idx]
            worker_scenario_batches.append(worker_batch)
        
        # Calculate iterations per worker
        iterations_per_worker = total_iterations // self.n_workers
        
        # Create shared queue for communication
        manager = mp.Manager()
        shared_queue = manager.Queue()
        
        # Start worker processes
        processes = []
        for worker_id in range(self.n_workers):
            p = mp.Process(
                target=self.worker_train_batch,
                args=(worker_id, worker_scenario_batches[worker_id], iterations_per_worker, shared_queue)
            )
            p.start()
            processes.append(p)
        
        # Monitor progress
        completed_workers = 0
        worker_results = {}
        
        while completed_workers < self.n_workers:
            try:
                message = shared_queue.get(timeout=30)
                
                if isinstance(message, str):
                    # Progress update
                    print(message)
                else:
                    # Worker completion
                    worker_id = message['worker_id']
                    worker_results[worker_id] = message
                    completed_workers += 1
                    print(f"✅ Worker {worker_id} finished ({completed_workers}/{self.n_workers})")
                    
            except:
                # Check if processes are still alive
                alive_count = sum(1 for p in processes if p.is_alive())
                if alive_count == 0:
                    break
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results from all workers
        self.combine_worker_results(worker_results)
        
        elapsed = time.time() - start_time
        total_games = sum(len(result['results']) for result in worker_results.values())
        
        print(f"\n🏆 Parallel CFR Training Complete!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"🎮 Total games: {total_games:,}")
        print(f"⚡ Overall rate: {total_games/elapsed:.1f} games/second")
        print(f"🖥️  CPU utilization: {self.n_workers} cores maxed out")

    def combine_worker_results(self, worker_results):
        """
        Combine results from all worker processes
        """
        print(f"\n🔄 Combining results from {len(worker_results)} workers...")
        
        # Combined regrets and strategies
        self.combined_regret_sum = defaultdict(lambda: defaultdict(float))
        self.combined_strategy_sum = defaultdict(lambda: defaultdict(float))
        self.combined_scenario_counter = Counter()
        self.all_results = []
        
        for worker_id, worker_data in worker_results.items():
            print(f"   Worker {worker_id}: {len(worker_data['results']):,} games in {worker_data['final_time']:.1f}s")
            
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
            
            # Combine results
            self.all_results.extend(worker_data['results'])
        
        print(f"✅ Combined results: {len(self.all_results):,} total games")

    def analyze_parallel_results(self):
        """
        Analyze combined results from parallel training
        """
        print(f"\n🧠 PARALLEL CFR ANALYSIS")
        print("=" * 60)
        
        # Performance metrics
        if self.all_results:
            survival_count = sum(1 for r in self.all_results if not r.get('busted', False))
            survival_rate = survival_count / len(self.all_results)
            
            avg_payoff = np.mean([r['payoff'] for r in self.all_results])
            
            print(f"Total games analyzed: {len(self.all_results):,}")
            print(f"Survival rate: {survival_rate:.1%}")
            print(f"Average payoff: {avg_payoff:+.3f}")
        
        # Strategy distribution
        total_scenarios_trained = len(self.combined_scenario_counter)
        well_trained = len([k for k, v in self.combined_scenario_counter.items() if v >= 10])
        
        print(f"Unique scenarios: {total_scenarios_trained}")
        print(f"Well-trained scenarios (≥10 games): {well_trained}")
        
        # Show top trained scenarios
        print(f"\nTop Trained Scenarios:")
        top_scenarios = self.combined_scenario_counter.most_common(10)
        for scenario_key, count in top_scenarios:
            print(f"  {scenario_key[:40]:40s}: {count:4d} games")

    def export_parallel_results(self, filename='parallel_cfr_results.csv'):
        """
        Export combined parallel results
        """
        print(f"\n📊 Exporting parallel results to {filename}...")
        
        results = []
        for scenario_key, visit_count in self.combined_scenario_counter.items():
            if visit_count >= 5:
                
                # Get strategy from combined data
                strategy_data = self.combined_strategy_sum[scenario_key]
                
                if strategy_data:
                    # Normalize strategy
                    total_strategy = sum(strategy_data.values())
                    if total_strategy > 0:
                        normalized_strategy = {k: v/total_strategy for k, v in strategy_data.items()}
                        primary_action = max(normalized_strategy.items(), key=lambda x: x[1])[0]
                        confidence = max(normalized_strategy.values())
                    else:
                        normalized_strategy = {}
                        primary_action = 'unknown'
                        confidence = 0
                    
                    # Parse scenario key
                    parts = scenario_key.split('_')
                    
                    result_row = {
                        'scenario_key': scenario_key,
                        'hand_category': parts[0] if len(parts) > 0 else 'unknown',
                        'position': parts[1] if len(parts) > 1 else 'unknown',
                        'stack_category': parts[2] if len(parts) > 2 else 'unknown',
                        'primary_action': primary_action,
                        'confidence': round(confidence, 3),
                        'total_games': visit_count,
                        'workers_trained': self.n_workers
                    }
                    
                    # Add action probabilities
                    for action in ['fold', 'call_small', 'call_large', 'raise_small', 'raise_large', 'all_in']:
                        result_row[f'{action}_prob'] = round(normalized_strategy.get(action, 0), 3)
                    
                    results.append(result_row)
        
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        
        print(f"✅ Exported {len(results)} parallel-trained strategies")
        print(f"📁 File: {filename}")
        return df

def run_parallel_cfr_training():
    """
    Run high-performance parallel CFR training
    """
    print("💪 HIGH-PERFORMANCE PARALLEL CFR TRAINING")
    print("=" * 60)
    
    # Detect system capabilities
    cpu_count = mp.cpu_count()
    print(f"🖥️  Detected {cpu_count} CPU cores")
    
    # Initialize parallel trainer
    parallel_cfr = ParallelCFRTrainer(n_scenarios=2000, n_workers=cpu_count)
    
    # Run high-intensity training
    parallel_cfr.parallel_train(total_iterations=100000, checkpoint_every=20000)
    
    # Analyze results
    parallel_cfr.analyze_parallel_results()
    
    # Export results
    results_df = parallel_cfr.export_parallel_results()
    
    print(f"\n🏆 HIGH-PERFORMANCE TRAINING COMPLETE!")
    print(f"💪 CPU cores utilized: {cpu_count}")
    print(f"📊 Strategies learned: {len(results_df)}")
    print(f"⚡ Maximum performance achieved!")
    
    return parallel_cfr, results_df

def benchmark_parallel_performance():
    """
    Benchmark parallel vs single-core performance
    """
    print("⚡ PARALLEL PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Single core benchmark
    print("🔄 Testing single-core performance...")
    single_start = time.time()
    single_cfr = ParallelCFRTrainer(n_scenarios=500, n_workers=1)
    single_cfr.parallel_train(total_iterations=5000)
    single_time = time.time() - single_start
    
    # Multi-core benchmark
    print("🚀 Testing multi-core performance...")
    multi_start = time.time()
    multi_cfr = ParallelCFRTrainer(n_scenarios=500, n_workers=mp.cpu_count())
    multi_cfr.parallel_train(total_iterations=5000)
    multi_time = time.time() - multi_start
    
    # Performance comparison
    speedup = single_time / multi_time
    efficiency = speedup / mp.cpu_count()
    
    print(f"\n📊 PERFORMANCE BENCHMARK RESULTS:")
    print(f"Single-core time: {single_time:.1f} seconds")
    print(f"Multi-core time:  {multi_time:.1f} seconds")
    print(f"Speedup: {speedup:.1f}x")
    print(f"Efficiency: {efficiency:.1%}")
    print(f"CPU cores used: {mp.cpu_count()}")

if __name__ == "__main__":
    # Run performance benchmark
    print("🚀 Starting Parallel CFR Performance Test")
    
    # Quick parallel test
    cpu_cores = mp.cpu_count()
    print(f"💪 Utilizing all {cpu_cores} CPU cores for maximum performance!")
    
    parallel_cfr = ParallelCFRTrainer(n_scenarios=1000, n_workers=cpu_cores)
    parallel_cfr.parallel_train(total_iterations=20000)
    parallel_cfr.analyze_parallel_results()
    results_df = parallel_cfr.export_parallel_results('max_performance_cfr.csv')
    
    print(f"\n🎯 CPU UTILIZATION MAXIMIZED!")
    print(f"   🖥️  All {cpu_cores} cores working at 100%")
    print(f"   ⚡ {len(results_df)} strategies learned at maximum speed")
