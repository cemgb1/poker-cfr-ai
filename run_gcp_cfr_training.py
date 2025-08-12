#!/usr/bin/env python3
"""
GCP CFR Training Script - Production-ready CFR training with multiprocessing

Features:
- Uses all CPU cores (multiprocessing)
- Default: 40,000 iterations per worker (total = n_workers * 40,000)
- Samples least-trained hand groups for balanced coverage  
- Outputs lookup-table CSV with percentage choices per action
- Logs every 15 minutes or at least every 500 iterations
- Records model performance to separate CSV for analysis/charting
- NEW: Unified scenario lookup table (scenario_lookup_table.csv) updated at every log interval
- NEW: Real-time monitoring of learning progress and scenario coverage
- Moves unused files to archivefolder at end
- Uses EnhancedCFRTrainer and generate_enhanced_scenarios
- Default tournament penalty: 0.2 (encourages more risk-taking)
"""

import multiprocessing as mp
import os
import shutil
import time
import logging
import pickle
import glob
import psutil  # For memory monitoring
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
    
    def __init__(self, n_workers=None, log_interval_minutes=15):
        self.n_workers = n_workers or mp.cpu_count()
        self.log_interval_minutes = log_interval_minutes
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # File management
        self.output_files = []
        
        # Checkpointing configuration
        self.checkpoints_dir = Path("checkpoints")
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.checkpoint_interval_minutes = log_interval_minutes  # Same as log interval (15 minutes)
        self.last_checkpoint_time = self.start_time
        
        # Setup logging
        self.setup_logging()
        
        # Initialize scenario pool with all possible combinations
        self.logger.info(f"🚀 Initializing GCP CFR Trainer with {self.n_workers} workers")
        self.logger.info(f"🎯 Generating all possible scenario combinations...")
        
        self.scenarios = generate_enhanced_scenarios()
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
        
        # Shutdown management for graceful termination
        self.shutdown_requested = False
        self.current_worker_results = {}  # Store current results for emergency backup
        
        self.logger.info(f"✅ GCP CFR Trainer initialized successfully")
        self.logger.info(f"   🖥️  CPU cores: {self.n_workers}")
        self.logger.info(f"   📊 Scenarios: {len(self.scenarios):,}")
        self.logger.info(f"   📝 Log interval: {self.log_interval_minutes} minutes")
        self.logger.info(f"   💾 Checkpoints directory: {self.checkpoints_dir}")
        
        # Check for existing checkpoints and offer to resume
        self.resumed_from_checkpoint = self.prompt_checkpoint_resume()
        
    def setup_logging(self):
        """Setup comprehensive logging to file and dedicated error logging"""
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
        
        # Setup dedicated error logging
        self.setup_error_logging()
        
        self.logger.info("🔧 Logging system initialized")
        self.logger.info(f"📝 Log file: {log_filename}")
        self.logger.info(f"🚨 Error log file: {self.error_log_filename}")
    
    def setup_error_logging(self):
        """Setup dedicated error logging to errors.log file"""
        # Error log file path
        self.error_log_filename = "logs/errors.log"
        self.output_files.append(self.error_log_filename)
        
        # Create error logger with plain text format
        self.error_logger = logging.getLogger('error_logger')
        self.error_logger.setLevel(logging.ERROR)
        
        # Remove any existing handlers to avoid duplicates
        for handler in self.error_logger.handlers[:]:
            self.error_logger.removeHandler(handler)
        
        # Create error file handler with plain text format
        error_handler = logging.FileHandler(self.error_log_filename, mode='a')
        error_formatter = logging.Formatter('%(message)s')  # Plain text format
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Prevent propagation to avoid duplicate logging
        self.error_logger.propagate = False
        
        # Setup sys.excepthook for uncaught exceptions
        self.setup_excepthook()
    
    def setup_excepthook(self):
        """Setup sys.excepthook to log uncaught exceptions to errors.log"""
        def excepthook(exc_type, exc_value, exc_traceback):
            # Don't log KeyboardInterrupt as an error
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            # Format the error for errors.log
            import traceback
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_entry = (
                f"TIMESTAMP: {timestamp}\n"
                f"WORKER_ID: main_process\n"
                f"ITERATION: N/A\n"
                f"ERROR_TYPE: {exc_type.__name__}\n"
                f"MESSAGE: {str(exc_value)}\n"
                f"SCENARIO_CONTEXT: N/A\n"
                f"TRACEBACK:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}\n"
                f"{'='*80}\n"
            )
            self.error_logger.error(error_entry)
            
            # Call the default handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = excepthook
    
    def log_error_to_file(self, error_type, message, worker_id=None, iteration=None, 
                         scenario_context=None, traceback_str=None, exit_code=None):
        """
        Log error details to dedicated errors.log file in plain text format.
        
        Args:
            error_type: Type of error (e.g., 'KeyError', 'WorkerExitError')
            message: Error message
            worker_id: Worker ID if applicable
            iteration: Current iteration if applicable
            scenario_context: Scenario context if applicable
            traceback_str: Full traceback string if applicable
            exit_code: Process exit code if applicable
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        error_entry = (
            f"TIMESTAMP: {timestamp}\n"
            f"WORKER_ID: {worker_id or 'N/A'}\n"
            f"ITERATION: {iteration or 'N/A'}\n"
            f"ERROR_TYPE: {error_type}\n"
            f"MESSAGE: {message}\n"
            f"SCENARIO_CONTEXT: {scenario_context or 'N/A'}\n"
        )
        
        if exit_code is not None:
            error_entry += f"EXIT_CODE: {exit_code}\n"
        
        if traceback_str:
            error_entry += f"TRACEBACK:\n{traceback_str}\n"
        else:
            error_entry += f"TRACEBACK: N/A\n"
        
        error_entry += f"{'='*80}\n"
        
        self.error_logger.error(error_entry)
        
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
        Worker process for parallel CFR training with balanced scenario sampling.
        Enhanced with robust error handling and detailed exception logging.
        """
        worker_start_time = time.time()
        local_iteration_count = 0
        current_scenario = None
        current_scenario_key = None
        
        try:
            self.logger.info(f"Worker {worker_id}: Starting {iterations_per_worker:,} iterations")
            
            # Create local CFR trainer for this worker with reduced tournament penalty
            # tournament_survival_penalty=0.2 encourages more risk-taking vs default 0.6
            local_trainer = EnhancedCFRTrainer(scenarios=self.scenarios, tournament_survival_penalty=0.2)
            local_trainer.start_performance_tracking()
            
            for iteration in range(iterations_per_worker):
                try:
                    # Use balanced scenario selection for better coverage
                    current_scenario = local_trainer.select_balanced_scenario()
                    current_scenario_key = local_trainer.get_scenario_key(current_scenario)
                    
                    # Validate scenario before processing
                    if not self._validate_scenario(current_scenario, worker_id):
                        continue
                    
                    # Train on scenario with action validation
                    result = local_trainer.play_enhanced_scenario(current_scenario)
                    
                    # Validate result before updating counters
                    if not self._validate_training_result(result, current_scenario, worker_id):
                        continue
                    
                    local_trainer.scenario_counter[current_scenario_key] += 1
                    local_iteration_count += 1
                    
                    # Record performance metrics periodically
                    if iteration % 100 == 0:
                        try:
                            metrics = local_trainer.record_iteration_metrics(iteration)
                            self.shared_queue.put(('metrics', worker_id, metrics))
                        except Exception as metrics_error:
                            # Log metrics error but don't fail the worker
                            self.logger.warning(f"Worker {worker_id}: Metrics recording failed at iteration {iteration}: {metrics_error}")
                    
                    # Progress logging every 500 iterations
                    if (iteration + 1) % 500 == 0:
                        elapsed = time.time() - worker_start_time
                        rate = local_iteration_count / elapsed
                        progress_msg = f"Worker {worker_id}: {iteration+1:,}/{iterations_per_worker:,} iterations ({rate:.1f}/sec)"
                        try:
                            self.shared_queue.put(('progress', worker_id, progress_msg))
                        except Exception as queue_error:
                            # Log queue error but continue training
                            self.logger.warning(f"Worker {worker_id}: Progress reporting failed: {queue_error}")
                
                except Exception as iteration_error:
                    # Handle individual iteration errors - log details and continue
                    import traceback
                    traceback_str = traceback.format_exc()
                    
                    error_details = {
                        'worker_id': worker_id,
                        'iteration': iteration,
                        'scenario_key': current_scenario_key,
                        'scenario': current_scenario,
                        'error_type': type(iteration_error).__name__,
                        'error_message': str(iteration_error),
                        'traceback': traceback_str
                    }
                    
                    # Log to standard log
                    self.logger.error(f"❌ Worker {worker_id}: Iteration {iteration} failed with {type(iteration_error).__name__}: {iteration_error}")
                    self.logger.error(f"   📊 Scenario key: {current_scenario_key}")
                    self.logger.error(f"   🎯 Scenario details: {current_scenario}")
                    self.logger.error(f"   🔍 Full traceback:\n{traceback_str}")
                    
                    # Log to dedicated errors.log file
                    scenario_context = f"Key: {current_scenario_key}, Details: {current_scenario}"
                    self.log_error_to_file(
                        error_type=type(iteration_error).__name__,
                        message=str(iteration_error),
                        worker_id=worker_id,
                        iteration=iteration,
                        scenario_context=scenario_context,
                        traceback_str=traceback_str
                    )
                    
                    # Try to report the error (but don't fail if queue is broken)
                    try:
                        self.shared_queue.put(('iteration_error', worker_id, error_details))
                    except:
                        pass  # Queue might be broken, continue anyway
                    
                    # Continue with next iteration rather than failing entire worker
                    continue
            
            # Send final results back
            worker_results = {
                'worker_id': worker_id,
                'iterations_completed': local_iteration_count,  # Use actual count, not target
                'iterations_attempted': iterations_per_worker,
                'regret_sum': dict(local_trainer.regret_sum),
                'strategy_sum': dict(local_trainer.strategy_sum),
                'scenario_counter': dict(local_trainer.scenario_counter),
                'performance_metrics': local_trainer.performance_metrics,
                'final_time': time.time() - worker_start_time
            }
            
            self.shared_queue.put(('results', worker_id, worker_results))
            self.logger.info(f"✅ Worker {worker_id}: Completed successfully ({local_iteration_count}/{iterations_per_worker} iterations)")
            
        except Exception as worker_error:
            # Handle critical worker-level errors with full details
            import traceback
            traceback_str = traceback.format_exc()
            elapsed_time = time.time() - worker_start_time
            
            error_details = {
                'worker_id': worker_id,
                'iterations_completed': local_iteration_count,
                'current_scenario_key': current_scenario_key,
                'current_scenario': current_scenario,
                'error_type': type(worker_error).__name__,
                'error_message': str(worker_error),
                'traceback': traceback_str,
                'elapsed_time': elapsed_time
            }
            
            # Log to standard log
            self.logger.error(f"❌ Worker {worker_id}: CRITICAL FAILURE after {local_iteration_count} iterations")
            self.logger.error(f"   💥 Error type: {type(worker_error).__name__}")
            self.logger.error(f"   📝 Error message: {worker_error}")
            self.logger.error(f"   📊 Last scenario key: {current_scenario_key}")
            self.logger.error(f"   🎯 Last scenario: {current_scenario}")
            self.logger.error(f"   ⏱️  Elapsed time: {elapsed_time:.1f}s")
            self.logger.error(f"   🔍 Full traceback:\n{traceback_str}")
            
            # Log to dedicated errors.log file
            scenario_context = f"Key: {current_scenario_key}, Details: {current_scenario}"
            message = f"Worker critical failure after {local_iteration_count} iterations: {worker_error}"
            self.log_error_to_file(
                error_type="WorkerCriticalError",
                message=message,
                worker_id=worker_id,
                iteration=local_iteration_count,
                scenario_context=scenario_context,
                traceback_str=traceback_str
            )
            
            # Try to send error details (but worker will exit anyway)
            try:
                self.shared_queue.put(('worker_critical_error', worker_id, error_details))
            except:
                pass  # Queue might be broken, nothing we can do
    
    def _validate_scenario(self, scenario, worker_id):
        """
        Validate scenario has all required fields and values.
        Enhanced validation to prevent KeyError issues with 'fold' and other actions.
        
        Args:
            scenario: Dictionary containing scenario details
            worker_id: ID of the worker for logging context
            
        Returns:
            bool: True if valid, False if should skip this scenario
        """
        try:
            # Check required scenario fields to prevent KeyError exceptions
            required_fields = ['hand_category', 'hero_position', 'stack_category', 'blinds_level']
            for field in required_fields:
                if field not in scenario:
                    self.logger.warning(f"Worker {worker_id}: Invalid scenario missing field '{field}': {scenario}")
                    return False
            
            # Validate actions are available - specifically check for 'fold' action
            # This addresses the KeyError: 'fold' issue mentioned in the problem statement
            from enhanced_cfr_preflop_generator_v2 import ACTIONS
            if not ACTIONS or 'fold' not in ACTIONS:
                self.logger.error(f"Worker {worker_id}: ACTIONS dictionary missing or invalid - 'fold' not found")
                return False
            
            return True
            
        except Exception as validation_error:
            self.logger.error(f"Worker {worker_id}: Scenario validation failed: {validation_error}")
            return False
    
    def _validate_training_result(self, result, scenario, worker_id):
        """
        Validate training result has expected structure.
        Prevents errors from malformed results that could cause KeyError exceptions.
        
        Args:
            result: Training result dictionary
            scenario: Original scenario for context
            worker_id: ID of the worker for logging context
            
        Returns:
            bool: True if valid, False if should skip this result
        """
        try:
            if not isinstance(result, dict):
                self.logger.warning(f"Worker {worker_id}: Invalid result type {type(result)} for scenario {scenario}")
                return False
            
            # Check required result fields to prevent KeyError during processing
            required_result_fields = ['scenario_key', 'hero_action', 'villain_action', 'payoff']
            for field in required_result_fields:
                if field not in result:
                    self.logger.warning(f"Worker {worker_id}: Result missing field '{field}': {result}")
                    return False
            
            # Validate hero action is a known action (specifically check fold is recognized)
            from enhanced_cfr_preflop_generator_v2 import ACTIONS
            if result['hero_action'] not in ACTIONS:
                self.logger.warning(f"Worker {worker_id}: Unknown hero action '{result['hero_action']}' in result")
                return False
            
            return True
            
        except Exception as validation_error:
            self.logger.error(f"Worker {worker_id}: Result validation failed: {validation_error}")
            return False
    
    def run_sequential_training(self, iterations_per_scenario=1000, 
                              stopping_condition_window=100, regret_stability_threshold=0.01):
        """
        Run sequential training approach - processes all scenarios in order
        until each meets its stopping condition
        """
        self.logger.info(f"🎯 Starting GCP Sequential CFR Training")
        self.logger.info(f"   📊 Total scenarios: {len(self.scenarios):,}")
        self.logger.info(f"   🔄 Iterations per scenario: {iterations_per_scenario:,}")
        self.logger.info(f"   🛑 Stopping window: {stopping_condition_window}")
        self.logger.info(f"   📈 Regret threshold: {regret_stability_threshold}")
        
        # Initialize sequential trainer with reduced tournament penalty
        from enhanced_cfr_trainer_v2 import SequentialScenarioTrainer
        sequential_trainer = SequentialScenarioTrainer(
            scenarios=self.scenarios,
            iterations_per_scenario=iterations_per_scenario,
            stopping_condition_window=stopping_condition_window,
            regret_stability_threshold=regret_stability_threshold,
            tournament_survival_penalty=0.2  # Reduced from default 0.6 for more risk-taking
        )
        
        training_start_time = time.time()
        
        # Run sequential training
        results = sequential_trainer.run_sequential_training()
        
        training_end_time = time.time()
        total_time = training_end_time - training_start_time
        
        # Log comprehensive results
        self.logger.info(f"✅ Sequential training completed!")
        self.logger.info(f"   ⏱️  Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"   📊 Scenarios processed: {len(results):,}")
        
        total_iterations = sum(r['iterations_completed'] for r in results)
        avg_iterations = total_iterations / len(results) if results else 0
        
        self.logger.info(f"   🔢 Total iterations: {total_iterations:,}")
        self.logger.info(f"   🎯 Avg iterations per scenario: {avg_iterations:.1f}")
        
        # Stopping condition analysis
        from collections import Counter
        stop_reasons = Counter(r['stop_reason'] for r in results)
        self.logger.info(f"   📈 Stopping reasons:")
        for reason, count in stop_reasons.items():
            percentage = (count / len(results)) * 100
            self.logger.info(f"      {reason}: {count} scenarios ({percentage:.1f}%)")
        
        # Export enhanced results with timestamps
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        strategies_file = f"gcp_sequential_strategies_{timestamp}.csv"
        completion_file = f"gcp_scenario_completion_{timestamp}.csv"
        
        sequential_trainer.export_strategies_to_csv(strategies_file)
        sequential_trainer.export_scenario_completion_report(completion_file)
        
        self.output_files.extend([strategies_file, completion_file])
        
        self.logger.info(f"📁 Exported files:")
        self.logger.info(f"   - {strategies_file}")
        self.logger.info(f"   - {completion_file}")
        
        return {
            'results': results,
            'trainer': sequential_trainer,
            'total_time': total_time,
            'total_iterations': total_iterations,
            'output_files': [strategies_file, completion_file]
        }

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
        
        while completed_workers < self.n_workers and not self.shutdown_requested:
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
                    self.current_worker_results = worker_results  # Store for emergency backup
                    completed_workers += 1
                    
                    # Log completion details and check for incomplete iterations
                    iterations_completed = data.get('iterations_completed', 0)
                    iterations_attempted = data.get('iterations_attempted', 0)
                    success_rate = (iterations_completed / iterations_attempted * 100) if iterations_attempted > 0 else 0
                    
                    self.logger.info(f"✅ Worker {worker_id} completed ({completed_workers}/{self.n_workers})")
                    self.logger.info(f"   📊 Iterations: {iterations_completed:,}/{iterations_attempted:,} ({success_rate:.1f}% success)")
                    
                    # Check for incomplete iterations and log as critical error
                    if iterations_completed < iterations_attempted:
                        incomplete_count = iterations_attempted - iterations_completed
                        incomplete_rate = (incomplete_count / iterations_attempted * 100) if iterations_attempted > 0 else 0
                        
                        error_message = (f"Worker completed fewer iterations than assigned. "
                                       f"Missing {incomplete_count:,} iterations ({incomplete_rate:.1f}% incomplete)")
                        
                        self.log_error_to_file(
                            error_type="WorkerIncompleteError",
                            message=error_message,
                            worker_id=worker_id,
                            iteration=iterations_completed,
                            scenario_context=f"Expected: {iterations_attempted}, Completed: {iterations_completed}"
                        )
                    
                elif msg_type == 'error':
                    self.logger.error(f"❌ Worker {worker_id} error: {data}")
                
                elif msg_type == 'iteration_error':
                    # Handle detailed iteration error reports
                    error_info = data
                    self.logger.warning(f"⚠️  Worker {worker_id}: Iteration error at {error_info.get('iteration', 'unknown')}")
                    self.logger.warning(f"   🎯 Scenario: {error_info.get('scenario_key', 'unknown')}")
                    self.logger.warning(f"   💥 Error: {error_info.get('error_type', 'unknown')}: {error_info.get('error_message', 'unknown')}")
                
                elif msg_type == 'worker_critical_error':
                    # Handle critical worker failure
                    error_info = data
                    self.logger.error(f"💥 Worker {worker_id}: CRITICAL FAILURE")
                    self.logger.error(f"   📊 Completed {error_info.get('iterations_completed', 0)} iterations")
                    self.logger.error(f"   🎯 Last scenario: {error_info.get('current_scenario_key', 'unknown')}")
                    self.logger.error(f"   💥 Error: {error_info.get('error_type', 'unknown')}: {error_info.get('error_message', 'unknown')}")
                    # Don't increment completed_workers for critical failures
                
                # Periodic logging and checkpointing (every 15 minutes or as configured)
                current_time = time.time()
                if current_time - last_log_time >= (self.log_interval_minutes * 60):
                    # Combine worker results before logging to ensure unified CSV has latest data
                    if worker_results:
                        self.combine_worker_results(worker_results)
                    self.log_training_progress(current_time)
                    self.save_checkpoint(worker_results, current_time)  # Add checkpointing
                    last_log_time = current_time
                    
            except:
                # Check for shutdown request even during timeout
                if self.shutdown_requested:
                    self.logger.info("🛑 Shutdown requested - stopping training loop")
                    break
                    
                # Timeout - check if any processes are still alive
                alive_count = sum(1 for p in processes if p.is_alive())
                if alive_count == 0:
                    break
                    
                # Periodic logging even during quiet periods
                current_time = time.time()
                if current_time - last_log_time >= (self.log_interval_minutes * 60):
                    # Combine worker results before logging to ensure unified CSV has latest data
                    if worker_results:
                        self.combine_worker_results(worker_results)
                    self.log_training_progress(current_time)
                    self.save_checkpoint(worker_results, current_time)
                    last_log_time = current_time
        
        # Handle shutdown scenario
        if self.shutdown_requested:
            self.logger.info("🛑 Graceful shutdown in progress - terminating workers...")
            # Terminate any remaining processes
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    self.logger.info(f"   🛑 Terminated worker process {p.pid}")
            
            # Wait a bit for clean termination
            time.sleep(2)
            
            # Force kill if necessary
            for p in processes:
                if p.is_alive():
                    p.kill()
                    self.logger.warning(f"   💀 Force killed worker process {p.pid}")
        else:
            # Normal completion - wait for all processes to complete and check exit codes
            for i, p in enumerate(processes):
                p.join()
                exit_code = p.exitcode
                
                # Check for abnormal exit codes
                if exit_code != 0:
                    # Find the last known iteration for this worker
                    worker_id = i
                    last_iteration = 0
                    if worker_id in worker_results:
                        last_iteration = worker_results[worker_id].get('iterations_completed', 0)
                    
                    error_message = f"Worker process exited abnormally with exit code {exit_code}"
                    
                    # Log abnormal exit as critical error
                    self.log_error_to_file(
                        error_type="WorkerAbnormalExit",
                        message=error_message,
                        worker_id=worker_id,
                        iteration=last_iteration,
                        scenario_context=f"Process ID: {p.pid}",
                        exit_code=exit_code
                    )
                    
                    self.logger.error(f"❌ Worker {worker_id}: Process exited abnormally with code {exit_code}")
        
        # Check for workers that never reported results (silent failures)
        for worker_id in range(self.n_workers):
            if worker_id not in worker_results:
                error_message = f"Worker never reported completion results (silent failure)"
                
                self.log_error_to_file(
                    error_type="WorkerSilentFailure",
                    message=error_message,
                    worker_id=worker_id,
                    iteration=0,
                    scenario_context="Worker never sent results through queue"
                )
                
                self.logger.error(f"❌ Worker {worker_id}: Silent failure - no results received")
        
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
        """
        Log periodic training progress with memory usage monitoring.
        Enhanced for long-running GCP jobs.
        Now includes unified scenario lookup table export at each logging interval.
        """
        elapsed_time = current_time - self.start_time
        
        # Get memory usage information
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            system_memory_gb = system_memory.total / 1024 / 1024 / 1024
            system_memory_used_pct = system_memory.percent
            
            # Get CPU usage
            cpu_percent = process.cpu_percent()
            
        except Exception as memory_error:
            self.logger.warning(f"⚠️ Memory monitoring failed: {memory_error}")
            memory_mb = 0
            system_memory_gb = 0
            system_memory_used_pct = 0
            cpu_percent = 0
        
        self.logger.info(f"📊 Training Progress Update")
        self.logger.info(f"   ⏱️  Elapsed time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.1f} hours)")
        self.logger.info(f"   📈 Performance metrics collected: {len(self.performance_metrics)}")
        self.logger.info(f"   🎯 Scenario training distribution balance in progress...")
        self.logger.info(f"   💾 Process memory usage: {memory_mb:.1f} MB")
        self.logger.info(f"   🖥️  System memory: {system_memory_used_pct:.1f}% of {system_memory_gb:.1f} GB used")
        self.logger.info(f"   ⚡ CPU usage: {cpu_percent:.1f}%")
        
        # Memory usage warnings for long-running jobs
        if memory_mb > 1000:  # More than 1GB
            self.logger.warning(f"⚠️ High memory usage detected: {memory_mb:.1f} MB")
            self.logger.warning(f"   💡 Consider reducing worker count or enabling more aggressive pruning")
        
        if system_memory_used_pct > 90:
            self.logger.warning(f"⚠️ System memory critically low: {system_memory_used_pct:.1f}% used")
            self.logger.warning(f"   💡 System may become unstable - consider reducing workload")
        
        # Log worker progress if available
        if hasattr(self, 'combined_scenario_counter') and self.combined_scenario_counter:
            total_trained_scenarios = len(self.combined_scenario_counter)
            total_training_iterations = sum(self.combined_scenario_counter.values())
            avg_iterations_per_scenario = total_training_iterations / total_trained_scenarios if total_trained_scenarios > 0 else 0
            
            self.logger.info(f"   📊 Scenarios trained: {total_trained_scenarios}/330 ({total_trained_scenarios/330*100:.1f}%)")
            self.logger.info(f"   🔄 Total training iterations: {total_training_iterations:,}")
            self.logger.info(f"   📈 Avg iterations per scenario: {avg_iterations_per_scenario:.1f}")
        
        # Export unified scenario lookup table at each logging interval
        try:
            self.export_unified_scenario_lookup_csv("scenario_lookup_table.csv")
        except Exception as export_error:
            self.logger.warning(f"⚠️ Scenario lookup table export failed: {export_error}")
    
    def save_checkpoint(self, worker_results, current_time):
        """
        Save training state to checkpoint file for recovery.
        This implements the periodic checkpointing requirement (every 15 minutes).
        Saves current strategies, regrets, performance metrics, and training progress.
        
        Args:
            worker_results: Current worker results dictionary
            current_time: Current timestamp for the checkpoint
        """
        try:
            # Create timestamped checkpoint filename for easy identification
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoints_dir / f"cfr_checkpoint_{timestamp}.pkl"
            
            # Combine current worker results if available to get latest state
            if worker_results:
                self.combine_worker_results(worker_results)
            
            # Prepare comprehensive checkpoint data for recovery
            checkpoint_data = {
                'timestamp': timestamp,
                'elapsed_time': current_time - self.start_time,
                'start_time': self.start_time,
                'current_time': current_time,
                'n_workers': self.n_workers,
                'log_interval_minutes': self.log_interval_minutes,
                'scenarios': self.scenarios,
                'scenario_training_counts': dict(self.scenario_training_counts),
                'performance_metrics': self.performance_metrics,
                'iteration_count': self.iteration_count,
                # Save combined training state for resumption
                'combined_regret_sum': dict(self.combined_regret_sum) if hasattr(self, 'combined_regret_sum') else {},
                'combined_strategy_sum': dict(self.combined_strategy_sum) if hasattr(self, 'combined_strategy_sum') else {},
                'combined_scenario_counter': dict(self.combined_scenario_counter) if hasattr(self, 'combined_scenario_counter') else {},
                'worker_results': worker_results,
                'version': '1.0'  # For compatibility checking during restoration
            }
            
            # Save checkpoint to disk using pickle for efficient storage
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
            self.logger.info(f"   📊 Data size: {len(pickle.dumps(checkpoint_data)) / 1024 / 1024:.1f} MB")
            self.logger.info(f"   ⏱️  Training time: {(current_time - self.start_time)/60:.1f} minutes")
            
            # Clean up old checkpoints to save disk space (keep only last 5)
            self._cleanup_old_checkpoints()
            
        except Exception as checkpoint_error:
            self.logger.error(f"❌ Checkpoint save failed: {checkpoint_error}")
            import traceback
            self.logger.error(f"   🔍 Checkpoint error traceback:\n{traceback.format_exc()}")
    
    def _cleanup_old_checkpoints(self):
        """
        Keep only the 5 most recent checkpoints to save disk space.
        Important for long-running GCP jobs to prevent disk space issues.
        """
        try:
            checkpoint_files = list(self.checkpoints_dir.glob("cfr_checkpoint_*.pkl"))
            if len(checkpoint_files) > 5:
                # Sort by modification time (newest first)
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                # Remove oldest checkpoints beyond the 5 most recent
                for old_checkpoint in checkpoint_files[5:]:
                    old_checkpoint.unlink()
                    self.logger.info(f"   🗑️  Cleaned up old checkpoint: {old_checkpoint.name}")
        except Exception as cleanup_error:
            self.logger.warning(f"⚠️  Checkpoint cleanup warning: {cleanup_error}")
    
    def load_latest_checkpoint(self):
        """
        Load the most recent checkpoint if available for training resumption.
        This implements the checkpoint restoration requirement.
        
        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        try:
            checkpoint_files = list(self.checkpoints_dir.glob("cfr_checkpoint_*.pkl"))
            if not checkpoint_files:
                self.logger.info("📁 No existing checkpoints found")
                return False
            
            # Find most recent checkpoint based on file modification time
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            self.logger.info(f"🔍 Found checkpoint: {latest_checkpoint}")
            
            # Load checkpoint data from disk
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint version for compatibility
            if checkpoint_data.get('version') != '1.0':
                self.logger.warning(f"⚠️  Checkpoint version mismatch: {checkpoint_data.get('version')} vs 1.0")
                return False
            
            # Restore training state from checkpoint
            self.scenario_training_counts = Counter(checkpoint_data.get('scenario_training_counts', {}))
            self.performance_metrics = checkpoint_data.get('performance_metrics', [])
            self.iteration_count = checkpoint_data.get('iteration_count', 0)
            
            # Restore combined results if available (for continued training)
            if checkpoint_data.get('combined_regret_sum'):
                self.combined_regret_sum = defaultdict(lambda: defaultdict(float))
                for scenario_key, regrets in checkpoint_data['combined_regret_sum'].items():
                    for action, value in regrets.items():
                        self.combined_regret_sum[scenario_key][action] = value
            
            if checkpoint_data.get('combined_strategy_sum'):
                self.combined_strategy_sum = defaultdict(lambda: defaultdict(float))
                for scenario_key, strategies in checkpoint_data['combined_strategy_sum'].items():
                    for action, value in strategies.items():
                        self.combined_strategy_sum[scenario_key][action] = value
            
            if checkpoint_data.get('combined_scenario_counter'):
                self.combined_scenario_counter = Counter(checkpoint_data['combined_scenario_counter'])
            
            # Calculate resumed training time for logging
            saved_elapsed = checkpoint_data.get('elapsed_time', 0)
            
            self.logger.info(f"✅ Checkpoint loaded successfully!")
            self.logger.info(f"   📅 Saved: {checkpoint_data.get('timestamp', 'unknown')}")
            self.logger.info(f"   ⏱️  Previous training time: {saved_elapsed/60:.1f} minutes")
            self.logger.info(f"   📊 Scenarios trained: {len(self.scenario_training_counts)}")
            self.logger.info(f"   📈 Performance metrics: {len(self.performance_metrics)}")
            self.logger.info(f"   🔄 Iterations completed: {self.iteration_count:,}")
            
            return True
            
        except Exception as load_error:
            self.logger.error(f"❌ Checkpoint load failed: {load_error}")
            import traceback
            self.logger.error(f"   🔍 Load error traceback:\n{traceback.format_exc()}")
            return False
    
    def prompt_checkpoint_resume(self):
        """
        Prompt user whether to resume from checkpoint or start fresh.
        For automated GCP jobs, this auto-resumes if checkpoint exists and is recent.
        This implements the user-friendly resume requirement.
        
        Returns:
            bool: True if resumed from checkpoint, False if starting fresh
        """
        checkpoint_files = list(self.checkpoints_dir.glob("cfr_checkpoint_*.pkl"))
        if not checkpoint_files:
            self.logger.info("🆕 No checkpoints found - starting fresh training")
            return False
        
        # Find most recent checkpoint and check its age
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        checkpoint_age = time.time() - latest_checkpoint.stat().st_mtime
        
        self.logger.info(f"🔍 Found checkpoint: {latest_checkpoint.name}")
        self.logger.info(f"   📅 Age: {checkpoint_age/3600:.1f} hours ago")
        
        # For automated/GCP environments, auto-resume from recent checkpoints
        # This provides robust recovery for long-running jobs
        if checkpoint_age < 86400:  # Less than 24 hours old
            self.logger.info("🔄 Auto-resuming from recent checkpoint (less than 24h old)")
            return self.load_latest_checkpoint()
        else:
            # For older checkpoints, try to resume but don't fail if it doesn't work
            self.logger.info("⚠️  Checkpoint is older than 24h - attempting to load anyway")
            if self.load_latest_checkpoint():
                return True
            else:
                self.logger.info("🆕 Checkpoint load failed - starting fresh training")
                return False
    
    def combine_worker_results(self, worker_results):
        """Combine results from all workers"""
        self.logger.info(f"🔄 Combining results from {len(worker_results)} workers...")
        
        self.combined_regret_sum = defaultdict(lambda: defaultdict(float))
        self.combined_strategy_sum = defaultdict(lambda: defaultdict(float))
        self.combined_scenario_counter = Counter()
        self.all_performance_metrics = []
        
        for worker_id, worker_data in worker_results.items():
            # Safe access to worker data fields
            iterations = worker_data.get('iterations_completed', 0)
            final_time = worker_data.get('final_time', 0.0)
            
            self.logger.info(f"   Worker {worker_id}: {iterations:,} iterations in {final_time:.1f}s")
            
            # Combine regrets safely
            regret_sum = worker_data.get('regret_sum', {})
            for scenario_key, regrets in regret_sum.items():
                if isinstance(regrets, dict):
                    for action, regret_value in regrets.items():
                        self.combined_regret_sum[scenario_key][action] += regret_value
            
            # Combine strategies safely
            strategy_sum = worker_data.get('strategy_sum', {})
            for scenario_key, strategies in strategy_sum.items():
                if isinstance(strategies, dict):
                    for action, strategy_value in strategies.items():
                        self.combined_strategy_sum[scenario_key][action] += strategy_value
            
            # Combine counters safely
            scenario_counter = worker_data.get('scenario_counter', {})
            for scenario_key, count in scenario_counter.items():
                self.combined_scenario_counter[scenario_key] += count
            
            # Combine performance metrics safely
            performance_metrics = worker_data.get('performance_metrics', [])
            if isinstance(performance_metrics, list):
                self.all_performance_metrics.extend(performance_metrics)
        
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
                
                # Parse scenario key (bet_size_category removed, blinds_level added) 
                parts = scenario_key.split("|")
                if len(parts) >= 4:
                    hand_category = parts[0]
                    position = parts[1] 
                    stack_category = parts[2]
                    blinds_level = parts[3]
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
                
                # GROUP ACTIONS FOR RECOMMENDED ACTION CALCULATION
                # This grouping logic ensures that the recommended action is based on
                # the highest total probability across action types, not just the single
                # highest action. This prevents recommending FOLD when combined raise
                # probabilities (raise_small + raise_mid + raise_high) are higher.
                #
                # Action Groups:
                # - FOLD: fold actions only
                # - CALL: call_small + call_mid + call_high  
                # - RAISE: raise_small + raise_mid + raise_high
                
                # Calculate group totals
                fold_total = action_percentages.get('fold_percent', 0.0)
                call_total = (action_percentages.get('call_small_percent', 0.0) + 
                             action_percentages.get('call_mid_percent', 0.0) + 
                             action_percentages.get('call_high_percent', 0.0))
                raise_total = (action_percentages.get('raise_small_percent', 0.0) + 
                              action_percentages.get('raise_mid_percent', 0.0) + 
                              action_percentages.get('raise_high_percent', 0.0))
                
                # Determine recommended action based on highest group total
                group_totals = {
                    'FOLD': fold_total,
                    'CALL': call_total, 
                    'RAISE': raise_total
                }
                
                # Find the group with the highest total probability
                recommended_action = max(group_totals.items(), key=lambda x: x[1])[0]
                recommended_confidence = max(group_totals.values())
                
                # Get training statistics
                training_games = self.combined_scenario_counter.get(scenario_key, 0)
                
                # Build lookup table row
                row = {
                    'scenario_key': scenario_key,
                    'hand_category': hand_category,
                    'example_hands': example_hands,
                    'position': position,
                    'stack_depth': stack_category,
                    'blinds_level': blinds_level,
                    'training_games': training_games,
                    'recommended_action': recommended_action,
                    'confidence_percent': round(recommended_confidence, 2),
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
    
    def export_unified_scenario_lookup_csv(self, filename="scenario_lookup_table.csv"):
        """
        Export unified scenario lookup table with aggregated data from all workers.
        This is updated at every logging interval to provide real-time learning progress.
        
        CSV contains:
        - scenario_key: Unique identifier combining all scenario metrics
        - hand_category: Type of poker hand (premium_pairs, medium_aces, etc.)
        - stack_category: Stack depth category (ultra_short, short, medium, deep, very_deep)
        - blinds_level: Blinds level (low, medium, high)
        - position: Player position (BTN, BB)
        - opponent_action: Current opponent context (if applicable)
        - iterations_completed: Number of training iterations completed for this scenario
        - total_rollouts: Total rollouts performed (same as iterations for enhanced CFR)
        - regret: Current average regret for this scenario
        - average_strategy: Primary learned strategy (FOLD/CALL/RAISE group)
        - strategy_confidence: Confidence percentage for the primary strategy
        """
        self.logger.info(f"📊 Exporting unified scenario lookup table to {filename}...")
        
        from enhanced_cfr_preflop_generator_v2 import PREFLOP_HAND_RANGES
        
        export_data = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get all unique scenarios that have been encountered
        all_scenario_keys = set()
        if hasattr(self, 'combined_scenario_counter'):
            all_scenario_keys.update(self.combined_scenario_counter.keys())
        if hasattr(self, 'combined_strategy_sum'):
            all_scenario_keys.update(self.combined_strategy_sum.keys())
        if hasattr(self, 'combined_regret_sum'):
            all_scenario_keys.update(self.combined_regret_sum.keys())
        
        for scenario_key in all_scenario_keys:
            # Parse scenario key: hand_category|position|stack_category|blinds_level
            parts = scenario_key.split("|")
            if len(parts) >= 4:
                hand_category = parts[0]
                position = parts[1] 
                stack_category = parts[2]
                blinds_level = parts[3]
            else:
                continue
            
            # Get training statistics
            iterations_completed = self.combined_scenario_counter.get(scenario_key, 0) if hasattr(self, 'combined_scenario_counter') else 0
            total_rollouts = iterations_completed  # Same as iterations for enhanced CFR
            
            # Calculate average regret
            average_regret = 0.0
            if hasattr(self, 'combined_regret_sum') and scenario_key in self.combined_regret_sum:
                regret_values = list(self.combined_regret_sum[scenario_key].values())
                if regret_values:
                    average_regret = sum(regret_values) / len(regret_values)
            
            # Calculate strategy information
            average_strategy = "UNKNOWN"
            strategy_confidence = 0.0
            opponent_action = "mixed"  # Default since we aggregate across different opponent contexts
            
            if hasattr(self, 'combined_strategy_sum') and scenario_key in self.combined_strategy_sum:
                strategy_counts = self.combined_strategy_sum[scenario_key]
                if sum(strategy_counts.values()) > 0:
                    total_count = sum(strategy_counts.values())
                    
                    # Group actions (same logic as existing export_lookup_table_csv)
                    fold_total = strategy_counts.get('fold', 0.0)
                    call_total = (strategy_counts.get('call_small', 0.0) + 
                                 strategy_counts.get('call_mid', 0.0) + 
                                 strategy_counts.get('call_high', 0.0))
                    raise_total = (strategy_counts.get('raise_small', 0.0) + 
                                  strategy_counts.get('raise_mid', 0.0) + 
                                  strategy_counts.get('raise_high', 0.0))
                    
                    # Determine primary strategy
                    group_totals = {
                        'FOLD': (fold_total / total_count) * 100,
                        'CALL': (call_total / total_count) * 100,
                        'RAISE': (raise_total / total_count) * 100
                    }
                    
                    if group_totals:
                        average_strategy = max(group_totals.items(), key=lambda x: x[1])[0]
                        strategy_confidence = max(group_totals.values())
            
            # Build unified lookup table row
            row = {
                'scenario_key': scenario_key,
                'hand_category': hand_category,
                'stack_category': stack_category,
                'blinds_level': blinds_level,
                'position': position,
                'opponent_action': opponent_action,
                'iterations_completed': iterations_completed,
                'total_rollouts': total_rollouts,
                'regret': round(average_regret, 6),
                'average_strategy': average_strategy,
                'strategy_confidence': round(strategy_confidence, 2),
                'last_updated': current_time
            }
            
            export_data.append(row)
        
        if export_data:
            df = pd.DataFrame(export_data)
            
            # Sort by iterations_completed descending, then by strategy_confidence
            df = df.sort_values(['iterations_completed', 'strategy_confidence'], ascending=[False, False])
            
            # Export to CSV, replacing previous version
            df.to_csv(filename, index=False)
            
            self.logger.info(f"✅ Exported unified scenario lookup table: {len(export_data)} scenarios")
            self.logger.info(f"   📊 Total iterations across all scenarios: {df['iterations_completed'].sum():,}")
            self.logger.info(f"   🎯 Average iterations per scenario: {df['iterations_completed'].mean():.1f}")
            self.logger.info(f"   📈 Scenarios with >100 iterations: {len(df[df['iterations_completed'] > 100])}")
            
            # Show strategy distribution
            if len(df) > 0:
                strategy_dist = df['average_strategy'].value_counts()
                self.logger.info(f"   🎯 Strategy Distribution:")
                for strategy, count in strategy_dist.items():
                    pct = count/len(export_data)*100 if len(export_data) > 0 else 0
                    self.logger.info(f"      {strategy}: {count} scenarios ({pct:.1f}%)")
            
            return df
        else:
            self.logger.info("📊 No scenario data available yet for lookup table export")
            # Create empty CSV with headers for consistency
            empty_df = pd.DataFrame(columns=[
                'scenario_key', 'hand_category', 'stack_category', 'blinds_level', 
                'position', 'opponent_action', 'iterations_completed', 'total_rollouts', 
                'regret', 'average_strategy', 'strategy_confidence', 'last_updated'
            ])
            empty_df.to_csv(filename, index=False)
            return empty_df
    
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
            'gcp_cfr_performance.csv',   # Performance metrics
            'scenario_lookup_table.csv'  # Unified scenario lookup table (NEW)
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
    """
    Setup graceful shutdown handlers for robust GCP job management.
    Handles SIGINT (Ctrl+C), SIGTERM (container termination), and other signals.
    """
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        trainer.logger.info(f"🛑 Received signal {signal_name} ({signum}), initiating graceful shutdown...")
        
        try:
            # Set shutdown flag to stop workers gracefully
            if hasattr(trainer, 'shutdown_requested'):
                trainer.shutdown_requested = True
            
            # Export current progress with emergency prefix
            trainer.logger.info("💾 Creating emergency backup before shutdown...")
            
            if hasattr(trainer, 'combined_strategy_sum') and trainer.combined_strategy_sum:
                emergency_lookup = f"emergency_backup_lookup_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                emergency_performance = f"emergency_backup_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                trainer.export_lookup_table_csv(emergency_lookup)
                trainer.export_performance_metrics_csv(emergency_performance)
                
                trainer.logger.info(f"✅ Emergency backup saved:")
                trainer.logger.info(f"   📄 Lookup table: {emergency_lookup}")
                trainer.logger.info(f"   📄 Performance: {emergency_performance}")
            
            # Save final checkpoint
            if hasattr(trainer, 'save_checkpoint'):
                try:
                    trainer.logger.info("💾 Saving final checkpoint...")
                    # Get current worker results if available
                    worker_results = getattr(trainer, 'current_worker_results', {})
                    trainer.save_checkpoint(worker_results, time.time())
                    trainer.logger.info("✅ Final checkpoint saved successfully")
                except Exception as checkpoint_error:
                    trainer.logger.error(f"❌ Final checkpoint save failed: {checkpoint_error}")
            
            trainer.logger.info(f"🏁 Graceful shutdown complete for signal {signal_name}")
            
        except Exception as shutdown_error:
            trainer.logger.error(f"❌ Error during graceful shutdown: {shutdown_error}")
            import traceback
            trainer.logger.error(f"   🔍 Shutdown error traceback:\n{traceback.format_exc()}")
        
        finally:
            # Exit gracefully
            sys.exit(0)
    
    # Register handlers for common termination signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    # Also handle SIGUSR1 for custom graceful shutdown (if supported)
    try:
        signal.signal(signal.SIGUSR1, signal_handler)  # Custom graceful shutdown
        trainer.logger.info("📡 Signal handlers registered: SIGINT, SIGTERM, SIGUSR1")
    except AttributeError:
        # SIGUSR1 might not be available on all platforms
        trainer.logger.info("📡 Signal handlers registered: SIGINT, SIGTERM")
    
    trainer.logger.info("🛡️ Graceful shutdown system initialized")


def main():
    """
    Main GCP CFR Training execution with support for both parallel and sequential approaches
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="GCP CFR Training System")
    parser.add_argument("--mode", choices=["parallel", "sequential"], default="parallel",
                       help="Training mode: parallel (original) or sequential (new)")
    parser.add_argument("--iterations", type=int, default=None,
                       help="Total iterations for parallel mode (default: 40,000 per worker)")
    parser.add_argument("--iterations-per-scenario", type=int, default=1000,
                       help="Iterations per scenario for sequential mode")
    parser.add_argument("--stopping-window", type=int, default=100,
                       help="Stopping condition window size for sequential mode")
    parser.add_argument("--regret-threshold", type=float, default=0.01,
                       help="Regret stability threshold for sequential mode")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker processes (default: all CPU cores)")
    
    args = parser.parse_args()
    
    print("🚀 GCP CFR Training System Starting...")
    print("=" * 80)
    print(f"🎯 Training mode: {args.mode.upper()}")
    
    # Initialize trainer with production settings
    trainer = GCPCFRTrainer(
        n_workers=args.workers or mp.cpu_count(),
        log_interval_minutes=15
    )
    
    # Calculate default iterations if not provided: 40,000 per worker
    if args.iterations is None:
        args.iterations = trainer.n_workers * 40000
        print(f"💡 Using default: {args.iterations:,} total iterations (40,000 per worker × {trainer.n_workers} workers)")
    
    # Setup graceful shutdown
    setup_signal_handlers(trainer)
    
    try:
        if args.mode == "sequential":
            # Run new sequential training approach
            print(f"📊 Sequential Training Parameters:")
            print(f"   🔄 Iterations per scenario: {args.iterations_per_scenario:,}")
            print(f"   🛑 Stopping window: {args.stopping_window}")
            print(f"   📈 Regret threshold: {args.regret_threshold}")
            
            result = trainer.run_sequential_training(
                iterations_per_scenario=args.iterations_per_scenario,
                stopping_condition_window=args.stopping_window,
                regret_stability_threshold=args.regret_threshold
            )
            
            # Sequential training handles its own exports
            print("\n🎯 SEQUENTIAL CFR TRAINING COMPLETE!")
            print("✅ New approach features:")
            print("   📋 Sequential scenario processing")
            print("   🛑 Automatic stopping conditions")
            print("   ⏱️  Real-time progress estimation")
            print("   📊 Enhanced completion reporting")
            
        else:
            # Run original parallel training approach
            print(f"📊 Parallel Training Parameters:")
            print(f"   🔢 Total iterations: {args.iterations:,}")
            print(f"   👥 Workers: {trainer.n_workers}")
            print(f"   ⚡ Iterations per worker: {args.iterations // trainer.n_workers:,}")
            
            trainer.run_parallel_training(total_iterations=args.iterations)
            
            # Export lookup table CSV (main requirement)
            lookup_df = trainer.export_lookup_table_csv("gcp_cfr_lookup_table.csv")
            
            # Export performance metrics CSV (requirement)
            performance_df = trainer.export_performance_metrics_csv("gcp_cfr_performance.csv")
            
            # Final training summary
            trainer.logger.info("🏆 GCP CFR Training Completed Successfully!")
            trainer.logger.info(f"📊 Lookup table exported: {len(lookup_df) if lookup_df is not None else 0} scenarios")
            trainer.logger.info(f"📈 Performance metrics exported: {len(performance_df) if performance_df is not None else 0} data points")
            
            print("\n🎯 PARALLEL CFR TRAINING COMPLETE!")
            print("✅ Original approach features:")
            print("   🖥️  Used all CPU cores (multiprocessing)")
            print("   ⚖️  Balanced hand category coverage")
            print("   📊 Generated lookup-table CSV")
            print("   📝 Comprehensive logging")
        
        # Cleanup and archive unused files
        trainer.cleanup_and_archive_files()
        
        print("\n📁 File management completed:")
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