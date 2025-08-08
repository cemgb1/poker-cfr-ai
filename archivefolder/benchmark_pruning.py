#!/usr/bin/env python3
"""
Benchmarking script for pruning techniques effectiveness analysis

This script compares performance metrics between different pruning configurations
to evaluate the effectiveness of regret-based, strategy, and action space pruning.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import tempfile
import os

from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer, SequentialScenarioTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios


class PruningBenchmark:
    """Comprehensive benchmarking suite for pruning techniques"""
    
    def __init__(self, num_scenarios=50, iterations_per_test=500):
        """
        Initialize benchmarking suite.
        
        Args:
            num_scenarios: Number of scenarios to use for benchmarking
            iterations_per_test: Number of training iterations per configuration
        """
        self.num_scenarios = num_scenarios
        self.iterations_per_test = iterations_per_test
        
        # Generate test scenarios
        all_scenarios = generate_enhanced_scenarios()
        self.test_scenarios = all_scenarios[:num_scenarios]
        print(f"🎯 Initialized benchmarking with {num_scenarios} scenarios")
        
        # Results storage
        self.results = {}
        self.timing_results = {}
        self.pruning_stats = {}
    
    def benchmark_regret_pruning_thresholds(self):
        """Benchmark different regret pruning thresholds"""
        print(f"\n🔬 Benchmarking Regret Pruning Thresholds")
        print("=" * 50)
        
        thresholds = [-500.0, -300.0, -200.0, -100.0, -50.0, 0.0]  # No pruning at 0.0
        results = {}
        
        for threshold in thresholds:
            print(f"⚙️ Testing regret threshold: {threshold}")
            
            trainer = EnhancedCFRTrainer(
                scenarios=self.test_scenarios,
                enable_pruning=True,
                regret_pruning_threshold=threshold,
                strategy_pruning_threshold=0.001  # Keep strategy pruning minimal
            )
            
            # Run training iterations
            start_time = time.time()
            convergence_scores = []
            
            for i in range(self.iterations_per_test):
                scenario = trainer.select_balanced_scenario()
                result = trainer.play_enhanced_scenario(scenario)
                trainer.scenario_counter[result['scenario_key']] += 1
                
                # Track convergence every 50 iterations
                if i % 50 == 0:
                    avg_regret = self._calculate_average_regret(trainer)
                    convergence_scores.append(avg_regret)
            
            training_time = time.time() - start_time
            pruning_stats = trainer.get_pruning_statistics()
            
            results[threshold] = {
                'training_time': training_time,
                'final_convergence': convergence_scores[-1] if convergence_scores else 0,
                'convergence_trajectory': convergence_scores,
                'pruning_stats': pruning_stats,
                'scenarios_trained': len(trainer.scenario_counter),
                'avg_iterations_per_scenario': np.mean(list(trainer.scenario_counter.values()))
            }
            
            print(f"   ⏱️ Training time: {training_time:.2f}s")
            print(f"   📊 Actions pruned: {pruning_stats['regret_pruned_count']}")
            print(f"   📈 Final convergence: {convergence_scores[-1]:.4f}")
        
        self.results['regret_thresholds'] = results
        return results
    
    def benchmark_strategy_pruning_thresholds(self):
        """Benchmark different strategy pruning thresholds"""
        print(f"\n🔬 Benchmarking Strategy Pruning Thresholds")
        print("=" * 50)
        
        thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        results = {}
        
        for threshold in thresholds:
            print(f"⚙️ Testing strategy threshold: {threshold}")
            
            trainer = EnhancedCFRTrainer(
                scenarios=self.test_scenarios,
                enable_pruning=True,
                regret_pruning_threshold=-300.0,  # Keep regret pruning standard
                strategy_pruning_threshold=threshold
            )
            
            # Run training to build up strategies
            for i in range(self.iterations_per_test):
                scenario = trainer.select_balanced_scenario()
                result = trainer.play_enhanced_scenario(scenario)
                trainer.scenario_counter[result['scenario_key']] += 1
            
            # Test strategy export with pruning
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                start_time = time.time()
                df = trainer.export_strategies_to_csv(tmp.name)
                export_time = time.time() - start_time
                
                if df is not None:
                    total_actions_pruned = df['actions_pruned_count'].sum()
                    avg_actions_pruned = df['actions_pruned_count'].mean()
                    confidence_scores = df['confidence'].tolist()
                else:
                    total_actions_pruned = 0
                    avg_actions_pruned = 0
                    confidence_scores = []
                
                os.unlink(tmp.name)
            
            pruning_stats = trainer.get_pruning_statistics()
            
            results[threshold] = {
                'export_time': export_time,
                'total_actions_pruned': total_actions_pruned,
                'avg_actions_pruned_per_scenario': avg_actions_pruned,
                'confidence_scores': confidence_scores,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'pruning_stats': pruning_stats
            }
            
            print(f"   ⏱️ Export time: {export_time:.4f}s")
            print(f"   ✂️ Actions pruned in export: {total_actions_pruned}")
            print(f"   📈 Avg confidence: {np.mean(confidence_scores):.3f}" if confidence_scores else "   📈 Avg confidence: N/A")
        
        self.results['strategy_thresholds'] = results
        return results
    
    def benchmark_action_space_pruning(self):
        """Benchmark action space pruning effectiveness"""
        print(f"\n🔬 Benchmarking Action Space Pruning")
        print("=" * 50)
        
        configs = [
            ('disabled', False),
            ('enabled', True)
        ]
        
        results = {}
        
        for config_name, enable_pruning in configs:
            print(f"⚙️ Testing action space pruning: {config_name}")
            
            trainer = EnhancedCFRTrainer(
                scenarios=self.test_scenarios,
                enable_pruning=enable_pruning
            )
            
            # Track action space reduction
            action_space_stats = defaultdict(list)
            training_times = []
            
            start_time = time.time()
            
            for i in range(self.iterations_per_test):
                scenario = trainer.select_balanced_scenario()
                result = trainer.play_enhanced_scenario(scenario)
                
                # Track action space reduction
                initial_actions = len(result.get('initial_actions', []))
                final_actions = len(result.get('available_actions', []))
                
                if initial_actions > 0:
                    reduction_ratio = (initial_actions - final_actions) / initial_actions
                    action_space_stats['reduction_ratios'].append(reduction_ratio)
                
                action_space_stats['initial_action_counts'].append(initial_actions)
                action_space_stats['final_action_counts'].append(final_actions)
                
                trainer.scenario_counter[result['scenario_key']] += 1
            
            total_training_time = time.time() - start_time
            pruning_stats = trainer.get_pruning_statistics()
            
            results[config_name] = {
                'training_time': total_training_time,
                'avg_reduction_ratio': np.mean(action_space_stats['reduction_ratios']) if action_space_stats['reduction_ratios'] else 0,
                'avg_initial_actions': np.mean(action_space_stats['initial_action_counts']),
                'avg_final_actions': np.mean(action_space_stats['final_action_counts']),
                'pruning_events': pruning_stats.get('total_pruning_events', 0),
                'pruning_stats': pruning_stats
            }
            
            print(f"   ⏱️ Training time: {total_training_time:.2f}s")
            print(f"   📊 Avg action reduction: {results[config_name]['avg_reduction_ratio']:.1%}")
            print(f"   🎯 Pruning events: {results[config_name]['pruning_events']}")
        
        self.results['action_space'] = results
        return results
    
    def benchmark_stopping_conditions(self):
        """Benchmark different stopping condition configurations"""
        print(f"\n🔬 Benchmarking Stopping Conditions")
        print("=" * 50)
        
        configs = [
            ('flexible_default', {
                'stopping_condition_mode': 'flexible',
                'enable_min_rollouts_stopping': True,
                'enable_regret_stability_stopping': True,
                'enable_max_iterations_stopping': True,
                'min_rollouts_before_convergence': 50
            }),
            ('strict_mode', {
                'stopping_condition_mode': 'strict',
                'enable_min_rollouts_stopping': True,
                'enable_regret_stability_stopping': True,
                'enable_max_iterations_stopping': True,
                'min_rollouts_before_convergence': 50
            }),
            ('regret_only', {
                'stopping_condition_mode': 'flexible',
                'enable_min_rollouts_stopping': True,
                'enable_regret_stability_stopping': True,
                'enable_max_iterations_stopping': False,
                'min_rollouts_before_convergence': 30
            })
        ]
        
        results = {}
        
        for config_name, config_params in configs:
            print(f"⚙️ Testing stopping configuration: {config_name}")
            
            trainer = SequentialScenarioTrainer(
                scenarios=self.test_scenarios[:5],  # Use fewer scenarios for stopping tests
                rollouts_per_visit=2,
                iterations_per_scenario=200,
                **config_params
            )
            
            # Track stopping behavior
            stopping_times = []
            stopping_reasons = []
            
            for scenario in trainer.scenario_list[:3]:  # Test 3 scenarios
                scenario_key = trainer.get_scenario_key(scenario)
                start_time = time.time()
                
                for iteration in range(200):  # Max iterations
                    # Simulate training
                    results_batch = trainer.play_scenario_multiple_rollouts(scenario)
                    
                    # Track regret
                    current_regret = trainer.get_current_scenario_regret(scenario_key)
                    trainer.scenario_regret_history[scenario_key].append(current_regret)
                    
                    # Check stopping condition
                    should_stop, reason = trainer.check_stopping_condition(scenario_key, iteration + 1)
                    if should_stop:
                        stop_time = time.time() - start_time
                        stopping_times.append(stop_time)
                        stopping_reasons.append(reason)
                        break
                else:
                    # Max iterations reached
                    stop_time = time.time() - start_time
                    stopping_times.append(stop_time)
                    stopping_reasons.append("max_iterations_reached")
            
            stopping_stats = trainer.get_stopping_statistics()
            
            results[config_name] = {
                'avg_stopping_time': np.mean(stopping_times),
                'stopping_times': stopping_times,
                'stopping_reasons': stopping_reasons,
                'stopping_stats': stopping_stats
            }
            
            print(f"   ⏱️ Avg stopping time: {np.mean(stopping_times):.2f}s")
            print(f"   🛑 Stopping reasons: {set(stopping_reasons)}")
        
        self.results['stopping_conditions'] = results
        return results
    
    def benchmark_combined_effectiveness(self):
        """Benchmark combined pruning techniques vs baseline"""
        print(f"\n🔬 Benchmarking Combined Effectiveness")
        print("=" * 50)
        
        configs = [
            ('baseline', {
                'enable_pruning': False
            }),
            ('pruning_conservative', {
                'enable_pruning': True,
                'regret_pruning_threshold': -300.0,
                'strategy_pruning_threshold': 0.001
            }),
            ('pruning_aggressive', {
                'enable_pruning': True,
                'regret_pruning_threshold': -150.0,
                'strategy_pruning_threshold': 0.01
            })
        ]
        
        results = {}
        
        for config_name, config_params in configs:
            print(f"⚙️ Testing configuration: {config_name}")
            
            trainer = EnhancedCFRTrainer(
                scenarios=self.test_scenarios,
                **config_params
            )
            
            # Comprehensive performance tracking
            start_time = time.time()
            convergence_history = []
            memory_efficiency = []
            
            for i in range(self.iterations_per_test):
                scenario = trainer.select_balanced_scenario()
                result = trainer.play_enhanced_scenario(scenario)
                trainer.scenario_counter[result['scenario_key']] += 1
                
                # Track convergence every 100 iterations
                if i % 100 == 0:
                    avg_regret = self._calculate_average_regret(trainer)
                    convergence_history.append(avg_regret)
                    
                    # Approximate memory usage by counting stored regrets
                    total_regrets = sum(len(regrets) for regrets in trainer.regret_sum.values())
                    memory_efficiency.append(total_regrets)
            
            training_time = time.time() - start_time
            pruning_stats = trainer.get_pruning_statistics()
            
            results[config_name] = {
                'training_time': training_time,
                'convergence_history': convergence_history,
                'final_convergence': convergence_history[-1] if convergence_history else 0,
                'memory_efficiency': memory_efficiency[-1] if memory_efficiency else 0,
                'pruning_stats': pruning_stats,
                'scenarios_covered': len(trainer.scenario_counter),
                'total_iterations': self.iterations_per_test
            }
            
            print(f"   ⏱️ Training time: {training_time:.2f}s")
            print(f"   📈 Final convergence: {convergence_history[-1]:.4f}" if convergence_history else "   📈 No convergence data")
            print(f"   🧠 Memory usage (regrets): {memory_efficiency[-1]}" if memory_efficiency else "   🧠 No memory data")
            print(f"   ✂️ Pruning events: {pruning_stats.get('total_pruning_events', 0)}")
        
        self.results['combined_effectiveness'] = results
        return results
    
    def _calculate_average_regret(self, trainer):
        """Calculate average regret across all scenarios"""
        total_regret = 0
        regret_count = 0
        
        for scenario_regrets in trainer.regret_sum.values():
            for regret in scenario_regrets.values():
                total_regret += abs(regret)
                regret_count += 1
        
        return total_regret / max(regret_count, 1)
    
    def generate_performance_report(self):
        """Generate comprehensive performance analysis report"""
        print(f"\n📊 PRUNING EFFECTIVENESS ANALYSIS REPORT")
        print("=" * 80)
        
        if 'regret_thresholds' in self.results:
            self._report_regret_analysis()
        
        if 'strategy_thresholds' in self.results:
            self._report_strategy_analysis()
        
        if 'action_space' in self.results:
            self._report_action_space_analysis()
        
        if 'stopping_conditions' in self.results:
            self._report_stopping_analysis()
        
        if 'combined_effectiveness' in self.results:
            self._report_combined_analysis()
    
    def _report_regret_analysis(self):
        """Report regret pruning threshold analysis"""
        print(f"\n🎯 REGRET PRUNING THRESHOLD ANALYSIS")
        print("-" * 40)
        
        results = self.results['regret_thresholds']
        
        # Find optimal threshold
        best_threshold = min(results.keys(), key=lambda t: results[t]['training_time'])
        fastest_convergence = min(results.keys(), key=lambda t: results[t]['final_convergence'])
        
        print(f"🏆 Fastest training: threshold {best_threshold} ({results[best_threshold]['training_time']:.2f}s)")
        print(f"📈 Best convergence: threshold {fastest_convergence} (regret: {results[fastest_convergence]['final_convergence']:.4f})")
        
        print(f"\n📋 Detailed Results:")
        for threshold in sorted(results.keys()):
            r = results[threshold]
            print(f"   Threshold {threshold:7.1f}: {r['training_time']:6.2f}s, "
                  f"convergence {r['final_convergence']:7.4f}, "
                  f"pruned {r['pruning_stats']['regret_pruned_count']:4d}")
    
    def _report_strategy_analysis(self):
        """Report strategy pruning threshold analysis"""
        print(f"\n🎯 STRATEGY PRUNING THRESHOLD ANALYSIS")
        print("-" * 40)
        
        results = self.results['strategy_thresholds']
        
        # Find optimal balance
        best_balance = min(results.keys(), key=lambda t: results[t]['export_time'] * (1 + results[t]['total_actions_pruned'] * 0.001))
        
        print(f"🏆 Best balance: threshold {best_balance}")
        print(f"📋 Detailed Results:")
        for threshold in sorted(results.keys()):
            r = results[threshold]
            print(f"   Threshold {threshold:5.3f}: {r['export_time']:6.4f}s export, "
                  f"pruned {r['total_actions_pruned']:4.0f} actions, "
                  f"confidence {r['avg_confidence']:5.3f}")
    
    def _report_action_space_analysis(self):
        """Report action space pruning analysis"""
        print(f"\n🎯 ACTION SPACE PRUNING ANALYSIS")
        print("-" * 40)
        
        results = self.results['action_space']
        
        disabled_result = results.get('disabled', {})
        enabled_result = results.get('enabled', {})
        
        if disabled_result and enabled_result:
            time_improvement = ((disabled_result['training_time'] - enabled_result['training_time']) / 
                              disabled_result['training_time'] * 100)
            
            print(f"⏱️ Training time improvement: {time_improvement:+.1f}%")
            print(f"📊 Average action reduction: {enabled_result['avg_reduction_ratio']:.1%}")
            print(f"🎯 Pruning events: {enabled_result['pruning_events']}")
    
    def _report_stopping_analysis(self):
        """Report stopping conditions analysis"""
        print(f"\n🎯 STOPPING CONDITIONS ANALYSIS")
        print("-" * 40)
        
        results = self.results['stopping_conditions']
        
        for config_name, result in results.items():
            print(f"⚙️ {config_name}:")
            print(f"   Average time: {result['avg_stopping_time']:.2f}s")
            print(f"   Reasons: {', '.join(set(result['stopping_reasons']))}")
    
    def _report_combined_analysis(self):
        """Report combined effectiveness analysis"""
        print(f"\n🎯 COMBINED EFFECTIVENESS ANALYSIS")
        print("-" * 40)
        
        results = self.results['combined_effectiveness']
        
        baseline = results.get('baseline', {})
        conservative = results.get('pruning_conservative', {})
        aggressive = results.get('pruning_aggressive', {})
        
        if baseline and conservative:
            time_improvement_conservative = ((baseline['training_time'] - conservative['training_time']) / 
                                           baseline['training_time'] * 100)
            print(f"🐌 Conservative pruning time improvement: {time_improvement_conservative:+.1f}%")
        
        if baseline and aggressive:
            time_improvement_aggressive = ((baseline['training_time'] - aggressive['training_time']) / 
                                         baseline['training_time'] * 100)
            print(f"🚀 Aggressive pruning time improvement: {time_improvement_aggressive:+.1f}%")
        
        print(f"\n🏆 RECOMMENDATIONS:")
        print(f"   • Use conservative pruning for stability")
        print(f"   • Use aggressive pruning for speed")
        print(f"   • Action space pruning provides consistent benefits")
        print(f"   • Flexible stopping conditions work well for most cases")


def run_comprehensive_benchmark():
    """Run comprehensive benchmarking suite"""
    print("🚀 Starting Comprehensive Pruning Techniques Benchmark")
    print("=" * 80)
    
    # Initialize benchmark
    benchmark = PruningBenchmark(num_scenarios=30, iterations_per_test=300)
    
    # Run all benchmarks
    print("🔬 Running benchmarks...")
    benchmark.benchmark_regret_pruning_thresholds()
    benchmark.benchmark_strategy_pruning_thresholds()
    benchmark.benchmark_action_space_pruning()
    benchmark.benchmark_stopping_conditions()
    benchmark.benchmark_combined_effectiveness()
    
    # Generate report
    benchmark.generate_performance_report()
    
    print("\n✅ Benchmark complete!")
    return benchmark


def run_quick_benchmark():
    """Run quick benchmark for development/testing"""
    print("⚡ Starting Quick Benchmark")
    print("=" * 50)
    
    benchmark = PruningBenchmark(num_scenarios=10, iterations_per_test=100)
    
    # Run subset of benchmarks
    benchmark.benchmark_combined_effectiveness()
    benchmark.generate_performance_report()
    
    return benchmark


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        run_quick_benchmark()
    else:
        run_comprehensive_benchmark()