# Performance Metrics Tracking

The CFR trainer now includes comprehensive performance metrics tracking to help analyze model convergence and learning dynamics.

## Features

### New CSV Output: `model_performance.csv`

When performance tracking is enabled, the trainer creates a `model_performance.csv` file with the following metrics tracked each iteration:

- **iteration**: Iteration number
- **time_per_iteration**: Time taken for this iteration (seconds)
- **total_elapsed_time**: Total elapsed training time (seconds)
- **average_regret**: Average regret across all scenarios and actions
- **max_regret**: Maximum regret value encountered
- **unique_scenarios_visited**: Number of unique scenarios encountered so far
- **scenario_coverage_***: Histogram of how many times each scenario has been visited
  - `scenario_coverage_0_10`: Scenarios visited 0-10 times
  - `scenario_coverage_11_25`: Scenarios visited 11-25 times
  - `scenario_coverage_26_50`: Scenarios visited 26-50 times
  - `scenario_coverage_51_100`: Scenarios visited 51-100 times
  - `scenario_coverage_100_plus`: Scenarios visited 100+ times

## Usage

### Option 1: Using Enhanced Trainer Directly

```python
from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
import random

# Generate scenarios and create trainer
scenarios = generate_enhanced_scenarios(100)
trainer = EnhancedCFRTrainer(scenarios=scenarios)

# Start performance tracking
trainer.start_performance_tracking()

# Training loop with metrics
for iteration in range(1000):
    scenario = random.choice(scenarios)
    trainer.play_enhanced_scenario(scenario)
    trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1
    
    # Record metrics every 100 iterations
    if iteration % 100 == 0:
        metrics = trainer.record_iteration_metrics(iteration)

# Export both results and performance metrics
trainer.export_strategies_to_csv("results.csv")
trainer.export_performance_metrics("model_performance.csv")
```

### Option 2: Using Expanded Demo with Performance Tracking

```python
from expanded_preflop_cfr_demo import run_expanded_preflop_training

# Run with performance tracking enabled
trainer = run_expanded_preflop_training(
    n_scenarios=200,
    n_iterations=2000,
    output_file="strategies.csv",
    enable_performance_tracking=True,  # Enable metrics
    metrics_interval=100  # Record every 100 iterations
)
```

### Option 3: Using Standalone Demo

```python
from demo_with_performance_metrics import demo_performance_tracking

# Run demo that shows performance tracking
trainer, perf_df, strategy_df = demo_performance_tracking(
    n_scenarios=50,
    n_iterations=500,
    metrics_interval=100
)
```

## Benefits

1. **Convergence Analysis**: Track how regret decreases over time to verify model is learning
2. **Training Efficiency**: Monitor time per iteration to optimize training parameters
3. **Scenario Coverage**: Ensure all scenarios get adequate training (avoid blind spots)
4. **Learning Dynamics**: Understand how the model explores different strategies
5. **Debug Training**: Identify issues like exploding regret or poor coverage

## Performance Impact

The performance tracking system is designed to be efficient:
- Metrics are calculated only at specified intervals (not every iteration)
- Regret calculations reuse existing data structures
- Timing overhead is minimal
- **Zero impact** when tracking is disabled (default mode)

## File Outputs

- **strategies CSV**: Unchanged - contains best actions and probabilities per scenario
- **model_performance.csv**: New - contains iteration-by-iteration training metrics
- Both files can be used together for comprehensive analysis

## Example Analysis

Use the performance metrics to:

1. **Check convergence**: Look for decreasing average regret over time
2. **Optimize training**: Find the ideal number of iterations before diminishing returns
3. **Verify coverage**: Ensure no scenarios are undertrained (stuck in 0-10 visits bin)
4. **Performance tuning**: Monitor training speed and adjust parameters accordingly

This feature maintains full backward compatibility - existing code continues to work unchanged.