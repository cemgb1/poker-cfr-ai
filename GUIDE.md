# CFR Poker AI Training System - Comprehensive Guide

A complete guide for setting up, running, and configuring the CFR (Counterfactual Regret Minimization) poker AI training system with real-time scenario analytics.

## 🎯 Overview

This system implements advanced CFR training for poker AI with:
- **Multiprocessing Training**: Parallel workers for fast convergence
- **Real-time Analytics**: Live CSV scenario lookup table with action frequencies
- **Natural Game Simulation**: Monte Carlo game-based training
- **Dynamic Scenarios**: Randomized stacks, blinds, and hand distributions
- **3-bet Detection**: Binary indicators for aggressive play patterns
- **Comprehensive Logging**: Progress tracking and performance metrics

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Training Modes](#training-modes)
5. [Configuration Options](#configuration-options)
6. [Output Files](#output-files)
7. [GCP Setup](#gcp-setup)
8. [Monitoring & Analysis](#monitoring--analysis)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 2+ cores (4+ recommended for parallel training)
- **RAM**: 4GB (8GB+ recommended for large-scale training)
- **Storage**: 2GB free space for logs and output files
- **Python**: 3.8+ with pip package manager

### Recommended for Production
- **CPU**: 8+ cores for optimal parallel performance
- **RAM**: 16GB+ for extensive scenario coverage
- **Storage**: 10GB+ for comprehensive logging and checkpoints
- **Network**: Stable connection for GCP deployment

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/cemgb1/poker-cfr-ai.git
cd poker-cfr-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python test_unified_scenario_lookup.py
```

Expected output: All tests pass with ✅ indicators.

## ⚡ Quick Start

### Basic Training Demo (5 minutes)
```bash
# Natural game training - recommended for beginners
python run_natural_cfr_training.py --mode demo --games 100

# GCP parallel training - for production use
python -c "
from run_gcp_cfr_training import GCPCFRTrainer
trainer = GCPCFRTrainer(n_workers=2, log_interval_minutes=1)
trainer.run_parallel_training(total_iterations=1000)
"
```

### Quick Test with Very Low Iterations
```bash
# Test that CSV output works even with minimal training
python test_low_iteration_csv_output.py
```

### View Results
After training, check the unified scenario lookup table:
```bash
# View the CSV with action frequencies and 3-bet indicators
head -20 scenario_lookup_table.csv
```

## 🎮 Training Modes

### 1. Natural Game CFR Training
**Best for**: Learning realistic poker scenarios through self-play

```bash
# Basic natural training
python run_natural_cfr_training.py --games 1000 --epsilon 0.1

# Advanced natural training with custom parameters
python run_natural_cfr_training.py \
    --games 5000 \
    --epsilon 0.2 \
    --min-visits 10 \
    --tournament-penalty 0.15 \
    --save-interval 500 \
    --log-interval 50
```

**Features:**
- Natural emergence of scenarios through gameplay
- Co-evolving hero and villain strategies
- Randomized game conditions (stacks, blinds, positions)
- Tournament-style survival penalties

### 2. GCP Parallel Training
**Best for**: Large-scale production training with maximum efficiency

```bash
# Production training with 8 workers
python -c "
from run_gcp_cfr_training import GCPCFRTrainer
trainer = GCPCFRTrainer(n_workers=8, log_interval_minutes=15)
trainer.run_parallel_training(total_iterations=200000)
"
```

**Features:**
- Utilizes all CPU cores for parallel processing
- Automatic load balancing across workers
- Real-time progress monitoring and checkpointing
- Memory-efficient scenario distribution

### 3. Sequential Scenario Training
**Best for**: Systematic coverage of specific scenario sets

```bash
# Sequential training with convergence criteria
python -c "
from run_gcp_cfr_training import GCPCFRTrainer
trainer = GCPCFRTrainer(n_workers=1)
trainer.run_sequential_training(
    iterations_per_scenario=1000,
    stopping_condition_window=50,
    regret_stability_threshold=0.01
)
"
```

## ⚙️ Configuration Options

### Natural CFR Trainer Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `games` | 10000 | 100-100000 | Number of complete games to simulate |
| `epsilon` | 0.1 | 0.0-1.0 | Exploration rate for epsilon-greedy |
| `min_visits` | 5 | 1-100 | Minimum visits before exploitation |
| `tournament_penalty` | 0.2 | 0.1-2.0 | Tournament survival penalty factor |
| `log_interval` | 100 | 10-1000 | Log progress every N games |
| `save_interval` | 1000 | 100-5000 | Save checkpoint every N games |

### GCP Trainer Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_workers` | CPU count | 1-32 | Number of parallel worker processes |
| `total_iterations` | 200000 | 1000-1000000 | Total training iterations across all workers |
| `log_interval_minutes` | 15 | 1-60 | Log progress every N minutes |

### Advanced CFR Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `enable_pruning` | True | True/False | Enable CFR pruning techniques |
| `regret_pruning_threshold` | -300.0 | -1000 to 0 | Threshold for regret-based pruning |
| `strategy_pruning_threshold` | 0.001 | 0.0001-0.01 | Threshold for strategy pruning |

## 📊 Output Files

The system generates several types of output files with timestamps:

### 1. Unified Scenario Lookup Table
**File**: `scenario_lookup_table.csv`
**Updated**: Every logging interval
**Contains**: Real-time scenario analytics with action frequencies

```csv
scenario_key,hand_category,stack_category,blinds_level,position,opponent_action,iterations_completed,total_rollouts,regret,average_strategy,strategy_confidence,fold_pct,call_pct,raise_small_pct,raise_mid_pct,raise_high_pct,is_3bet,last_updated
premium_pairs|BTN|medium|low,premium_pairs,medium,low,BTN,mixed,150,150,0.033333,RAISE,85.2,5.0,10.0,30.0,40.0,15.0,0,2025-08-12 16:16:14
```

**Key Columns:**
- `fold_pct`, `call_pct`, `raise_small_pct`, `raise_mid_pct`, `raise_high_pct`: Action frequency percentages
- `is_3bet`: Binary indicator (1 = 3-bet scenario, 0 = standard)
- `opponent_action`: Opponent context (mixed for aggregated data)
- `strategy_confidence`: Confidence level in the primary strategy

### 2. Training Checkpoints
**Files**: `checkpoints/cfr_checkpoint_YYYYMMDD_HHMMSS.pkl`
**Purpose**: Resume training from interruption points

### 3. Performance Metrics
**Files**: `gcp_cfr_performance_YYYYMMDD_HHMMSS.csv`
**Contains**: Training progress, convergence metrics, system performance

### 4. Natural Game Logs
**Files**: `natural_scenarios_YYYYMMDD_HHMMSS.csv`, `hero_natural_strategies_YYYYMMDD_HHMMSS.csv`
**Contains**: Detailed natural game simulation results

## 🌐 GCP Setup

### 1. Environment Setup
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and authenticate
gcloud init
gcloud auth login
```

### 2. Create Compute Instance
```bash
# Create a high-performance VM
gcloud compute instances create poker-cfr-trainer \
    --zone=us-central1-a \
    --machine-type=c2-standard-16 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --disk=name=poker-data,size=100GB,type=pd-ssd
```

### 3. Install on GCP Instance
```bash
# Connect to instance
gcloud compute ssh poker-cfr-trainer --zone=us-central1-a

# Setup environment
sudo apt update && sudo apt install -y python3-pip git
git clone https://github.com/cemgb1/poker-cfr-ai.git
cd poker-cfr-ai
pip3 install -r requirements.txt
```

### 4. Run Production Training
```bash
# Start large-scale training with screen/tmux for persistence
screen -S poker-training
python3 -c "
from run_gcp_cfr_training import GCPCFRTrainer
trainer = GCPCFRTrainer(n_workers=16, log_interval_minutes=30)
trainer.run_parallel_training(total_iterations=1000000)
"
```

### 5. Monitor Progress
```bash
# Detach from screen: Ctrl+A, D
# Reattach: screen -r poker-training
# Monitor CSV updates:
watch -n 30 "tail -5 scenario_lookup_table.csv"
```

## 📈 Monitoring & Analysis

### Real-time Progress Monitoring
```bash
# Watch the unified scenario table updates
tail -f scenario_lookup_table.csv

# Monitor training logs
tail -f logs/gcp_cfr_training_*.log

# Check system resource usage
htop
```

### Analyzing Results
```python
import pandas as pd

# Load and analyze the scenario lookup table
df = pd.read_csv('scenario_lookup_table.csv')

# View most trained scenarios
print(df.nlargest(10, 'iterations_completed'))

# Analyze action distributions
print("Overall action distribution:")
print(f"Fold: {df['fold_pct'].mean():.1f}%")
print(f"Call: {df['call_pct'].mean():.1f}%") 
print(f"Raise: {(df['raise_small_pct'] + df['raise_mid_pct'] + df['raise_high_pct']).mean():.1f}%")

# 3-bet analysis
threebets = df[df['is_3bet'] == 1]
print(f"3-bet scenarios: {len(threebets)} ({len(threebets)/len(df)*100:.1f}%)")
```

### Key Metrics to Monitor
1. **Scenario Coverage**: Number of unique scenarios discovered
2. **Iteration Distribution**: Balance across hand categories and positions
3. **Strategy Confidence**: Average confidence levels across scenarios
4. **3-bet Frequency**: Percentage of aggressive 3-bet scenarios
5. **Action Percentages**: Realistic fold/call/raise distributions

## 🔧 Advanced Usage

### Custom Scenario Generation
```python
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios
from run_gcp_cfr_training import GCPCFRTrainer

# Generate custom scenario set
scenarios = generate_enhanced_scenarios()
print(f"Generated {len(scenarios)} scenarios")

# Train with custom parameters
trainer = GCPCFRTrainer(n_workers=4)
trainer.scenarios = scenarios  # Use custom scenarios
trainer.run_parallel_training(total_iterations=50000)
```

### Resume from Checkpoint
```python
from run_gcp_cfr_training import GCPCFRTrainer

# Trainer automatically detects and loads recent checkpoints
trainer = GCPCFRTrainer(n_workers=8)
# Will auto-resume if checkpoint < 24 hours old

# Or manually specify checkpoint
trainer.load_checkpoint('checkpoints/cfr_checkpoint_20250812_134909.pkl')
trainer.run_parallel_training(total_iterations=100000)
```

### Export Custom CSV Reports
```python
from run_gcp_cfr_training import GCPCFRTrainer

trainer = GCPCFRTrainer(n_workers=1)
# ... run training ...

# Export custom scenario analysis
df = trainer.export_unified_scenario_lookup_csv('custom_analysis.csv')

# Filter and analyze specific scenarios
premium_scenarios = df[df['hand_category'] == 'premium_pairs']
aggressive_scenarios = df[df['is_3bet'] == 1]
```

## 🐛 Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Reduce worker count if running out of memory
python -c "
from run_gcp_cfr_training import GCPCFRTrainer
trainer = GCPCFRTrainer(n_workers=2)  # Reduce from default
trainer.run_parallel_training(total_iterations=10000)
"
```

#### No Scenarios Generated
```bash
# Check if training is actually running
python test_low_iteration_csv_output.py

# Increase logging frequency for debugging
python run_natural_cfr_training.py --games 50 --log-interval 5
```

#### Checkpoint Loading Issues
```bash
# Clear old checkpoints if corrupted
rm -rf checkpoints/*.pkl

# Start fresh training
python run_natural_cfr_training.py --games 1000
```

### Performance Optimization

#### For Maximum Speed
- Use GCP parallel training with all CPU cores
- Set longer log intervals (15-30 minutes)
- Use SSD storage for checkpoints
- Monitor memory usage and adjust worker count

#### For Maximum Accuracy
- Use Natural CFR training for realistic scenarios
- Lower epsilon for less exploration (0.05-0.1)
- Higher min_visits threshold (10-20)
- Longer training runs (10,000+ games)

### Logs and Debugging
```bash
# Check error logs
tail -f logs/errors.log

# Debug mode for detailed output
python run_natural_cfr_training.py --games 10 --epsilon 0.5 --log-interval 1
```

## 📞 Support

### Test Suite
Run comprehensive tests to verify system health:
```bash
python test_unified_scenario_lookup.py
python test_low_iteration_csv_output.py
python test_natural_cfr.py
```

### Key Files for Support
- `scenario_lookup_table.csv` - Current training state
- `logs/gcp_cfr_training_*.log` - Training logs
- `logs/errors.log` - Error messages
- `checkpoints/` - Training checkpoints

### Performance Validation
```python
# Quick validation that system is working
import pandas as pd
df = pd.read_csv('scenario_lookup_table.csv')
assert len(df) > 0, "No scenarios found"
assert 'fold_pct' in df.columns, "Action percentages missing"
assert 'is_3bet' in df.columns, "3-bet column missing"
print("✅ System validation passed")
```

---

## 🎉 Conclusion

This CFR poker AI training system provides a comprehensive platform for developing advanced poker strategies through:

- **Real-time Analytics**: Live CSV updates with action frequencies
- **Flexible Training**: Multiple modes for different use cases  
- **Production Ready**: GCP deployment with multiprocessing
- **Comprehensive Monitoring**: Detailed logs and performance metrics
- **3-bet Detection**: Binary indicators for aggressive play patterns

The unified scenario lookup table with action frequency percentages and 3-bet indicators provides unprecedented insight into AI learning progress, making this system ideal for both research and production poker AI development.

For additional support or advanced configurations, refer to the test suite and example scripts included in the repository.