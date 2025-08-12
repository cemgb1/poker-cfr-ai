# GCP Monte Carlo Natural CFR Training Guide

This guide explains how to run the multi-core Monte Carlo Natural CFR trainer on Google Cloud Platform (GCP) with automatic checkpoint/resume functionality.

## Overview

The Natural CFR training system uses a game-based simulation structure where each "game" consists of multiple poker hands played with fixed stack and blind sizes until one player is busted. The system supports multi-core parallel processing, allowing you to distribute training across multiple CPU cores for faster convergence. Each worker processes an equal share of games independently and saves its own results.

### Key Concepts:
- **Games**: Complete poker sessions from equal stacks to bust (primary simulation unit)
- **Hands**: Individual poker hands within a game (preflop-only currently)  
- **Fixed Parameters**: Stack size and blinds remain constant within each game
- **Variable Elements**: Cards and positions randomized per hand within game

## Quick Start

### Basic Multi-Core Training
```bash
# Run 50,000 games across 8 workers (default) - each game plays multiple hands until bust
python run_natural_cfr_training.py --games 50000

# Run with custom worker count - distributes games across workers
python run_natural_cfr_training.py --games 100000 --workers 16

# Single-process mode (backward compatible)
python run_natural_cfr_training.py --games 10000 --workers 1
```

### Auto-Resume Training
The system automatically searches for the latest checkpoint and resumes training:

```bash
# Auto-resume from latest checkpoint (no --resume needed)
python run_natural_cfr_training.py --games 50000 --workers 8

# Manual resume from specific checkpoint
python run_natural_cfr_training.py --resume checkpoints/my_checkpoint.pkl --games 20000 --workers 4
```

## GCP Setup

### 1. Create a GCP Compute Instance

**Recommended Configuration:**
- **Machine Type**: `c2-standard-16` (16 vCPUs, 64GB RAM)
- **Boot Disk**: Ubuntu 20.04 LTS, 100GB SSD
- **Zone**: Choose based on your location (e.g., `us-central1-a`)

```bash
# Create instance via gcloud CLI
gcloud compute instances create poker-cfr-trainer \
    --zone=us-central1-a \
    --machine-type=c2-standard-16 \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

### 2. Connect and Setup Environment

```bash
# SSH into your instance
gcloud compute ssh poker-cfr-trainer --zone=us-central1-a

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip git

# Clone repository
git clone https://github.com/cemgb1/poker-cfr-ai.git
cd poker-cfr-ai

# Install Python dependencies
pip3 install -r requirements.txt
```

### 3. Configure Training Parameters

Create a training script for your specific needs:

```bash
# create training_config.sh
cat > training_config.sh << 'EOF'
#!/bin/bash

# Training configuration
GAMES=1000000        # 1M games total
WORKERS=16           # Use all 16 vCPUs
SAVE_INTERVAL=5000   # Save every 5K games
LOG_INTERVAL=1000    # Log every 1K games

# Run training with auto-resume
python3 run_natural_cfr_training.py \
    --games $GAMES \
    --workers $WORKERS \
    --save-interval $SAVE_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --epsilon 0.05 \
    --tournament-penalty 0.4

echo "Training completed at $(date)"
EOF

chmod +x training_config.sh
```

## Checkpoint and Resume Logic

### Automatic Checkpoint Discovery

The system automatically finds and resumes from the latest checkpoint:

1. **Search Order**: 
   - `checkpoints/*.pkl` (checkpoints directory first)
   - `*.pkl` (repository root second)

2. **Selection**: Latest file by modification time

3. **Resume**: Only worker 0 loads the checkpoint in multi-worker mode

### Manual Checkpoint Management

```bash
# List available checkpoints
ls -lat checkpoints/

# Resume from specific checkpoint
python run_natural_cfr_training.py --resume checkpoints/natural_cfr_final_20240101_120000.pkl --games 50000

# Create emergency checkpoint during training (Ctrl+C)
# System automatically saves: natural_cfr_emergency_TIMESTAMP.pkl
```

## Multi-Worker Output Files

Each worker generates its own set of output files:

### Per-Worker Files
```
worker_0_natural_scenarios_TIMESTAMP.csv       # Worker 0 scenarios
hero_worker_0_natural_strategies_TIMESTAMP.csv # Worker 0 hero strategies  
villain_worker_0_natural_strategies_TIMESTAMP.csv # Worker 0 villain strategies
checkpoints/worker_0_natural_cfr_TIMESTAMP.pkl # Worker 0 checkpoint

worker_1_natural_scenarios_TIMESTAMP.csv       # Worker 1 scenarios
# ... and so on for each worker
```

### Aggregated Results

At completion, the system provides:
- **Summary statistics** across all workers
- **Total scenario coverage**
- **Combined training metrics**
- **List of all worker output files**

## Performance Optimization

### Choosing Worker Count

```bash
# Check available CPU cores
nproc

# Rule of thumb: Use 1 worker per CPU core
# For c2-standard-16: --workers 16
# For c2-standard-8: --workers 8
```

### Memory Considerations

Each worker requires approximately 2-4GB RAM:
- **8 workers**: 16-32GB RAM recommended
- **16 workers**: 32-64GB RAM recommended

### Storage Requirements

- **Scenarios**: ~1MB per 1000 games per worker
- **Strategies**: ~500KB per worker
- **Checkpoints**: ~5MB per worker
- **Total**: Plan for ~50-100MB per 10K games with 8 workers

## Monitoring and Logs

### Real-time Monitoring

```bash
# Watch training progress
tail -f logs/natural_cfr_training_*.log

# Monitor all workers
tail -f logs/natural_cfr_worker_*.log

# Check system resources
htop
```

### Log Files

- `logs/natural_cfr_main_TIMESTAMP.log` - Main process log
- `logs/natural_cfr_training_TIMESTAMP.log` - Training coordinator log
- `logs/natural_cfr_worker_N_TIMESTAMP.log` - Worker N logs
- `logs/natural_cfr_aggregator_TIMESTAMP.log` - Results aggregation log

## Production Training Workflow

### Long-Running Training

```bash
# Use tmux for persistent sessions
sudo apt install tmux
tmux new-session -d -s poker-training

# Attach to session
tmux attach-session -t poker-training

# Run training inside tmux
./training_config.sh

# Detach: Ctrl+B, then D
# Training continues in background
```

### Batch Training Script

```bash
# Create comprehensive training script
cat > run_production_training.sh << 'EOF'
#!/bin/bash

set -e  # Exit on error

echo "Starting poker CFR training at $(date)"

# Environment check
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $4}')"

# Training parameters
TOTAL_GAMES=5000000
WORKERS=$(nproc)
BATCH_SIZE=100000

echo "Training config: $TOTAL_GAMES games, $WORKERS workers, $BATCH_SIZE batch size"

# Run training in batches for better checkpointing
REMAINING_GAMES=$TOTAL_GAMES
BATCH_NUM=1

while [ $REMAINING_GAMES -gt 0 ]; do
    CURRENT_BATCH=$((REMAINING_GAMES > BATCH_SIZE ? BATCH_SIZE : REMAINING_GAMES))
    
    echo "Batch $BATCH_NUM: Running $CURRENT_BATCH games..."
    
    python3 run_natural_cfr_training.py \
        --games $CURRENT_BATCH \
        --workers $WORKERS \
        --save-interval 5000 \
        --log-interval 1000 \
        --epsilon 0.03 \
        --tournament-penalty 0.35
    
    REMAINING_GAMES=$((REMAINING_GAMES - CURRENT_BATCH))
    BATCH_NUM=$((BATCH_NUM + 1))
    
    echo "Batch completed. Remaining: $REMAINING_GAMES games"
    sleep 5  # Brief pause between batches
done

echo "All training completed at $(date)"

# Create summary
echo "Final results summary:"
ls -la worker_*_natural_scenarios_*.csv | wc -l | xargs echo "Scenario files:"
ls -la hero_worker_*_natural_strategies_*.csv | wc -l | xargs echo "Strategy files:"
ls -la checkpoints/ | wc -l | xargs echo "Checkpoints:"

EOF

chmod +x run_production_training.sh
```

## Cost Optimization

### Preemptible Instances

Use preemptible instances for 60-90% cost savings:

```bash
gcloud compute instances create poker-cfr-trainer-preempt \
    --zone=us-central1-a \
    --machine-type=c2-standard-16 \
    --preemptible \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud
```

**Note**: Preemptible instances can be terminated. The auto-resume feature makes this viable for long training runs.

### Storage Management

```bash
# Compress old results periodically
find . -name "worker_*_natural_*.csv" -mtime +7 -exec gzip {} \;

# Archive old checkpoints
mkdir -p archived_checkpoints
find checkpoints/ -name "*.pkl" -mtime +3 -exec mv {} archived_checkpoints/ \;
```

## Troubleshooting

### Common Issues

**1. Memory Errors**
```bash
# Reduce workers if memory is insufficient
python run_natural_cfr_training.py --games 50000 --workers 4
```

**2. Disk Space**
```bash
# Clean up old files
rm worker_*_natural_*.csv
rm hero_worker_*.csv villain_worker_*.csv
```

**3. Worker Failures**
- Check individual worker logs: `logs/natural_cfr_worker_N_*.log`
- Reduce worker count to isolate issues
- Verify sufficient memory per worker

**4. Checkpoint Issues**
```bash
# Manually specify checkpoint if auto-resume fails
python run_natural_cfr_training.py --resume checkpoints/specific_file.pkl --games 10000
```

### Performance Debugging

```bash
# Monitor CPU usage per worker
htop -p $(pgrep -f "natural_cfr_worker")

# Check I/O bottlenecks
iotop

# Memory usage by worker
ps aux | grep natural_cfr_worker | awk '{print $2, $4, $11}' | sort -k2 -nr
```

## Example Training Sessions

### Small Scale Test (15 minutes)
```bash
python run_natural_cfr_training.py --games 10000 --workers 4 --log-interval 500
```

### Medium Scale Training (2-4 hours)
```bash
python run_natural_cfr_training.py --games 200000 --workers 8 --save-interval 5000
```

### Large Scale Production (12-24 hours)
```bash
python run_natural_cfr_training.py --games 2000000 --workers 16 --save-interval 10000 --epsilon 0.02
```

## Results Analysis

After training completion, analyze results across all workers:

```bash
# Count total scenarios across all workers
wc -l worker_*_natural_scenarios_*.csv

# Combine all worker scenarios for analysis
head -1 worker_0_natural_scenarios_*.csv > combined_scenarios.csv
tail -n +2 -q worker_*_natural_scenarios_*.csv >> combined_scenarios.csv

# Strategy analysis across workers
grep -h "FOLD" hero_worker_*_natural_strategies_*.csv | wc -l
grep -h "RAISE" hero_worker_*_natural_strategies_*.csv | wc -l
```

This guide provides a complete framework for running scalable poker CFR training on GCP with automatic resume capabilities and multi-worker parallelization.