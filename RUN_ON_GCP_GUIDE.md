# GCP Deployment Guide - Natural Game CFR Trainer

A comprehensive guide for deploying and running the Natural Game CFR trainer on Google Cloud Platform (GCP).

## 🎯 Overview

This guide covers how to set up and run the Natural Game CFR training system on GCP virtual machines for large-scale poker AI training. The system uses natural Monte Carlo game simulation with co-evolving strategies for realistic poker AI development.

## 📋 Table of Contents

1. [GCP VM Setup](#gcp-vm-setup)
2. [Repository Setup](#repository-setup)
3. [Python Environment Setup](#python-environment-setup)
4. [Running Training](#running-training)
5. [Configuration Options](#configuration-options)
6. [Monitoring Training](#monitoring-training)
7. [Resuming Training](#resuming-training)
8. [File Management](#file-management)
9. [Troubleshooting](#troubleshooting)

## 🌐 GCP VM Setup

### 1. Create a GCP VM Instance

```bash
# Create a high-performance VM for training
gcloud compute instances create poker-cfr-trainer \
    --zone=us-central1-a \
    --machine-type=c2-standard-16 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=MIGRATE
```

### 2. Connect to the VM

```bash
gcloud compute ssh poker-cfr-trainer --zone=us-central1-a
```

### 3. Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git htop screen
```

## 📦 Repository Setup

### 1. Remove Previous Installations

If you have previous clones, clean them up first:

```bash
# Remove any existing poker-cfr-ai directories
rm -rf ~/poker-cfr-ai
rm -rf ~/poker-cfr-ai-*

# Clean up any old virtual environments
rm -rf ~/venv-poker-cfr
rm -rf ~/.local/share/virtualenvs/poker-cfr-*
```

### 2. Clone Fresh Repository

```bash
# Clone the repository
cd ~
git clone https://github.com/cemgb1/poker-cfr-ai.git
cd poker-cfr-ai
```

### 3. Verify Repository Contents

```bash
# Check that essential files are present
ls -la
# Should see: run_natural_cfr_training.py, natural_game_cfr_trainer.py, requirements.txt, etc.
```

## 🐍 Python Environment Setup

### 1. Create and Activate Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv ~/venv-poker-cfr

# Activate the virtual environment
source ~/venv-poker-cfr/bin/activate

# Verify activation (should show virtual env path)
which python
```

### 2. Install Requirements

```bash
# Ensure you're in the project directory and virtual environment is active
cd ~/poker-cfr-ai
source ~/venv-poker-cfr/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, pandas, treys; print('All dependencies installed successfully')"
```

## 🚀 Running Training

### 1. Test Installation

First, verify everything works with a quick demo:

```bash
# Activate virtual environment
source ~/venv-poker-cfr/bin/activate
cd ~/poker-cfr-ai

# Run a quick demo (1000 games)
python run_natural_cfr_training.py --mode demo --games 1000
```

### 2. Running in Background with nohup

For long-running training sessions, use `nohup` to run in the background:

```bash
# Activate virtual environment
source ~/venv-poker-cfr/bin/activate
cd ~/poker-cfr-ai

# Run training in background with nohup
nohup python run_natural_cfr_training.py \
    --games 50000 \
    --workers 8 \
    --epsilon 0.1 \
    --save-interval 1000 \
    --log-interval 100 \
    > training_output.log 2>&1 &

# Get the process ID
echo $! > training_pid.txt
echo "Training started with PID: $(cat training_pid.txt)"
```

### 3. Alternative: Using Screen Sessions

```bash
# Start a screen session
screen -S poker-training

# Inside screen, activate environment and run training
source ~/venv-poker-cfr/bin/activate
cd ~/poker-cfr-ai
python run_natural_cfr_training.py --games 50000 --workers 8

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r poker-training
```

## ⚙️ Configuration Options

### Base Configuration

The script accepts many configuration options. Here are the key parameters:

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | train | Training mode: `train`, `demo`, or `analysis` |
| `--games` | 10000 | Number of complete games to simulate |
| `--workers` | 8 | Number of parallel workers |
| `--epsilon` | 0.1 | Exploration rate (0.0-1.0) |
| `--min-visits` | 5 | Minimum visits before scenario is considered trained |
| `--tournament-penalty` | 0.2 | Tournament survival penalty factor |

#### CFR Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable-pruning` | True | Enable CFR pruning for efficiency |
| `--regret-threshold` | -300.0 | Regret pruning threshold |
| `--strategy-threshold` | 0.001 | Strategy pruning threshold |

#### Logging and Saving

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--save-interval` | 1000 | Save checkpoint every N games |
| `--log-interval` | 100 | Log progress every N games |
| `--resume` | None | Resume from checkpoint file |

### Configuration Examples

#### Large-Scale Production Training
```bash
python run_natural_cfr_training.py \
    --games 100000 \
    --workers 16 \
    --epsilon 0.05 \
    --save-interval 2000 \
    --log-interval 500 \
    --tournament-penalty 0.2
```

#### High Exploration Training
```bash
python run_natural_cfr_training.py \
    --games 50000 \
    --epsilon 0.2 \
    --min-visits 3 \
    --workers 8
```

#### Conservative Training (Lower Risk)
```bash
python run_natural_cfr_training.py \
    --games 50000 \
    --tournament-penalty 0.4 \
    --epsilon 0.05 \
    --workers 8
```

#### Single-Process Training
```bash
python run_natural_cfr_training.py \
    --games 20000 \
    --workers 1 \
    --epsilon 0.1
```

## 📊 Monitoring Training

### 1. Check Training Progress

```bash
# Monitor the output log
tail -f training_output.log

# Check last 50 lines
tail -50 training_output.log

# Search for specific information
grep "Training completed" training_output.log
grep "Games per minute" training_output.log
```

### 2. Monitor System Resources

```bash
# Check CPU and memory usage
htop

# Check process status
ps aux | grep python

# Monitor disk usage
df -h
du -sh ~/poker-cfr-ai/
```

### 3. Check Generated Files

```bash
# List generated files
ls -lt *.csv *.pkl

# Check checkpoints directory
ls -la checkpoints/

# Monitor file sizes
ls -lh natural_scenarios_*.csv
ls -lh hero_natural_strategies_*.csv
ls -lh villain_natural_strategies_*.csv
```

## 🔄 Resuming Training

### 1. Auto-Resume from Latest Checkpoint

The script automatically finds the latest checkpoint:

```bash
# Script will auto-detect latest checkpoint
python run_natural_cfr_training.py --games 20000 --workers 8
```

### 2. Resume from Specific Checkpoint

```bash
# Resume from a specific checkpoint file
python run_natural_cfr_training.py \
    --resume checkpoints/natural_cfr_final_20241203_143022.pkl \
    --games 10000 \
    --workers 8
```

### 3. Resume After Interruption

If training was interrupted:

```bash
# Check if process is still running
ps aux | grep "run_natural_cfr_training"

# Kill if necessary
kill $(cat training_pid.txt)

# Resume training
nohup python run_natural_cfr_training.py \
    --games 30000 \
    --workers 8 \
    > training_resume.log 2>&1 &
```

## 📁 File Management

### Generated Files

The training system generates several types of files:

#### Scenario Files
- `natural_scenarios_TIMESTAMP.csv` - Natural scenarios discovered during training
- `scenario_lookup_table.csv` - Real-time scenario lookup table

#### Strategy Files  
- `hero_natural_strategies_TIMESTAMP.csv` - Hero's learned strategies
- `villain_natural_strategies_TIMESTAMP.csv` - Villain's learned strategies

#### Checkpoint Files
- `checkpoints/natural_cfr_final_TIMESTAMP.pkl` - Complete training state
- `checkpoints/natural_cfr_emergency_TIMESTAMP.pkl` - Emergency saves

#### Performance Files
- `performance/performance_summary_TIMESTAMP.csv` - Training performance metrics

### Archiving Old Files

```bash
# Create archive directory
mkdir -p ~/poker-cfr-archive

# Move old training files
mv natural_scenarios_*.csv ~/poker-cfr-archive/
mv hero_natural_strategies_*.csv ~/poker-cfr-archive/
mv villain_natural_strategies_*.csv ~/poker-cfr-archive/

# Keep only recent checkpoints (last 3)
cd checkpoints/
ls -t *.pkl | tail -n +4 | xargs -I {} mv {} ~/poker-cfr-archive/
```

### Downloading Results

```bash
# Compress results for download
tar -czf training_results_$(date +%Y%m%d).tar.gz \
    natural_scenarios_*.csv \
    hero_natural_strategies_*.csv \
    villain_natural_strategies_*.csv \
    checkpoints/ \
    performance/

# Download using gcloud
gcloud compute scp poker-cfr-trainer:~/poker-cfr-ai/training_results_*.tar.gz . --zone=us-central1-a
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Monitor memory usage
free -h
# If memory is low, reduce workers:
python run_natural_cfr_training.py --games 20000 --workers 4
```

#### 2. Disk Space Issues
```bash
# Check disk space
df -h
# Clean up old files if needed
rm -f old_scenarios_*.csv
rm -f old_strategies_*.csv
```

#### 3. Process Hanging
```bash
# Check if process is responsive
ps aux | grep python
# Kill and restart if necessary
kill -9 $(cat training_pid.txt)
```

#### 4. Import Errors
```bash
# Reinstall requirements
source ~/venv-poker-cfr/bin/activate
pip install --force-reinstall -r requirements.txt
```

#### 5. Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf ~/venv-poker-cfr
python3 -m venv ~/venv-poker-cfr
source ~/venv-poker-cfr/bin/activate
pip install -r requirements.txt
```

### Performance Optimization

#### 1. VM Instance Optimization
- Use `c2-standard-16` or higher for CPU-intensive training
- Consider `n2-highmem` instances for memory-intensive scenarios
- Use SSD persistent disks for faster I/O

#### 2. Training Parameter Tuning
- Adjust `--workers` based on CPU cores (usually CPU cores - 1)
- Increase `--save-interval` and `--log-interval` for faster training
- Use lower `--epsilon` for exploitation-focused training

#### 3. Monitoring Performance
```bash
# Check games per minute in logs
grep "games per minute" training_output.log | tail -5

# Monitor CPU utilization
top -p $(cat training_pid.txt)
```

### Emergency Recovery

#### If VM is Terminated
```bash
# Recreate VM with same name and zone
gcloud compute instances create poker-cfr-trainer \
    --zone=us-central1-a \
    --machine-type=c2-standard-16 \
    --boot-disk-size=50GB

# Follow setup steps again and resume from checkpoint
```

#### If Training Corrupts
```bash
# Start fresh but resume from an older checkpoint
python run_natural_cfr_training.py \
    --resume checkpoints/natural_cfr_final_OLDER_TIMESTAMP.pkl \
    --games 10000
```

## 📈 Best Practices

### 1. Training Strategy
- Start with smaller `--games` values (10,000-20,000) to verify setup
- Use demo mode for quick verification
- Run analysis mode on checkpoints to understand progress

### 2. Resource Management
- Monitor disk space regularly
- Archive old files periodically
- Use appropriate VM sizing for your training load

### 3. Backup Strategy
- Regular checkpoint saves (default: every 1000 games)
- Download important results periodically
- Keep multiple checkpoint generations

### 4. Cost Optimization
- Use preemptible instances for non-critical training
- Stop VMs when not in use
- Monitor billing regularly

This guide provides comprehensive coverage for running the Natural Game CFR trainer on GCP. For additional questions or issues, refer to the main README_NATURAL_CFR.md file or the script's help output.