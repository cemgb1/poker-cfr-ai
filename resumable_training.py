#!/usr/bin/env python3
"""
Enhanced training script with checkpointing support
"""

import time
import psutil
import os
from cfr_with_checkpointing import ResumableTrainingOrchestrator

def main():
    print("🚀 RESUMABLE CFR TRAINING WITH CHECKPOINTS")
    print("=" * 60)
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    start_time = time.time()
    
    # Create orchestrator
    orchestrator = ResumableTrainingOrchestrator()
    
    try:
        # Run training with automatic checkpointing
        orchestrator.run_full_training_with_checkpoints(
            checkpoint_every=20000  # Save every 20k iterations
        )
        
        total_time = time.time() - start_time
        print(f"\n🏆 TRAINING COMPLETE!")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"💰 Estimated cost: ${total_time/3600 * 0.35:.2f}")
        print(f"📁 Checkpoints saved in: ./checkpoints/")
        
        # Auto shutdown
        import subprocess
        subprocess.run(['sudo', 'shutdown', '-h', '+3'], check=False)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Checkpoints preserved for resume")

if __name__ == "__main__":
    main()
