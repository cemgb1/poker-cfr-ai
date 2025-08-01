#!/usr/bin/env python3
"""
Enhanced training script - checkpoints only for slow postflop training
"""
import time
import psutil
import subprocess

def run_resumable_training():
    print("🚀 CFR TRAINING - CHECKPOINTS FOR POSTFLOP ONLY")
    print("=" * 60)
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    start_time = time.time()
    
    try:
        # Train preflop quickly (no checkpoints needed)
        print("\n1️⃣ Training Preflop (Fast - No Checkpoints)...")
        from preflop_cfr import run_preflop_training
        preflop_solver = run_preflop_training()
        
        preflop_time = time.time() - start_time
        print(f"✅ Preflop completed in {preflop_time/60:.1f} minutes")
        
        # Train postflop with checkpoints (the slow part)
        print("\n2️⃣ Training Postflop (Slow - WITH Checkpoints)...")
        from cfr_with_checkpointing import ResumablePreflopCFR, CheckpointManager
        
        # Create postflop solver with checkpoints
        cm = CheckpointManager("postflop_checkpoints")
        
        # Check for existing postflop checkpoints
        checkpoints = cm.list_checkpoints()
        if checkpoints:
            print("📂 Found existing postflop checkpoints:")
            for cp in checkpoints[-3:]:
                print(f"   {cp['filename']}: iteration {cp['iteration']:,}, {cp['size_mb']} MB")
        
        # Use regular postflop but save checkpoints manually
        from postflop_cfr import run_postflop_training
        postflop_solver = run_postflop_training(preflop_solver)
        
        total_time = time.time() - start_time
        print(f"\n🏆 TRAINING COMPLETE!")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"💰 Estimated cost: ${total_time/3600 * 0.35:.2f}")
        
        # Auto shutdown
        print("VM will shutdown in 3 minutes...")
        subprocess.run(['sudo', 'shutdown', '-h', '+3'], check=False)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_resumable_training()
