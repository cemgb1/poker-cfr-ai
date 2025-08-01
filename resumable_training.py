# Remove the broken file and create a clean one
rm resumable_training.py

# Create clean resumable_training.py
cat > resumable_training.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced training script with checkpointing support
"""
import time
import psutil
import subprocess

def run_resumable_training():
    print("🚀 RESUMABLE CFR TRAINING WITH CHECKPOINTS")
    print("=" * 60)
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    start_time = time.time()
    
    try:
        # Import the checkpointing classes
        from cfr_with_checkpointing import ResumablePreflopCFR, CheckpointManager
        
        # Check for existing checkpoints
        cm = CheckpointManager()
        checkpoints = cm.list_checkpoints()
        
        if checkpoints:
            print("📂 Found existing checkpoints:")
            for cp in checkpoints[-3:]:  # Show last 3
                print(f"   {cp['filename']}: iteration {cp['iteration']:,}, {cp['size_mb']} MB")
            print("\nTo resume, run: python resumable_training.py resume")
        
        # Train preflop with checkpoints
        print("\n1️⃣ Training Preflop with Checkpoints...")
        preflop_solver = ResumablePreflopCFR(checkpoint_manager=cm)
        preflop_solver.train_with_checkpoints(
            total_iterations=300000,
            min_visits_per_scenario=300,
            checkpoint_every=500  # Save every 500 iterations
        )
        
        # For now, use regular postflop (until we add resumable postflop)
        print("\n2️⃣ Training Postflop (Regular - No Checkpoints Yet)...")
        from postflop_cfr import run_postflop_training
        postflop_solver = run_postflop_training(preflop_solver)
        
        total_time = time.time() - start_time
        print(f"\n🏆 TRAINING COMPLETE!")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"💰 Estimated cost: ${total_time/3600 * 0.35:.2f}")
        print(f"📁 Checkpoints saved in: ./checkpoints/")
        
        # Auto shutdown
        print("VM will shutdown in 3 minutes...")
        subprocess.run(['sudo', 'shutdown', '-h', '+3'], check=False)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        print("Checkpoints preserved for resume")

def resume_from_checkpoint():
    """Resume training from latest checkpoint"""
    print("🔄 RESUMING FROM CHECKPOINT")
    print("=" * 40)
    
    from cfr_with_checkpointing import ResumablePreflopCFR, CheckpointManager
    
    cm = CheckpointManager()
    checkpoints = cm.list_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    latest = checkpoints[-1]
    print(f"📂 Resuming from: {latest['filename']}")
    print(f"   Iteration: {latest['iteration']:,}")
    print(f"   Scenarios trained: {latest['scenarios_trained']:,}")
    
    # Resume training
    preflop_solver = ResumablePreflopCFR(checkpoint_manager=cm)
    preflop_solver.train_with_checkpoints(
        total_iterations=300000,
        resume_from=latest['path'],
        checkpoint_every=500
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        resume_from_checkpoint()
    else:
        run_resumable_training()
EOF

# Make it executable
chmod +x resumable_training.py

# Test that it works
python -c "import resumable_training; print('✅ resumable_training.py syntax OK')"

# Start training
python resumable_training.py 2>&1 | tee training_with_checkpoints.log
