#!/usr/bin/env python3
"""
Demo script to show the unified scenario lookup table in action.

This demonstrates:
1. Real-time CSV updates during training
2. Unified scenario tracking across workers
3. Tournament penalty of 0.2
"""

import os
import time
from pathlib import Path
from run_gcp_cfr_training import GCPCFRTrainer

def demo_unified_scenario_lookup():
    """Demo the unified scenario lookup table with a short training run."""
    print("🎯 Unified Scenario Lookup Table Demo")
    print("=" * 60)
    
    # Create trainer with fast logging for demo
    trainer = GCPCFRTrainer(n_workers=2, log_interval_minutes=0.1)  # Log every 6 seconds for demo
    
    print(f"📊 Setup:")
    print(f"   🖥️  Workers: {trainer.n_workers}")
    print(f"   📝 Log interval: {trainer.log_interval_minutes} minutes")
    print(f"   🎯 Tournament penalty: 0.2 (default)")
    print(f"   📁 CSV file: scenario_lookup_table.csv")
    
    print(f"\n🚀 Starting short training run (200 iterations total)...")
    
    # Run a very short training session
    total_iterations = 200
    
    try:
        # Monitor CSV file creation
        csv_path = Path("scenario_lookup_table.csv")
        
        print(f"⏰ Training will update CSV every {trainer.log_interval_minutes * 60:.0f} seconds...")
        
        # Start training
        trainer.run_parallel_training(total_iterations=total_iterations)
        
        # Show final results
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"\n📊 Final Results:")
            print(f"   📄 CSV file: {csv_path}")
            print(f"   📋 Scenarios in CSV: {len(df)}")
            print(f"   📈 Total iterations: {df['iterations_completed'].sum()}")
            
            if len(df) > 0:
                print(f"\n🎯 Top 5 Scenarios:")
                for i, (idx, row) in enumerate(df.head().iterrows()):
                    print(f"   {i+1}. {row['hand_category']}|{row['position']}|{row['stack_category']}: "
                          f"{row['iterations_completed']} iterations, {row['average_strategy']} "
                          f"({row['strategy_confidence']:.1f}%)")
            
            print(f"\n✅ Demo completed successfully!")
            print(f"   📄 Check {csv_path} for the unified scenario lookup table")
            
        else:
            print(f"❌ CSV file was not created")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup temporary files
    cleanup_files = ["scenario_lookup_table.csv"]
    for file in cleanup_files:
        if Path(file).exists():
            os.remove(file)
            print(f"🧹 Cleaned up {file}")

if __name__ == "__main__":
    demo_unified_scenario_lookup()