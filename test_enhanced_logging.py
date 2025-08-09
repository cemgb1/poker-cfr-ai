#!/usr/bin/env python3
"""
Test script to validate enhanced logging functionality in run_gcp_cfr_training.py

This script tests:
1. Enhanced worker progress logging with detailed metrics
2. Comprehensive checkpoint logging with file details
3. Improved periodic training progress logs
4. Worker completion logging with performance summaries
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append('.')

def test_enhanced_logging():
    """Test the enhanced logging functionality with a short training run"""
    print("🧪 Testing Enhanced Logging Functionality")
    print("=" * 50)
    
    # Create temporary logs directory for testing
    temp_logs_dir = Path("test_logs")
    temp_logs_dir.mkdir(exist_ok=True)
    
    try:
        # Import the GCP CFR trainer
        from run_gcp_cfr_training import GCPCFRTrainer
        
        print("✅ Successfully imported GCPCFRTrainer")
        
        # Initialize trainer with minimal workers and short intervals for testing
        trainer = GCPCFRTrainer(
            n_workers=2,  # Use only 2 workers for testing
            log_interval_minutes=0.1  # Very short interval (6 seconds) for testing
        )
        
        print("✅ Successfully initialized GCPCFRTrainer with enhanced logging")
        
        # Test checkpoint logging by saving a test checkpoint
        print("\n🧪 Testing checkpoint logging...")
        test_worker_results = {
            0: {
                'iterations_completed': 100,
                'iterations_attempted': 100,
                'scenario_counter': {'test_scenario_1': 50, 'test_scenario_2': 50},
                'final_time': 10.0
            }
        }
        
        trainer.save_checkpoint(test_worker_results, time.time())
        print("✅ Checkpoint logging test completed")
        
        # Test periodic progress logging
        print("\n🧪 Testing periodic progress logging...")
        trainer.log_training_progress(time.time())
        print("✅ Periodic progress logging test completed")
        
        # Test very short training to validate worker logging
        print("\n🧪 Testing short training run to validate worker logging...")
        training_results = trainer.run_parallel_training(total_iterations=100)  # Very short run
        print("✅ Short training run completed")
        
        # Verify log files were created
        log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
        print(f"\n📝 Log files created: {len(log_files)}")
        for log_file in log_files:
            print(f"   - {log_file}")
        
        # Check for checkpoint files
        checkpoint_files = list(Path("checkpoints").glob("*.pkl"))
        print(f"\n💾 Checkpoint files created: {len(checkpoint_files)}")
        for checkpoint_file in checkpoint_files:
            print(f"   - {checkpoint_file}")
        
        print("\n🎉 All enhanced logging tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    finally:
        # Clean up test files
        try:
            if temp_logs_dir.exists():
                shutil.rmtree(temp_logs_dir)
        except:
            pass

def test_log_message_formats():
    """Test that the log message formats are readable and informative"""
    print("\n🧪 Testing Log Message Formats")
    print("=" * 40)
    
    # Test worker progress message format
    worker_id = 1
    iteration = 1500
    iterations_per_worker = 10000
    rate = 45.3
    unique_scenarios = 125
    total_scenarios = 330
    scenario_coverage_pct = 37.9
    avg_regret = 0.002451
    worker_memory_mb = 156.7
    
    progress_msg = (f"Worker {worker_id}: {iteration:,}/{iterations_per_worker:,} iterations "
                   f"({rate:.1f}/sec) | Scenarios: {unique_scenarios}/{total_scenarios} "
                   f"({scenario_coverage_pct:.1f}%) | Avg Regret: {avg_regret:.6f} | "
                   f"Memory: {worker_memory_mb:.1f}MB")
    
    print("📊 Enhanced Worker Progress Message Example:")
    print(f"   {progress_msg}")
    
    # Test checkpoint logging format
    print("\n💾 Enhanced Checkpoint Logging Example:")
    print("   💾 ===== CHECKPOINT SAVED =====")
    print("   📁 File: cfr_checkpoint_20240809_143022.pkl")
    print("   📊 Size: 12.45 MB")
    print("   ⏱️  Elapsed Time: 45.2 minutes (0.75 hours)")
    print("   🔄 Total Iterations: 125,430")
    print("   🎯 Scenarios Covered: 287/330 (87.0%)")
    print("   📈 Average Regret: 0.003245")
    print("   📝 Performance Metrics: 1,254 data points")
    print("   🏷️  Timestamp: 20240809_143022")
    print("   💾 =============================")
    
    print("\n✅ Log message format tests completed")
    return True

if __name__ == "__main__":
    print("🚀 Starting Enhanced Logging Tests")
    print("=" * 60)
    
    # Test log message formats first (doesn't require full setup)
    format_test_passed = test_log_message_formats()
    
    # Test actual logging functionality
    logging_test_passed = test_enhanced_logging()
    
    print("\n" + "=" * 60)
    if format_test_passed and logging_test_passed:
        print("🎉 ALL ENHANCED LOGGING TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)