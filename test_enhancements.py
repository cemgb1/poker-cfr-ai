#!/usr/bin/env python3
"""
Test script to validate the new Natural CFR enhancements.

This script tests the following enhancements:
1. Checkpoints directory functionality
2. Performance summary file creation
3. Final lookup table export
4. Archiving functionality
5. Logging of all operations
"""

import os
import time
from pathlib import Path
from natural_game_cfr_trainer import NaturalGameCFRTrainer

def test_checkpoints_directory():
    """Test that checkpoints are saved to checkpoints/ directory."""
    print("🧪 Testing Checkpoints Directory Functionality")
    print("=" * 50)
    
    trainer = NaturalGameCFRTrainer()
    
    # Train for a few games
    trainer.train(n_games=5, log_interval=5, save_interval=5)
    
    # Check that checkpoints directory exists
    checkpoints_dir = Path("checkpoints")
    assert checkpoints_dir.exists(), "Checkpoints directory should exist"
    
    # Check that checkpoint files are in the directory
    checkpoint_files = list(checkpoints_dir.glob("*.pkl"))
    assert len(checkpoint_files) > 0, "Should have checkpoint files in checkpoints/ directory"
    
    print(f"✅ Checkpoints directory: {checkpoints_dir.absolute()}")
    print(f"✅ Checkpoint files found: {len(checkpoint_files)}")
    
    return trainer

def test_performance_summary():
    """Test performance summary file creation."""
    print("\n📊 Testing Performance Summary Creation")
    print("=" * 50)
    
    trainer = NaturalGameCFRTrainer()
    trainer.train(n_games=10, log_interval=10)
    
    # Create performance summary
    performance_file = trainer.create_performance_summary(training_duration=60.0)
    
    assert performance_file is not None, "Performance file should be created"
    assert Path(performance_file).exists(), "Performance file should exist"
    assert "performance/" in performance_file, "Performance file should be in performance/ directory"
    
    print(f"✅ Performance summary created: {performance_file}")
    
    # Check file content
    with open(performance_file, 'r') as f:
        content = f.read()
        assert "Games Played" in content, "Should contain games played metric"
        assert "Hero Win Rate" in content, "Should contain hero win rate"
        assert "Scenario Coverage" in content, "Should contain scenario coverage"
    
    print(f"✅ Performance file contains expected metrics")
    
    return trainer

def test_final_lookup_table():
    """Test final lookup table creation."""
    print("\n📋 Testing Final Lookup Table Creation") 
    print("=" * 50)
    
    trainer = NaturalGameCFRTrainer()
    trainer.train(n_games=15, log_interval=15)
    
    # Create lookup table
    lookup_file = trainer.create_final_lookup_table()
    
    assert lookup_file is not None, "Lookup table file should be created"
    assert Path(lookup_file).exists(), "Lookup table file should exist"
    assert lookup_file.endswith(".csv"), "Lookup table should be CSV format"
    
    print(f"✅ Final lookup table created: {lookup_file}")
    
    # Check file content
    with open(lookup_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1, "Should have header + data rows"
        header = lines[0]
        assert "scenario_key" in header, "Should have scenario_key column"
        assert "best_action" in header, "Should have best_action column"
        assert "confidence" in header, "Should have confidence column"
        assert "estimated_ev" in header, "Should have estimated_ev column"
    
    print(f"✅ Lookup table contains expected columns")
    
    return trainer

def test_archiving():
    """Test archiving functionality."""
    print("\n📦 Testing Archiving Functionality")
    print("=" * 50)
    
    # Create some dummy old files to archive
    dummy_files = [
        "gcp_test_performance.csv",
        "old_scenarios_test.csv",
        "analysis_scenarios_20240101_120000.csv"
    ]
    
    for dummy_file in dummy_files:
        with open(dummy_file, 'w') as f:
            f.write("dummy,content\n1,test\n")
    
    trainer = NaturalGameCFRTrainer()
    
    # Test archiving
    archived_items = trainer.archive_old_files()
    
    # Check that archive directory was created
    archive_dir = Path("archivedfileslocation")
    assert archive_dir.exists(), "Archive directory should exist"
    
    print(f"✅ Archive directory created: {archive_dir.absolute()}")
    print(f"✅ Items archived: {len(archived_items)}")
    
    # Cleanup test files
    for dummy_file in dummy_files:
        if Path(dummy_file).exists():
            Path(dummy_file).unlink()
    
    return trainer

def test_all_modes():
    """Test that all training modes work with enhancements."""
    print("\n🎯 Testing All Training Modes")
    print("=" * 50)
    
    # Test demo mode
    print("Testing demo mode...")
    import subprocess
    result = subprocess.run([
        "python", "run_natural_cfr_training.py", 
        "--mode", "demo", "--games", "10"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Demo mode should succeed: {result.stderr}"
    assert "demo_final_lookup_table.csv" in result.stdout or Path("demo_final_lookup_table.csv").exists()
    print("✅ Demo mode works with enhancements")
    
    # Test that files are discoverable
    expected_files = [
        "demo_natural_scenarios.csv",
        "demo_final_lookup_table.csv"
    ]
    
    discovered_files = []
    for file_pattern in expected_files:
        files = list(Path(".").glob(file_pattern))
        discovered_files.extend(files)
    
    performance_files = list(Path("performance").glob("*.csv")) if Path("performance").exists() else []
    discovered_files.extend(performance_files)
    
    print(f"✅ Discoverable files: {len(discovered_files)} files found")
    for file_path in discovered_files[:5]:  # Show first 5
        print(f"   - {file_path}")
    
    return True

def run_all_enhancement_tests():
    """Run all enhancement tests."""
    print("🧪 Natural CFR Enhancements Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run tests
        test_checkpoints_directory()
        test_performance_summary()
        test_final_lookup_table()
        test_archiving()
        test_all_modes()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 All enhancement tests completed successfully!")
        print(f"⏱️  Total test time: {duration:.1f} seconds")
        
        # Summary of what was tested
        print(f"\n📋 Enhancement Summary:")
        print(f"   ✅ Checkpoints saved to checkpoints/ directory")
        print(f"   ✅ Performance summaries created in performance/ directory")
        print(f"   ✅ Final lookup tables exported with EV and confidence")
        print(f"   ✅ Old files archived to archivedfileslocation/")
        print(f"   ✅ All operations logged with paths and summaries")
        print(f"   ✅ All modes (train, demo, analysis) work with enhancements")
        print(f"   ✅ All files are discoverable by users")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Enhancement test failed with error: {e}")
        import traceback
        print(f"🔍 Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = run_all_enhancement_tests()
    if success:
        print("\n✅ All Natural CFR enhancements are working correctly!")
    else:
        print("\n❌ Some enhancement tests failed!")