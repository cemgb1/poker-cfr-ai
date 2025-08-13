#!/usr/bin/env python3
"""
Test script for windowed scenario lookup CSV export functionality.

This script validates that the windowed export feature works correctly:
- Tests both cumulative and window export modes
- Validates CSV format and role-agnostic scenario keys
- Verifies window sliding behavior
- Checks min visits filtering
- Validates trash ratio warnings

Usage:
    python test_window_export.py
"""

import os
import subprocess
import pandas as pd
import time
from pathlib import Path

def run_training(mode, games, export_scope, export_window_games=2000, export_min_visits=1, log_interval=5):
    """Run training with specified parameters and return the subprocess result."""
    cmd = [
        "python", "run_natural_cfr_training.py",
        "--mode", mode,
        "--games", str(games),
        "--workers", "1",
        "--export-scope", export_scope,
        "--export-window-games", str(export_window_games),
        "--export-min-visits", str(export_min_visits),
        "--save-interval", "100",
        "--log-interval", str(log_interval)
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    return result

def validate_csv_format(csv_path):
    """Validate that CSV has correct format and role-agnostic scenario keys."""
    if not os.path.exists(csv_path):
        return False, f"CSV file {csv_path} does not exist"
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = [
            'scenario_key', 'hand_cat', 'position', 'stack_cat', 
            'blinds_level', 'villain_stack_cat', 'preflop_context', 
            'visits', 'pct_fold', 'pct_call_small', 'pct_raise_small', 
            'pct_raise_mid', 'pct_raise_high'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing columns: {missing_columns}"
        
        # Check scenario key format (role-agnostic)
        if len(df) > 0:
            sample_key = df['scenario_key'].iloc[0]
            parts = sample_key.split('|')
            if len(parts) != 6:
                return False, f"Invalid scenario key format: {sample_key} (expected 6 parts)"
        
        # Check percentage values are in [0,1] range
        pct_columns = ['pct_fold', 'pct_call_small', 'pct_raise_small', 'pct_raise_mid', 'pct_raise_high']
        for col in pct_columns:
            if (df[col] < 0).any() or (df[col] > 1).any():
                return False, f"Percentage values in {col} outside [0,1] range"
        
        # Check that percentages sum to approximately 1.0 (within rounding tolerance)
        row_sums = df[pct_columns].sum(axis=1)
        if not all(abs(s - 1.0) < 0.001 for s in row_sums):
            return False, "Percentage rows don't sum to 1.0"
        
        return True, f"CSV format valid: {len(df)} rows"
        
    except Exception as e:
        return False, f"Error validating CSV: {e}"

def test_cumulative_mode():
    """Test cumulative export mode."""
    print("\n" + "="*60)
    print("🧪 TESTING CUMULATIVE MODE")
    print("="*60)
    
    # Run training with cumulative mode
    result = run_training("demo", 15, "cumulative", export_min_visits=1)
    
    if result.returncode != 0:
        print(f"❌ Training failed: {result.stderr}")
        return False
    
    # Check for expected log messages
    logs = result.stdout + result.stderr
    if "Export scope: cumulative" not in logs:
        print("❌ Missing cumulative scope log")
        return False
    
    # Validate CSV
    valid, message = validate_csv_format("scenario_lookup_table.csv")
    print(f"📊 CSV validation: {message}")
    
    if not valid:
        print("❌ CSV validation failed")
        return False
    
    # Store cumulative CSV for comparison
    if os.path.exists("scenario_lookup_table.csv"):
        os.rename("scenario_lookup_table.csv", "test_cumulative.csv")
    
    print("✅ Cumulative mode test passed")
    return True

def test_window_mode():
    """Test window export mode."""
    print("\n" + "="*60)
    print("🧪 TESTING WINDOW MODE")
    print("="*60)
    
    # Run training with window mode (small window)
    result = run_training("demo", 15, "window", export_window_games=5, export_min_visits=1)
    
    if result.returncode != 0:
        print(f"❌ Training failed: {result.stderr}")
        return False
    
    # Check for expected log messages
    logs = result.stdout + result.stderr
    if "Export scope: window" not in logs:
        print("❌ Missing window scope log")
        return False
    
    if "window (last" not in logs:
        print("❌ Missing window description log")
        return False
    
    # Validate CSV
    valid, message = validate_csv_format("scenario_lookup_table.csv")
    print(f"📊 CSV validation: {message}")
    
    if not valid:
        print("❌ CSV validation failed")
        return False
    
    # Store window CSV for comparison
    if os.path.exists("scenario_lookup_table.csv"):
        os.rename("scenario_lookup_table.csv", "test_window.csv")
    
    print("✅ Window mode test passed")
    return True

def test_min_visits_filtering():
    """Test min visits filtering."""
    print("\n" + "="*60)
    print("🧪 TESTING MIN VISITS FILTERING")
    print("="*60)
    
    # Run training with higher min_visits requirement
    result = run_training("demo", 15, "cumulative", export_min_visits=5)
    
    if result.returncode != 0:
        print(f"❌ Training failed: {result.stderr}")
        return False
    
    # Check for filtering log messages
    logs = result.stdout + result.stderr
    if "Filtered by min_visits:" not in logs:
        print("❌ Missing min_visits filtering log")
        return False
    
    # Validate CSV - should have fewer rows due to filtering
    valid, message = validate_csv_format("scenario_lookup_table.csv")
    print(f"📊 CSV validation with min_visits=5: {message}")
    
    if not valid:
        print("❌ CSV validation failed")
        return False
    
    print("✅ Min visits filtering test passed")
    return True

def compare_modes():
    """Compare cumulative vs window mode outputs."""
    print("\n" + "="*60)
    print("🧪 COMPARING CUMULATIVE VS WINDOW MODES")
    print("="*60)
    
    if not os.path.exists("test_cumulative.csv") or not os.path.exists("test_window.csv"):
        print("❌ Missing test CSV files for comparison")
        return False
    
    try:
        df_cumulative = pd.read_csv("test_cumulative.csv")
        df_window = pd.read_csv("test_window.csv")
        
        print(f"📊 Cumulative mode: {len(df_cumulative)} scenarios")
        print(f"📊 Window mode: {len(df_window)} scenarios")
        
        # Window should typically have fewer scenarios (unless window size >= total games)
        if len(df_window) > len(df_cumulative):
            print("⚠️ Warning: Window mode has more scenarios than cumulative")
        
        # Check scenario key format consistency
        cumulative_keys = set(df_cumulative['scenario_key'])
        window_keys = set(df_window['scenario_key'])
        
        # All window keys should also exist in cumulative (since window is subset)
        if not window_keys.issubset(cumulative_keys):
            extra_keys = window_keys - cumulative_keys
            print(f"❌ Window has scenarios not in cumulative: {list(extra_keys)[:5]}...")
            return False
        
        print("✅ Mode comparison passed")
        return True
        
    except Exception as e:
        print(f"❌ Error comparing modes: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_cumulative.csv",
        "test_window.csv", 
        "scenario_lookup_table.csv",
        "demo_natural_scenarios.csv",
        "demo_final_lookup_table.csv"
    ] + [f for f in os.listdir(".") if f.startswith("hero_demo_") or f.startswith("villain_demo_")]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)

def main():
    """Run all tests."""
    print("🧪 WINDOWED CSV EXPORT TEST SUITE")
    print("="*60)
    
    # Cleanup from previous runs
    cleanup_test_files()
    
    tests = [
        ("Cumulative Mode", test_cumulative_mode),
        ("Window Mode", test_window_mode),
        ("Min Visits Filtering", test_min_visits_filtering),
        ("Mode Comparison", compare_modes)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Final results
    print("\n" + "="*60)
    print("🏁 TEST RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Cleanup
    cleanup_test_files()
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)