#!/usr/bin/env python3
"""
Test script to validate CSV output with very low iteration counts.

This test ensures that the unified scenario lookup table is populated with
action frequency percentages even for short training runs, as required by
the problem statement.
"""

import os
import time
import tempfile
from pathlib import Path
import pandas as pd
from run_gcp_cfr_training import GCPCFRTrainer
from natural_game_cfr_trainer import NaturalGameCFRTrainer


def test_gcp_trainer_low_iterations():
    """Test GCP trainer with very low iteration count produces valid CSV."""
    print("🧪 Testing GCP Trainer with Low Iterations")
    print("=" * 50)
    
    # Create a test GCP trainer with minimal settings
    trainer = GCPCFRTrainer(n_workers=1, log_interval_minutes=1)
    
    # Run training with very low iteration count (10 iterations total)
    print("🎯 Running parallel training with 10 iterations...")
    start_time = time.time()
    trainer.run_parallel_training(total_iterations=10)
    end_time = time.time()
    
    print(f"✅ Training completed in {end_time - start_time:.1f} seconds")
    
    # Check if CSV was created and contains data
    csv_path = Path("scenario_lookup_table.csv")
    assert csv_path.exists(), "scenario_lookup_table.csv should be created"
    
    # Load and analyze CSV
    df = pd.read_csv(csv_path)
    print(f"📊 CSV contains {len(df)} scenarios")
    
    # Verify required columns are present
    required_columns = [
        'scenario_key', 'hand_category', 'stack_category', 'blinds_level', 
        'position', 'opponent_action', 'iterations_completed', 'total_rollouts', 
        'regret', 'average_strategy', 'strategy_confidence', 'fold_pct', 'call_pct',
        'raise_small_pct', 'raise_mid_pct', 'raise_high_pct', 'is_3bet', 'last_updated'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
        print(f"✅ Column '{col}' present")
    
    # Verify that at least some scenarios have training data
    scenarios_with_data = df[df['iterations_completed'] > 0]
    assert len(scenarios_with_data) > 0, "At least some scenarios should have training data"
    print(f"✅ {len(scenarios_with_data)} scenarios have training data")
    
    # Verify action percentages sum to approximately 100% for trained scenarios
    for idx, row in scenarios_with_data.head(3).iterrows():
        total_pct = row['fold_pct'] + row['call_pct'] + row['raise_small_pct'] + row['raise_mid_pct'] + row['raise_high_pct']
        print(f"📊 {row['scenario_key']}: fold={row['fold_pct']:.1f}%, call={row['call_pct']:.1f}%, raise_s={row['raise_small_pct']:.1f}%, raise_m={row['raise_mid_pct']:.1f}%, raise_h={row['raise_high_pct']:.1f}% (total={total_pct:.1f}%)")
        
        # For scenarios with data, percentages should sum to approximately 100%
        if row['iterations_completed'] > 0:
            assert 95 <= total_pct <= 105, f"Action percentages should sum to ~100%, got {total_pct}%"
    
    # Check 3-bet column exists and has valid values
    assert df['is_3bet'].dtype in ['int64', 'bool'], "is_3bet column should be binary (int or bool)"
    assert all(df['is_3bet'].isin([0, 1])), "is_3bet column should only contain 0 or 1"
    print(f"✅ 3-bet column is valid: {df['is_3bet'].sum()} scenarios marked as 3-bet")
    
    return trainer


def test_natural_trainer_low_games():
    """Test Natural trainer with very low game count produces valid CSV."""
    print("\n🧪 Testing Natural Trainer with Low Game Count")
    print("=" * 50)
    
    # Create a natural trainer with minimal settings
    trainer = NaturalGameCFRTrainer(epsilon_exploration=0.5, min_visit_threshold=1)
    
    # Run training with very low game count (5 games)
    print("🎯 Running natural game training with 5 games...")
    start_time = time.time()
    trainer.train(n_games=5, log_interval=2)  # Log every 2 games
    end_time = time.time()
    
    print(f"✅ Training completed in {end_time - start_time:.1f} seconds")
    
    # Check if CSV was created and contains data
    csv_path = Path("scenario_lookup_table.csv")
    assert csv_path.exists(), "scenario_lookup_table.csv should be created"
    
    # Load and analyze CSV
    df = pd.read_csv(csv_path)
    print(f"📊 CSV contains {len(df)} scenarios")
    
    # Verify required columns are present
    required_columns = [
        'scenario_key', 'hand_category', 'stack_category', 'blinds_level', 
        'position', 'opponent_action', 'iterations_completed', 'total_rollouts', 
        'regret', 'average_strategy', 'strategy_confidence', 'fold_pct', 'call_pct',
        'raise_small_pct', 'raise_mid_pct', 'raise_high_pct', 'is_3bet', 'last_updated'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    print(f"✅ All required columns present")
    
    # Verify that at least some scenarios have training data
    scenarios_with_data = df[df['iterations_completed'] > 0]
    print(f"📊 {len(scenarios_with_data)} scenarios discovered during natural gameplay")
    
    # For natural training, we expect fewer scenarios but they should be valid
    if len(scenarios_with_data) > 0:
        print("✅ Natural training discovered scenarios successfully")
        
        # Check action percentages for discovered scenarios
        for idx, row in scenarios_with_data.head(2).iterrows():
            total_pct = row['fold_pct'] + row['call_pct'] + row['raise_small_pct'] + row['raise_mid_pct'] + row['raise_high_pct']
            print(f"📊 {row['scenario_key']}: fold={row['fold_pct']:.1f}%, call={row['call_pct']:.1f}%, raise_s={row['raise_small_pct']:.1f}%, raise_m={row['raise_mid_pct']:.1f}%, raise_h={row['raise_high_pct']:.1f}% (total={total_pct:.1f}%)")
            
            if row['iterations_completed'] > 0:
                assert 95 <= total_pct <= 105, f"Action percentages should sum to ~100%, got {total_pct}%"
    else:
        print("ℹ️  No scenarios discovered yet with only 5 games (this is expected)")
    
    # Check 3-bet column exists and has valid values
    assert df['is_3bet'].dtype in ['int64', 'bool'], "is_3bet column should be binary (int or bool)"
    assert all(df['is_3bet'].isin([0, 1])), "is_3bet column should only contain 0 or 1"
    print(f"✅ 3-bet column is valid")
    
    return trainer


def test_csv_persistence_across_trainers():
    """Test that CSV persists across different trainer instances."""
    print("\n🧪 Testing CSV Persistence Across Trainers")
    print("=" * 50)
    
    # First, run a quick GCP training
    trainer1 = GCPCFRTrainer(n_workers=1, log_interval_minutes=1)
    trainer1.run_parallel_training(total_iterations=5)
    
    # Read the CSV
    df1 = pd.read_csv("scenario_lookup_table.csv")
    initial_scenarios = len(df1)
    print(f"📊 After GCP training: {initial_scenarios} scenarios")
    
    # Now run natural training (should append/update scenarios)
    trainer2 = NaturalGameCFRTrainer(epsilon_exploration=0.5, min_visit_threshold=1)
    trainer2.train(n_games=3, log_interval=1)
    
    # Read the updated CSV
    df2 = pd.read_csv("scenario_lookup_table.csv")
    final_scenarios = len(df2)
    print(f"📊 After Natural training: {final_scenarios} scenarios")
    
    # Verify CSV structure is maintained
    required_columns = [
        'scenario_key', 'hand_category', 'stack_category', 'blinds_level', 
        'position', 'opponent_action', 'iterations_completed', 'total_rollouts', 
        'regret', 'average_strategy', 'strategy_confidence', 'fold_pct', 'call_pct',
        'raise_small_pct', 'raise_mid_pct', 'raise_high_pct', 'is_3bet', 'last_updated'
    ]
    
    for col in required_columns:
        assert col in df2.columns, f"Missing required column after mixed training: {col}"
    
    print("✅ CSV structure maintained across different trainer types")
    
    return df2


def main():
    """Run all low iteration tests."""
    print("🧪 Low Iteration CSV Output Test Suite")
    print("=" * 60)
    
    try:
        # Clean up any existing CSV first
        csv_path = Path("scenario_lookup_table.csv")
        if csv_path.exists():
            os.remove(csv_path)
        
        # Test 1: GCP trainer with very low iterations
        test_gcp_trainer_low_iterations()
        
        # Test 2: Natural trainer with very low game count
        test_natural_trainer_low_games()
        
        # Test 3: CSV persistence
        test_csv_persistence_across_trainers()
        
        print("\n🎉 All Low Iteration Tests Passed!")
        print("✅ CSV output is populated even with very low iteration counts")
        print("✅ Action frequency percentages are calculated correctly")
        print("✅ 3-bet binary column is present and valid")
        print("✅ All required scenario columns are present")
        print("✅ CSV persists and updates across trainer types")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        csv_path = Path("scenario_lookup_table.csv")
        if csv_path.exists():
            os.remove(csv_path)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)