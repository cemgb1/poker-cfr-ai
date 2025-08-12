#!/usr/bin/env python3
"""
Test script to validate the unified scenario lookup table implementation.

This script tests:
1. Default tournament penalty is 0.2
2. Unified scenario lookup CSV export functionality
3. CSV structure and required columns
4. Real-time updates during training intervals
"""

import os
import time
import tempfile
from pathlib import Path
import pandas as pd
from run_gcp_cfr_training import GCPCFRTrainer
from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer

def test_tournament_penalty_default():
    """Test that tournament penalty defaults to 0.2."""
    print("🧪 Testing Tournament Penalty Default")
    print("=" * 50)
    
    # Test EnhancedCFRTrainer default
    trainer = EnhancedCFRTrainer()
    assert trainer.tournament_survival_penalty == 0.2, f"Expected 0.2, got {trainer.tournament_survival_penalty}"
    print(f"✅ EnhancedCFRTrainer default: {trainer.tournament_survival_penalty}")
    
    return trainer

def test_unified_scenario_lookup_csv_structure():
    """Test the unified scenario lookup CSV structure and columns."""
    print("\n📊 Testing Unified Scenario Lookup CSV Structure")
    print("=" * 50)
    
    # Create a test GCP trainer
    trainer = GCPCFRTrainer(n_workers=1, log_interval_minutes=1)  # 1 minute for testing
    
    # Test CSV export with no data (should create empty CSV with headers)
    csv_filename = "test_scenario_lookup_table.csv"
    df = trainer.export_unified_scenario_lookup_csv(csv_filename)
    
    # Check required columns
    required_columns = [
        'scenario_key', 'hand_category', 'stack_category', 'blinds_level', 
        'position', 'opponent_action', 'iterations_completed', 'total_rollouts', 
        'regret', 'average_strategy', 'strategy_confidence', 'last_updated'
    ]
    
    print(f"📋 CSV columns: {list(df.columns)}")
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
        print(f"✅ Column '{col}' present")
    
    # Check CSV file exists
    assert Path(csv_filename).exists(), f"CSV file {csv_filename} not created"
    print(f"✅ CSV file created: {csv_filename}")
    
    # Clean up
    os.remove(csv_filename)
    
    return trainer

def test_unified_scenario_lookup_with_data():
    """Test the unified scenario lookup CSV with some training data."""
    print("\n🎯 Testing Unified Scenario Lookup with Training Data")
    print("=" * 50)
    
    # Create a test trainer with some mock data
    trainer = GCPCFRTrainer(n_workers=1, log_interval_minutes=1)
    
    # Mock some combined data to simulate training progress
    trainer.combined_scenario_counter = {
        'premium_pairs|BTN|medium|low': 150,
        'medium_aces|BB|short|medium': 75,
        'suited_broadway|BTN|deep|high': 200
    }
    
    trainer.combined_strategy_sum = {
        'premium_pairs|BTN|medium|low': {
            'raise_small': 50, 'raise_mid': 80, 'raise_high': 20, 'call_small': 0, 'fold': 0
        },
        'medium_aces|BB|short|medium': {
            'call_small': 40, 'call_mid': 30, 'fold': 5, 'raise_small': 0
        },
        'suited_broadway|BTN|deep|high': {
            'raise_small': 100, 'raise_mid': 60, 'raise_high': 40, 'call_small': 0, 'fold': 0
        }
    }
    
    trainer.combined_regret_sum = {
        'premium_pairs|BTN|medium|low': {'raise_small': -0.1, 'raise_mid': 0.2, 'raise_high': -0.05},
        'medium_aces|BB|short|medium': {'call_small': 0.1, 'call_mid': -0.1, 'fold': -0.2},
        'suited_broadway|BTN|deep|high': {'raise_small': 0.05, 'raise_mid': 0.15, 'raise_high': 0.1}
    }
    
    # Export CSV
    csv_filename = "test_scenario_lookup_with_data.csv"
    df = trainer.export_unified_scenario_lookup_csv(csv_filename)
    
    print(f"📊 Exported {len(df)} scenarios")
    print(f"📋 Sample data:")
    if len(df) > 0:
        for idx, row in df.head(3).iterrows():
            print(f"   🎯 {row['scenario_key']}: {row['iterations_completed']} iterations, {row['average_strategy']} strategy ({row['strategy_confidence']:.1f}%)")
    
    # Validate data
    assert len(df) == 3, f"Expected 3 scenarios, got {len(df)}"
    
    # Check specific scenarios
    premium_row = df[df['scenario_key'] == 'premium_pairs|BTN|medium|low'].iloc[0]
    assert premium_row['iterations_completed'] == 150
    assert premium_row['hand_category'] == 'premium_pairs'
    assert premium_row['position'] == 'BTN'
    assert premium_row['average_strategy'] == 'RAISE'  # Should be RAISE since raise_total > others
    print(f"✅ Premium pairs scenario validated: {premium_row['average_strategy']} strategy")
    
    medium_aces_row = df[df['scenario_key'] == 'medium_aces|BB|short|medium'].iloc[0]
    assert medium_aces_row['iterations_completed'] == 75
    assert medium_aces_row['average_strategy'] == 'CALL'  # Should be CALL since call_total > others
    print(f"✅ Medium aces scenario validated: {medium_aces_row['average_strategy']} strategy")
    
    # Check CSV file contains data
    saved_df = pd.read_csv(csv_filename)
    assert len(saved_df) == 3, "CSV file should contain 3 rows"
    print(f"✅ CSV file saved with {len(saved_df)} scenarios")
    
    # Clean up
    os.remove(csv_filename)
    
    return trainer

def test_logging_interval_integration():
    """Test that the unified CSV export is called during logging intervals."""
    print("\n⏰ Testing Logging Interval Integration")
    print("=" * 50)
    
    # This test would require running actual training, which is complex for a test
    # Instead, we'll test the method call directly
    trainer = GCPCFRTrainer(n_workers=1, log_interval_minutes=1)
    
    # Mock some data
    trainer.combined_scenario_counter = {'test_scenario|BTN|medium|low': 10}
    trainer.combined_strategy_sum = {'test_scenario|BTN|medium|low': {'fold': 10}}
    
    # Test that log_training_progress calls the export method
    current_time = time.time()
    
    # This should create the CSV file via the logging method
    trainer.log_training_progress(current_time)
    
    # Check if the CSV was created
    csv_path = Path("scenario_lookup_table.csv")
    assert csv_path.exists(), "scenario_lookup_table.csv should be created during logging"
    
    # Check content
    df = pd.read_csv(csv_path)
    assert len(df) >= 1, "CSV should contain at least one scenario"
    print(f"✅ Logging interval created CSV with {len(df)} scenarios")
    
    # Clean up
    os.remove(csv_path)
    
    return trainer

def main():
    """Run all tests for the unified scenario lookup table implementation."""
    print("🧪 Unified Scenario Lookup Table Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Tournament penalty default
        test_tournament_penalty_default()
        
        # Test 2: CSV structure
        test_unified_scenario_lookup_csv_structure()
        
        # Test 3: CSV with data
        test_unified_scenario_lookup_with_data()
        
        # Test 4: Logging integration
        test_logging_interval_integration()
        
        print("\n🎉 All Tests Passed!")
        print("✅ Tournament penalty defaults to 0.2")
        print("✅ Unified scenario lookup CSV structure is correct")
        print("✅ CSV export works with training data")
        print("✅ CSV is exported during logging intervals")
        print("✅ Required columns are present and correctly populated")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)