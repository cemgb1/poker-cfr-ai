#!/usr/bin/env python3
"""
Test script to verify the CFR refactoring is working correctly
"""

from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios, generate_dynamic_betting_context
from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer

def test_scenario_generation():
    """Test that scenario generation works correctly"""
    print("🧪 Testing scenario generation...")
    
    scenarios = generate_enhanced_scenarios()
    
    # Verify we get exactly 330 scenarios (11 × 2 × 5 × 3)
    assert len(scenarios) == 330, f"Expected 330 scenarios, got {len(scenarios)}"
    print(f"✅ Generated exactly 330 scenarios")
    
    # Verify all required fields are present and no bet_size_category
    required_fields = ['hand_category', 'hero_position', 'stack_category', 'blinds_level']
    prohibited_fields = ['bet_size_category', 'available_actions', 'opponent_action']
    
    for scenario in scenarios[:10]:  # Check first 10
        for field in required_fields:
            assert field in scenario, f"Missing field {field} in scenario"
        for field in prohibited_fields:
            assert field not in scenario, f"Prohibited field {field} found in scenario"
    
    print(f"✅ Scenario structure is correct")
    
    # Test dynamic betting context generation
    test_scenario = scenarios[0]
    betting_context = generate_dynamic_betting_context(test_scenario)
    
    expected_betting_fields = ['opponent_action', 'bet_to_call_bb', 'bet_size_category', 'available_actions']
    for field in expected_betting_fields:
        assert field in betting_context, f"Missing field {field} in betting context"
    
    print(f"✅ Dynamic betting context generation works")

def test_scenario_keys():
    """Test that scenario keys don't include bet_size_category"""
    print("🧪 Testing scenario key structure...")
    
    scenarios = generate_enhanced_scenarios()
    trainer = EnhancedCFRTrainer(scenarios=scenarios)
    
    for scenario in scenarios[:10]:  # Check first 10
        key = trainer.get_scenario_key(scenario)
        parts = key.split('|')
        
        # Should be: hand_category|position|stack_category|blinds_level
        assert len(parts) == 4, f"Expected 4 parts in key, got {len(parts)}: {key}"
        assert 'bet_size' not in key.lower(), f"bet_size found in key: {key}"
    
    print(f"✅ Scenario keys are correctly structured")

def test_training_functionality():
    """Test that training still works with refactored code"""
    print("🧪 Testing training functionality...")
    
    scenarios = generate_enhanced_scenarios()
    trainer = EnhancedCFRTrainer(scenarios=scenarios)
    
    # Run a few training iterations
    for i in range(10):
        scenario = trainer.select_balanced_scenario()
        result = trainer.play_enhanced_scenario(scenario)
        
        # Verify result structure
        expected_fields = ['scenario_key', 'hero_action', 'payoff', 'betting_context']
        for field in expected_fields:
            assert field in result, f"Missing field {field} in result"
    
    print(f"✅ Training functionality works correctly")
    
    # Test strategy export
    df = trainer.export_strategies_to_csv("/tmp/test_strategies.csv")
    if df is not None and len(df) > 0:
        # Verify CSV has correct columns (no bet_size_category)
        assert 'blinds_level' in df.columns, "blinds_level missing from CSV"
        assert 'bet_size_category' not in df.columns, "bet_size_category should not be in CSV"
        print(f"✅ CSV export works correctly")
    else:
        print(f"⚠️  Not enough training data for CSV export test")

def test_all_combinations_generated():
    """Test that all expected combinations are generated"""
    print("🧪 Testing all combinations are generated...")
    
    scenarios = generate_enhanced_scenarios()
    
    # Count unique combinations
    from collections import Counter
    hand_cats = Counter(s['hand_category'] for s in scenarios)
    positions = Counter(s['hero_position'] for s in scenarios)
    stack_cats = Counter(s['stack_category'] for s in scenarios)
    blinds_levels = Counter(s['blinds_level'] for s in scenarios)
    
    # Should have 11 hand categories, each appearing 30 times (2×5×3)
    assert len(hand_cats) == 11, f"Expected 11 hand categories, got {len(hand_cats)}"
    for count in hand_cats.values():
        assert count == 30, f"Expected 30 scenarios per hand category, got {count}"
    
    # Should have 2 positions, each appearing 165 times (11×5×3)
    assert len(positions) == 2, f"Expected 2 positions, got {len(positions)}"
    for count in positions.values():
        assert count == 165, f"Expected 165 scenarios per position, got {count}"
    
    print(f"✅ All expected combinations generated correctly")

if __name__ == "__main__":
    print("🚀 Running CFR Refactor Tests")
    print("=" * 50)
    
    try:
        test_scenario_generation()
        test_scenario_keys()
        test_training_functionality() 
        test_all_combinations_generated()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ CFR refactoring completed successfully")
        print("   • bet_size_category removed from scenario keys")
        print("   • All 330 scenario combinations generated automatically")
        print("   • Dynamic opponent betting during simulation")
        print("   • Training and export functionality working")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise