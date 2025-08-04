# Expanded Preflop Action Abstraction - Implementation Notes

## Overview
This implementation expands the preflop action abstraction in the CFR model to include multiple bet sizes as requested in the problem statement.

## Changes Made

### 1. Updated Action Set
**File**: `enhanced_cfr_preflop_generator_v2.py`
- **Before**: 6 actions (fold, call_small, call_large, raise_small, raise_large, all_in)
- **After**: 7 actions (fold, call_small, call_mid, call_high, raise_small, raise_mid, raise_high)

```python
# NEW ACTION SET
ACTIONS = {
    "fold": 0,
    "call_small": 1,    # Call ≤15% of stack  
    "call_mid": 2,      # Call 15-30% of stack
    "call_high": 3,     # Call >30% of stack
    "raise_small": 4,   # Raise 2-2.5x
    "raise_mid": 5,     # Raise 2.5-3x  
    "raise_high": 6     # Raise 3x+ or all-in
}
```

### 2. Enhanced Scenario Representation
**File**: `enhanced_cfr_preflop_generator_v2.py`
- Updated `get_available_actions()` function to properly categorize bet sizes
- Scenarios now encode bet sizes at each decision node through `bet_size_category` and `bet_to_call_bb`
- Stack depth categories ensure different action availability based on effective stack sizes

### 3. Updated CFR Logic
**File**: `enhanced_cfr_trainer_v2.py`
- Enhanced `estimate_counterfactual_payoff()` to handle all 7 actions with appropriate risk/reward scaling
- CFR regret minimization and strategy tracking account for expanded action set
- All strategy calculation functions handle variable action sets per scenario

### 4. Comprehensive CSV Export
**File**: `enhanced_cfr_trainer_v2.py` - `export_strategies_to_csv()` method
- Exports probabilities for each action: fold_prob, call_small_prob, call_mid_prob, call_high_prob, raise_small_prob, raise_mid_prob, raise_high_prob
- Includes scenario details: hand_category, example_hands, position, stack_depth, bet_size_category, tournament_stage
- Provides best action as determined by model with confidence level
- Includes training metadata: training_games, scenario_key

## Key Features

### Action Coverage
All 7 actions are available in appropriate scenarios:
- **FOLD**: Always available when facing a bet
- **CALL_SMALL**: Available when bet is ≤15% of stack
- **CALL_MID**: Available when bet is 15-30% of stack  
- **CALL_HIGH**: Available when bet is >30% of stack
- **RAISE_SMALL**: Always available when sufficient stack
- **RAISE_MID**: Always available when sufficient stack
- **RAISE_HIGH**: Always available when sufficient stack

### Scenario Encoding
Each scenario captures:
- Hand category (premium_pairs, medium_aces, etc.)
- Position (BTN, BB)
- Stack depth (ultra_short, short, medium, deep, very_deep)
- Bet size category (tiny, small, medium, large, huge, no_bet)
- Tournament stage (early, middle, late, bubble)

### CSV Output Format
```csv
scenario_key,hand_category,example_hands,position,stack_depth,bet_size_category,tournament_stage,training_games,best_action,confidence,fold_prob,call_small_prob,call_mid_prob,call_high_prob,raise_small_prob,raise_mid_prob,raise_high_prob
premium_pairs_BTN_deep_tiny_late,premium,AA KK QQ,BTN,deep,tiny,late,5,RAISE_HIGH,0.542,0.067,0.067,0.0,0.0,0.218,0.157,0.542
```

## Usage

### Basic Training
```python
from enhanced_cfr_trainer_v2 import EnhancedCFRTrainer
from enhanced_cfr_preflop_generator_v2 import generate_enhanced_scenarios

# Generate scenarios
scenarios = generate_enhanced_scenarios(100)

# Train CFR model
trainer = EnhancedCFRTrainer(scenarios=scenarios)
for iteration in range(1000):
    scenario = random.choice(scenarios)
    trainer.play_enhanced_scenario(scenario)
    trainer.scenario_counter[trainer.get_scenario_key(scenario)] += 1

# Export results
trainer.export_strategies_to_csv("results.csv")
```

### Demo Script
Run `python expanded_preflop_cfr_demo.py` for a complete demonstration of the expanded action abstraction.

## Verification

1. **Action Set**: Verified all 7 actions are properly defined and available in appropriate scenarios
2. **Scenario Coverage**: Confirmed bet sizes are encoded at each decision node
3. **CFR Logic**: Tested regret minimization handles all expanded actions
4. **CSV Export**: Validated output contains all requested information
5. **Isolation**: Changes are isolated to preflop trainer logic as requested

The implementation successfully expands the preflop action abstraction while maintaining the existing architecture and ensuring comprehensive scenario coverage.