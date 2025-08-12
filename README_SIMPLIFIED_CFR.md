# Simplified CFR Training System

A streamlined poker AI training system using direct hole card scenarios and preflop-only simulation for comprehensive strategy learning.

## 🎯 Overview

This simplified CFR system addresses the key limitations of the previous hand category-based approach by:

- **Direct hole card scenarios**: No hand category abstraction - uses all 1326 possible hole card combinations
- **Random scenario generation**: Monte Carlo selection of hole cards and stack sizes for each iteration
- **Preflop-only simulation**: Focus on preflop betting with immediate showdown after betting concludes
- **Full coverage tracking**: Ensures all hole card combinations are eventually visited
- **Heads-up match mode**: Optional continuous play until one player is busted

## 🚀 Key Improvements

### Scenario Space Coverage
- **Old system**: Limited to hand categories (only 'trash' hands were explored)
- **New system**: 40% coverage of all 1326 hole card combinations in just 300 iterations

### Simplified Architecture
- **Old system**: Complex scenario keys `hand_category|position|stack_category|blinds_level` 
- **New system**: Simple keys `[Card1♠] [Card2♥]|stack_category`

### Training Efficiency
- **Old system**: ~200/min training rate with limited exploration
- **New system**: ~2000/min training rate with comprehensive exploration

## 📁 Core Files

- **`simplified_scenario_generator.py`** - Direct hole card scenario generation
- **`simplified_cfr_trainer.py`** - Streamlined CFR trainer implementation  
- **`run_simplified_cfr_training.py`** - Command-line interface
- **`test_simplified_cfr.py`** - Comprehensive test suite

## 🎮 Quick Start

### Basic Training (1000 iterations)
```bash
python run_simplified_cfr_training.py --iterations 1000
```

### High Exploration Training  
```bash
python run_simplified_cfr_training.py --iterations 2000 --epsilon 0.3
```

### Heads-up Match Mode
```bash
python run_simplified_cfr_training.py --heads-up --starting-stack 50 --iterations 500
```

### Resume from Checkpoint
```bash
python run_simplified_cfr_training.py --resume simplified_cfr_final.pkl --iterations 1000
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 1000 | Number of training iterations |
| `--epsilon` | 0.1 | Exploration rate (0.0-1.0) |
| `--min-visits` | 5 | Minimum visits before exploitation |
| `--heads-up` | False | Enable heads-up match mode |
| `--starting-stack` | 100 | Starting stack size (bb) for heads-up |
| `--save-interval` | 1000 | Checkpoint save frequency |
| `--output-prefix` | simplified_cfr | Output file prefix |

## 📊 Example Output

```
🚀 Simplified CFR Training System
============================================================
⚙️  Training Configuration:
   🎲 Iterations: 2,000
   🔍 Epsilon exploration: 0.2
   🎮 Heads-up mode: Disabled

📈 Training Progress:
   Iteration | Scenarios | Coverage | Rate/min | Status
   --------------------------------------------------
   
Iteration 100: 199 scenarios, coverage=14.9%, rate=2058.3/min
Iteration 200: 390 scenarios, coverage=27.7%, rate=1973.2/min
...
Iteration 2000: 1850 scenarios, coverage=85.2%, rate=2010.5/min

🎉 Training Completed Successfully!
============================================================
⏱️  Total time: 1.0 minutes
🎲 Iterations completed: 2,000
🎯 Scenarios visited: 1,850
📈 Coverage: 85.2% (1130/1326 combinations)
```

## 🎯 Scenario Generation

### Random Scenario Selection
Each iteration generates a completely random scenario:
```python
scenario = {
    "hero_cards": [Card1, Card2],      # Random from 1326 combinations
    "villain_cards": [Card3, Card4],   # Random from remaining cards
    "hero_stack_bb": 45,               # Random stack size
    "villain_stack_bb": 45,            # Equal stacks
    "stack_category": "medium"         # Derived from stack size
}
```

### Stack Categories
- **Short**: 10-25bb (push/fold territory)
- **Medium**: 26-75bb (standard play)  
- **Deep**: 76-150bb (deep stack strategy)

### Scenario Keys
Simple format: `[A♠] [K♥]|medium`
- Direct hole card representation (no abstraction)
- Stack category for action mapping

## 🃏 Preflop-Only Simulation

### Betting Process
1. Players receive random hole cards
2. Preflop betting occurs (fold/call/raise actions)
3. After betting concludes, all 5 community cards are dealt immediately
4. Hands are evaluated and winner determined
5. No postflop actions - pure preflop strategy learning

### Action Mapping
Actions mapped by bet size relative to stack:
- **call_small**: ≤15% of stack
- **call_mid**: 15-30% of stack  
- **call_high**: >30% of stack
- **raise_small/mid/high**: Various raise sizes

## 🎮 Heads-up Match Mode

Continuous play mode where:
- Both players start with equal stacks
- Game continues hand-by-hand until one player is busted
- Realistic tournament-style progression
- Stack sizes evolve based on results

Example:
```bash
python run_simplified_cfr_training.py --heads-up --starting-stack 30

# Output:
🏁 Heads-up match completed after 22 hands
💰 Final stacks: Hero=50.5bb, Villain=-0.5bb  
🏆 Hero wins the heads-up match!
```

## 📈 Coverage Tracking

The system tracks exploration of all possible hole card combinations:

### Coverage Statistics
```json
{
  "total_possible_combinations": 1326,
  "unique_hole_cards_visited": 531, 
  "coverage_percent": 40.0,
  "total_scenarios_visited": 574
}
```

### Coverage Progression
- 100 iterations: ~15% coverage
- 300 iterations: ~40% coverage  
- 1000 iterations: ~70% coverage
- 5000 iterations: ~95% coverage

## 💾 Checkpointing & Resuming

### Automatic Checkpointing
```bash
# Saves checkpoint every 1000 iterations
python run_simplified_cfr_training.py --iterations 5000 --save-interval 1000
```

### Resume Training
```bash
# Resume from any checkpoint
python run_simplified_cfr_training.py --resume checkpoints/simplified_cfr_1000.pkl --iterations 2000
```

### Emergency Recovery
Interrupted training automatically saves emergency checkpoint for seamless resuming.

## 📊 Output Files

Training generates comprehensive output files:

### Scenarios
`simplified_cfr_scenarios_TIMESTAMP.csv` - All played scenarios with results
```csv
hero_cards_str,villain_cards_str,hero_action,villain_action,result,hero_stack_change
[A♠] [K♥],[Q♣] [J♦],raise_high,fold,hero_wins,1.5
```

### Strategies  
`simplified_cfr_hero_strategies_TIMESTAMP.csv` - Learned strategies
```csv
scenario,action,probability
[A♠] [K♥]|medium,raise_high,0.65
[A♠] [K♥]|medium,call_small,0.35
```

### Coverage Report
`simplified_cfr_coverage_TIMESTAMP.json` - Exploration statistics

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_simplified_cfr.py
```

Tests validate:
- ✅ Basic functionality
- ✅ Coverage progression  
- ✅ Preflop simulation
- ✅ Heads-up mode
- ✅ Checkpointing
- ✅ Coverage diversity
- ✅ Scenario uniqueness

## 🔄 Migration Guide

### From Old System
The simplified system completely replaces the hand category-based approach:

**Old approach**: 
- Hand categories: 11 predefined groups
- Limited exploration (only 'trash' hands)
- Complex scenario keys

**New approach**:
- Direct hole cards: All 1326 combinations
- Comprehensive exploration (40%+ coverage in 300 iterations)
- Simple scenario keys

### Backward Compatibility
The old system files remain intact. Use the simplified system for new training:
```bash
# Old system (limited exploration)
python run_natural_cfr_training.py --games 1000

# New system (comprehensive exploration)  
python run_simplified_cfr_training.py --iterations 1000
```

## 📋 Architecture Benefits

### Separation of Concerns
- **Scenario Generation**: `simplified_scenario_generator.py`
- **CFR Training**: `simplified_cfr_trainer.py`  
- **Command Interface**: `run_simplified_cfr_training.py`

### Modular Design
- Easy to extend to postflop scenarios
- Pluggable evaluation functions
- Configurable exploration strategies

### Performance Optimized
- 10x faster iteration rate
- Memory efficient scenario storage
- Scalable to millions of iterations

## 🎯 Use Cases

### Research
- Study preflop strategy convergence
- Analyze coverage requirements for CFR
- Compare scenario abstraction approaches

### Training
- Develop robust preflop strategies
- Generate training data for neural networks
- Baseline for more complex poker AI systems

### Analysis
- Understand optimal preflop play across stack sizes
- Generate strategy charts for different scenarios
- Validate theoretical poker concepts

This simplified CFR system represents a significant improvement in poker AI training efficiency and coverage, providing a solid foundation for developing strong preflop strategies.