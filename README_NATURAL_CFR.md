# Poker CFR AI Training System

A comprehensive poker AI training system with both traditional and simplified CFR approaches for developing optimal preflop strategies.

## 🚀 **NEW: Simplified CFR System (Recommended)**

**Major breakthrough in scenario coverage and training efficiency!**

The new simplified CFR system addresses critical limitations of hand category abstraction:

### 🎯 Key Improvements
- **Coverage**: 28.3% of all hole card combinations in 200 iterations (vs 0% non-trash coverage in old system)  
- **Speed**: ~2000 iterations/min (vs ~200/min in old system)
- **Scenarios**: Direct hole card combinations (vs limited hand categories)
- **Architecture**: Clean separation of scenario generation and CFR logic

### Quick Start
```bash
# Comprehensive training with full hole card exploration
python run_simplified_cfr_training.py --iterations 1000 --epsilon 0.2

# Heads-up match mode
python run_simplified_cfr_training.py --heads-up --starting-stack 50 --iterations 500

# High exploration for maximum coverage
python run_simplified_cfr_training.py --iterations 2000 --epsilon 0.4
```

**📖 [Read the Simplified CFR Guide](README_SIMPLIFIED_CFR.md)** for complete documentation.

## 📊 System Comparison

| Feature | Simplified CFR | Traditional CFR |
|---------|---------------|-----------------|
| **Scenario Space** | All 1326 hole card combinations | 11 hand categories only |
| **Coverage (200 iter)** | 28.3% of all combinations | Only 'trash' hands (0% premium) |
| **Training Speed** | ~2000 iterations/min | ~200 iterations/min |
| **Exploration** | Random Monte Carlo scenarios | Limited predefined scenarios |
| **Architecture** | Clean, modular design | Complex hand category abstraction |

## 🎯 Recommended Usage

- **New projects**: Use the Simplified CFR System (`run_simplified_cfr_training.py`)
- **Research**: Compare both approaches for academic studies  
- **Legacy**: Traditional system remains available for backward compatibility

---

## 🎮 Traditional Natural Game CFR Training System

## 🎯 Overview

Instead of training on pre-defined scenarios, this system generates natural poker games where both hero and villain evolve their strategies through self-play. This leads to more realistic and robust poker AI.

## 🚀 Key Features

### Natural Monte Carlo Simulation
- Deals random cards to both players
- Randomizes position, stack sizes, blinds, and game conditions
- Both players act using their learned strategies (no hardcoded behavior)
- Multi-step betting sequences with proper game tree handling

### Co-evolving Strategies
- Hero and villain maintain separate strategy databases
- Both players learn from each other's actions through CFR updates
- Realistic opponent modeling through mutual adaptation
- Counter-strategies develop naturally through gameplay

### Epsilon-greedy Exploration
- Tracks visit count per state-action pair
- Forced exploration ensures rare/important scenarios get visited
- Configurable exploration rate and minimum visit thresholds

### Natural Scenario Recording
- Records scenarios that emerge naturally from gameplay:
  - Hand category, position, stack depth, blinds level
  - Villain stack category, opponent actions, 3-bet indicators
  - Full action history and payoffs
- No forced scenario lists - everything emerges organically

## 📁 Files

- **`natural_game_cfr_trainer.py`** - Main trainer implementing natural CFR
- **`run_natural_cfr_training.py`** - Command-line runner script
- **`test_natural_cfr.py`** - Comprehensive test suite

## 🎮 Quick Start

### Basic Demo (1000 games)
```bash
python run_natural_cfr_training.py --mode demo --games 1000
```

### Full Training (10,000 games)
```bash
python run_natural_cfr_training.py --games 10000 --epsilon 0.1
```

### Resume Training
```bash
python run_natural_cfr_training.py --resume checkpoint.pkl --games 5000
```

### Analyze Results
```bash
python run_natural_cfr_training.py --mode analysis --resume final.pkl
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--games` | 10000 | Number of games to simulate |
| `--epsilon` | 0.1 | Exploration rate (0.0-1.0) |
| `--min-visits` | 5 | Minimum visits before considering scenario trained |
| `--tournament-penalty` | 0.6 | Tournament survival penalty factor |
| `--save-interval` | 1000 | Save progress every N games |
| `--log-interval` | 100 | Log progress every N games |

## 📊 Example Output

```
🚀 Natural Game CFR Training System
🎯 Games to simulate: 1,000
🔍 Epsilon exploration: 0.1

Game   100:  25 scenarios, hero_wr=0.48, rate=224.5/min
Game   200:  27 scenarios, hero_wr=0.49, rate=228.6/min
...
Game 1,000:  42 scenarios, hero_wr=0.51, rate=235.2/min

🎉 Training completed!
📊 Unique scenarios discovered: 42
🎯 Hero win rate: 51.0%
🎲 Showdown rate: 58.6%
🔥 3-bet rate: 30.8%
```

## 📈 Strategy Evolution

The system tracks strategy development for both players:

**Hero Strategies (sample):**
```
trash|BTN|deep|low:
  fold: 70.4%
  call_small: 21.3% 
  raise_small: 8.3%

premium_pairs|BB|short|high:
  call_small: 15.2%
  raise_mid: 42.1%
  raise_high: 42.7%
```

**Villain Strategies (sample):**
```
medium_aces|BTN|medium|medium:
  fold: 25.1%
  call_small: 38.4%
  raise_small: 36.5%
```

## 🎯 Advantages Over Traditional CFR

1. **Natural Emergence:** Scenarios emerge from actual gameplay vs forced combinations
2. **Co-evolution:** Both players adapt to each other vs one-sided learning
3. **Realistic Training:** Based on natural game flow vs artificial scenario generation
4. **Exploration Balance:** Epsilon-greedy ensures comprehensive scenario coverage
5. **Extensible Design:** Multi-step betting easily extends to postflop play

## 🧪 Testing

Run the test suite to validate functionality:

```bash
python test_natural_cfr.py
```

Tests include:
- Basic functionality (game state generation, action availability)
- Game simulation (complete hand playouts)
- Strategy evolution (learning over time)
- Save/load functionality
- Hand category discovery

## 📋 Output Files

The system generates several output files:

- **`natural_scenarios_TIMESTAMP.csv`** - All natural scenarios recorded
- **`hero_natural_strategies_TIMESTAMP.csv`** - Hero's learned strategies
- **`villain_natural_strategies_TIMESTAMP.csv`** - Villain's learned strategies  
- **`natural_cfr_final_TIMESTAMP.pkl`** - Complete training state for resuming

## 🔬 Technical Details

### Game Flow
1. Deal random hole cards to both players
2. Assign random positions, stack sizes, blinds level
3. Players act using their current learned strategies
4. Record the natural scenario that emerges
5. Calculate payoffs and update both players' strategies via CFR
6. Repeat for thousands of iterations

### Strategy Storage
- Scenarios keyed by: `hand_category|position|stack_category|blinds_level`
- Separate regret and strategy tracking for hero and villain
- Action probabilities normalized from accumulated strategy sums

### Exploration Mechanism
- Epsilon probability of random action selection
- Visit count tracking ensures rare scenarios get attention
- Minimum visit thresholds before exploitation

## 📚 Integration

This system integrates seamlessly with the existing poker-cfr-ai repository:

- Inherits from `EnhancedCFRTrainer` for full CFR functionality
- Uses `enhanced_cfr_preflop_generator_v2.py` for hand classification
- Compatible with existing action sets and evaluation functions
- Leverages treys library for poker hand evaluation

## 🎭 Example Use Cases

1. **Research:** Study strategy evolution and convergence in poker AI
2. **Training:** Develop robust poker bots through natural self-play
3. **Analysis:** Understand optimal play in different stack/position scenarios
4. **Benchmarking:** Compare natural vs traditional CFR training approaches

This natural game CFR system represents a significant advancement in poker AI training, moving from artificial scenario-based learning to organic strategy development through realistic gameplay simulation.