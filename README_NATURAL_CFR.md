# Natural Game CFR Training System

A revolutionary poker AI training system based on natural Monte Carlo game simulation and co-evolving strategies.

## 🎯 Overview

Instead of training on pre-defined scenarios, this system generates natural poker games where both hero and villain evolve their strategies through self-play. This leads to more realistic and robust poker AI.

## 🚀 Key Features

## 🚀 Key Features

### Natural Monte Carlo Simulation
- Deals random cards to both players
- Randomizes specific stack sizes (500, 1000, 5000 BBs) - both players start with equal stacks
- Randomizes specific blind levels (2, 5, 10, 25, 50 BB) to simulate different tournament stages
- Both players act using their learned strategies (no hardcoded behavior)
- Multi-step betting sequences with proper game tree handling
- Preflop-only play (extendable to postflop)

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
  - Hand category (grouped: 'trash', 'premium_pairs', etc.)
  - Specific stack size (500, 1000, 5000 BBs) - both players equal
  - Specific blind size (2, 5, 10, 25, 50 BB)
  - Position, opponent actions, 3-bet indicators
  - Full action history and payoffs
- No forced scenario lists - everything emerges organically from natural gameplay

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
trash|BTN|1000|10:
  fold: 70.4%
  call_small: 21.3% 
  raise_small: 8.3%

premium_pairs|BB|500|50:
  call_small: 15.2%
  raise_mid: 42.1%
  raise_high: 42.7%
```

**Villain Strategies (sample):**
```
medium_aces|BTN|5000|25:
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

The system generates several output files with the new specific value format:

- **`natural_scenarios_TIMESTAMP.csv`** - All natural scenarios recorded with specific stack/blind values
- **`hero_natural_strategies_TIMESTAMP.csv`** - Hero's learned strategies  
- **`villain_natural_strategies_TIMESTAMP.csv`** - Villain's learned strategies
- **`final_lookup_table_TIMESTAMP.csv`** - **Single lookup table with all scenario strategies** (primary output)
- **`natural_cfr_final_TIMESTAMP.pkl`** - Complete training state for resuming

### New Output Format

The lookup table now uses **specific values** instead of categories:

**Scenario Key Format:** `hand_category|position|stack_size|blind_size`
- Example: `trash|BTN|1000|10` instead of `trash|BTN|very_deep|medium`

**CSV Columns Include:**
- `scenario_key`: Unique scenario identifier with specific values
- `hand_category`: Grouped hand types (trash, premium_pairs, etc.)
- `position`: BTN or BB
- `stack_size`: Specific stack size (500, 1000, 5000) 
- `blind_size`: Specific blind size (2, 5, 10, 25, 50)
- `stack_depth`/`blinds_level`: Legacy categories for backward compatibility
- `best_action`: Learned optimal action
- `confidence`: Strategy confidence level
- `estimated_ev`: Expected value from scenarios
- `player`: HERO or VILLAIN
- Action probabilities: `fold_pct`, `raise_small_pct`, etc.

## 🔬 Technical Details

### Game Flow
1. Deal random hole cards to both players
2. Assign random positions (BTN/BB)
3. Set equal stack sizes for both players from [500, 1000, 5000] BBs
4. Set random blind level from [2, 5, 10, 25, 50] BB
5. Players act using their current learned strategies (preflop only)
6. Reveal community cards and determine winner after betting complete
7. Record the natural scenario that emerges with specific values
8. Calculate payoffs and update both players' strategies via CFR
9. Repeat for thousands of iterations

### Strategy Storage
- Scenarios keyed by: `hand_category|position|stack_size|blind_size`
- Examples: `premium_pairs|BTN|1000|10`, `trash|BB|500|50`
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