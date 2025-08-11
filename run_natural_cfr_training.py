#!/usr/bin/env python3
"""
Natural CFR Training Runner Script

This script provides an easy way to run the new Natural Game CFR training system.
It supports different training modes and configurations for experimentation.

Features:
- Natural Monte Carlo game simulation
- Co-evolving hero and villain strategies  
- Epsilon-greedy exploration
- Multi-step betting sequences
- Natural scenario recording
- Automatic saving and loading
- Progress tracking and visualization

Usage:
    python run_natural_cfr_training.py --games 10000 --epsilon 0.1
    python run_natural_cfr_training.py --mode demo --games 1000
    python run_natural_cfr_training.py --resume checkpoint.pkl --games 5000
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

from natural_game_cfr_trainer import NaturalGameCFRTrainer


def run_natural_cfr_training(args):
    """
    Run natural CFR training with specified parameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        NaturalGameCFRTrainer: Trained model
    """
    print("🚀 Natural Game CFR Training System")
    print("=" * 60)
    print(f"🎲 Training mode: {args.mode}")
    print(f"🎯 Games to simulate: {args.games:,}")
    print(f"🔍 Epsilon exploration: {args.epsilon}")
    print(f"📊 Min visit threshold: {args.min_visits}")
    print(f"🏆 Tournament penalty: {args.tournament_penalty}")
    print(f"💾 Save interval: every {args.save_interval} games")
    print(f"📝 Log interval: every {args.log_interval} games")
    
    # Initialize trainer
    trainer = NaturalGameCFRTrainer(
        enable_pruning=args.enable_pruning,
        regret_pruning_threshold=args.regret_threshold,
        strategy_pruning_threshold=args.strategy_threshold,
        tournament_survival_penalty=args.tournament_penalty,
        epsilon_exploration=args.epsilon,
        min_visit_threshold=args.min_visits
    )
    
    # Load checkpoint if specified
    if args.resume:
        if Path(args.resume).exists():
            if trainer.load_training_state(args.resume):
                print(f"✅ Resumed training from {args.resume}")
            else:
                print(f"❌ Failed to load {args.resume}, starting fresh")
        else:
            print(f"❌ Checkpoint file {args.resume} not found, starting fresh")
    
    # Run training
    print(f"\n🎯 Starting natural game simulation...")
    training_start = time.time()
    
    try:
        results = trainer.train(
            n_games=args.games,
            save_interval=args.save_interval,
            log_interval=args.log_interval
        )
        
        training_end = time.time()
        training_duration = training_end - training_start
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Training time: {training_duration/60:.1f} minutes")
        print(f"🎲 Games per minute: {results['games_played'] / (training_duration/60):.1f}")
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n📊 Exporting results...")
        trainer.export_natural_scenarios_csv(f"natural_scenarios_{timestamp}.csv")
        trainer.export_strategies_csv(f"natural_strategies_{timestamp}.csv")
        
        # Save final checkpoint
        final_checkpoint = f"natural_cfr_final_{timestamp}.pkl"
        trainer.save_training_state(final_checkpoint)
        
        print(f"\n📁 Results saved:")
        print(f"   📊 Scenarios: natural_scenarios_{timestamp}.csv")
        print(f"   🎯 Hero strategies: hero_natural_strategies_{timestamp}.csv")
        print(f"   🎯 Villain strategies: villain_natural_strategies_{timestamp}.csv")
        print(f"   💾 Final checkpoint: {final_checkpoint}")
        
        # Show training summary
        print(f"\n📈 TRAINING SUMMARY:")
        print(f"   🎲 Total games played: {results['games_played']:,}")
        print(f"   📊 Unique scenarios discovered: {results['unique_scenarios']}")
        print(f"   🎯 Hero strategy scenarios: {results['hero_strategy_scenarios']}")
        print(f"   🎯 Villain strategy scenarios: {results['villain_strategy_scenarios']}")
        
        # Calculate coverage
        if results['unique_scenarios'] > 0:
            coverage_rate = (results['unique_scenarios'] / 330) * 100  # 330 is theoretical max
            print(f"   📈 Scenario space coverage: {coverage_rate:.1f}%")
        
        return trainer
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
        
        # Save emergency checkpoint
        emergency_checkpoint = f"natural_cfr_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_training_state(emergency_checkpoint)
        print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
        
        return trainer
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        print(f"🔍 Traceback:\n{traceback.format_exc()}")
        
        # Try to save emergency checkpoint
        try:
            emergency_checkpoint = f"natural_cfr_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            trainer.save_training_state(emergency_checkpoint)
            print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
        except:
            print("❌ Could not save emergency checkpoint")
        
        return None


def run_demo_mode(args):
    """
    Run a quick demo of the natural CFR training system.
    
    Args:
        args: Command line arguments
    """
    print("🎮 Demo Mode: Natural Game CFR Training")
    print("=" * 50)
    print("This demo will:")
    print("  🎲 Simulate natural poker games")
    print("  📊 Record emerging scenarios")
    print("  🧠 Train both hero and villain strategies")
    print("  🔄 Show co-evolution in action")
    print("")
    
    # Use demo-friendly parameters
    demo_trainer = NaturalGameCFRTrainer(
        epsilon_exploration=0.2,  # Higher exploration for demo
        min_visit_threshold=3,    # Lower threshold for demo
        tournament_survival_penalty=0.6
    )
    
    print(f"🎯 Running demo with {args.games} games...")
    
    # Run short demo training
    demo_results = demo_trainer.train(
        n_games=args.games,
        save_interval=max(args.games // 4, 100),  # Save 4 times during demo
        log_interval=max(args.games // 10, 10)    # Log 10 times during demo
    )
    
    print(f"\n🎉 Demo completed!")
    
    # Export demo results
    demo_trainer.export_natural_scenarios_csv("demo_natural_scenarios.csv")
    demo_trainer.export_strategies_csv("demo_natural_strategies.csv")
    
    print(f"\n📊 Demo Results:")
    print(f"   🎲 Games played: {demo_results['games_played']:,}")
    print(f"   📊 Scenarios discovered: {demo_results['unique_scenarios']}")
    print(f"   📁 Files created:")
    print(f"      📊 demo_natural_scenarios.csv")
    print(f"      🎯 hero_demo_natural_strategies.csv")
    print(f"      🎯 villain_demo_natural_strategies.csv")
    
    # Show some interesting statistics
    if demo_trainer.natural_scenarios:
        scenarios = demo_trainer.natural_scenarios
        showdown_rate = sum(s['showdown'] for s in scenarios) / len(scenarios)
        hero_win_rate = sum(s['hero_won'] for s in scenarios) / len(scenarios)
        three_bet_rate = sum(s['is_3bet'] for s in scenarios) / len(scenarios)
        
        print(f"\n🎯 Gameplay Statistics:")
        print(f"   👑 Hero win rate: {hero_win_rate:.1%}")
        print(f"   🎲 Showdown rate: {showdown_rate:.1%}")
        print(f"   🔥 3-bet rate: {three_bet_rate:.1%}")
    
    return demo_trainer


def run_analysis_mode(args):
    """
    Run analysis of existing training results.
    
    Args:
        args: Command line arguments
    """
    print("📊 Analysis Mode: Natural CFR Results")
    print("=" * 50)
    
    if not args.resume or not Path(args.resume).exists():
        print(f"❌ Checkpoint file required for analysis mode")
        print(f"   Use --resume <checkpoint.pkl> to specify checkpoint")
        return None
    
    # Load trainer from checkpoint
    trainer = NaturalGameCFRTrainer()
    if not trainer.load_training_state(args.resume):
        print(f"❌ Failed to load checkpoint {args.resume}")
        return None
    
    print(f"✅ Loaded training state from {args.resume}")
    
    # Analyze results
    print(f"\n📈 TRAINING ANALYSIS:")
    print(f"   🎲 Total games played: {trainer.natural_metrics['games_played']:,}")
    print(f"   📊 Unique scenarios: {trainer.natural_metrics['unique_scenarios']}")
    print(f"   🎯 Hero strategies learned: {len(trainer.strategy_sum)}")
    print(f"   🎯 Villain strategies learned: {len(trainer.villain_strategy_sum)}")
    
    if trainer.natural_scenarios:
        scenarios = trainer.natural_scenarios
        print(f"\n🎯 Gameplay Analysis:")
        print(f"   👑 Hero win rate: {sum(s['hero_won'] for s in scenarios) / len(scenarios):.1%}")
        print(f"   🎲 Showdown rate: {sum(s['showdown'] for s in scenarios) / len(scenarios):.1%}")
        print(f"   🔥 3-bet rate: {sum(s['is_3bet'] for s in scenarios) / len(scenarios):.1%}")
        
        # Analyze hand categories
        hand_categories = {}
        for scenario in scenarios:
            cat = scenario['hand_category']
            hand_categories[cat] = hand_categories.get(cat, 0) + 1
        
        print(f"\n📊 Hand Category Distribution:")
        for cat, count in sorted(hand_categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(scenarios)) * 100
            print(f"   {cat:15s}: {count:4d} games ({percentage:5.1f}%)")
    
    # Export analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.export_natural_scenarios_csv(f"analysis_scenarios_{timestamp}.csv")
    trainer.export_strategies_csv(f"analysis_strategies_{timestamp}.csv")
    
    print(f"\n📁 Analysis exported to:")
    print(f"   📊 analysis_scenarios_{timestamp}.csv")
    print(f"   🎯 hero_analysis_strategies_{timestamp}.csv")
    print(f"   🎯 villain_analysis_strategies_{timestamp}.csv")
    
    return trainer


def main():
    """Main entry point for natural CFR training."""
    parser = argparse.ArgumentParser(
        description="Natural Game CFR Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10,000 game training session
  python run_natural_cfr_training.py --games 10000
  
  # Quick demo with high exploration
  python run_natural_cfr_training.py --mode demo --games 1000
  
  # Resume from checkpoint
  python run_natural_cfr_training.py --resume checkpoint.pkl --games 5000
  
  # Analyze existing results
  python run_natural_cfr_training.py --mode analysis --resume final.pkl
  
  # Custom parameters
  python run_natural_cfr_training.py --games 20000 --epsilon 0.05 --tournament-penalty 0.4
        """
    )
    
    # Training mode
    parser.add_argument('--mode', choices=['train', 'demo', 'analysis'], default='train',
                       help='Training mode (default: train)')
    
    # Training parameters
    parser.add_argument('--games', type=int, default=10000,
                       help='Number of games to simulate (default: 10000)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Epsilon exploration rate (default: 0.1)')
    parser.add_argument('--min-visits', type=int, default=5,
                       help='Minimum visits threshold (default: 5)')
    parser.add_argument('--tournament-penalty', type=float, default=0.6,
                       help='Tournament survival penalty factor (default: 0.6)')
    
    # CFR parameters
    parser.add_argument('--enable-pruning', action='store_true', default=True,
                       help='Enable CFR pruning (default: True)')
    parser.add_argument('--regret-threshold', type=float, default=-300.0,
                       help='Regret pruning threshold (default: -300.0)')
    parser.add_argument('--strategy-threshold', type=float, default=0.001,
                       help='Strategy pruning threshold (default: 0.001)')
    
    # Logging and saving
    parser.add_argument('--save-interval', type=int, default=1000,
                       help='Save progress every N games (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Log progress every N games (default: 100)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.games <= 0:
        print("❌ Number of games must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.epsilon <= 1.0):
        print("❌ Epsilon must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.mode == 'analysis' and not args.resume:
        print("❌ Analysis mode requires --resume checkpoint file")
        sys.exit(1)
    
    # Run appropriate mode
    try:
        if args.mode == 'demo':
            trainer = run_demo_mode(args)
        elif args.mode == 'analysis':
            trainer = run_analysis_mode(args)
        else:  # train mode
            trainer = run_natural_cfr_training(args)
        
        if trainer:
            print(f"\n✅ Natural CFR training session completed successfully!")
        else:
            print(f"\n❌ Training session failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        print(f"🔍 Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()