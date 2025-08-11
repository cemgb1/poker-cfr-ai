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
from logging_config import setup_logging, log_exception, flush_logs


def run_natural_cfr_training(args):
    """
    Run natural CFR training with specified parameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        NaturalGameCFRTrainer: Trained model
    """
    # Set up logging
    logger = setup_logging("natural_cfr_training")
    
    logger.info("🚀 Natural Game CFR Training System")
    logger.info("=" * 60)
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Games to simulate: {args.games:,}")
    logger.info(f"Epsilon exploration: {args.epsilon}")
    logger.info(f"Min visit threshold: {args.min_visits}")
    logger.info(f"Tournament penalty: {args.tournament_penalty}")
    logger.info(f"Save interval: every {args.save_interval} games")
    logger.info(f"Log interval: every {args.log_interval} games")
    
    # Log initialization parameters
    logger.info("Model Initialization Parameters:")
    logger.info(f"  - enable_pruning: {args.enable_pruning}")
    logger.info(f"  - regret_pruning_threshold: {args.regret_threshold}")
    logger.info(f"  - strategy_pruning_threshold: {args.strategy_threshold}")
    logger.info(f"  - tournament_survival_penalty: {args.tournament_penalty}")
    logger.info(f"  - epsilon_exploration: {args.epsilon}")
    logger.info(f"  - min_visit_threshold: {args.min_visits}")
    
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
    logger.info("Initializing NaturalGameCFRTrainer...")
    trainer = NaturalGameCFRTrainer(
        enable_pruning=args.enable_pruning,
        regret_pruning_threshold=args.regret_threshold,
        strategy_pruning_threshold=args.strategy_threshold,
        tournament_survival_penalty=args.tournament_penalty,
        epsilon_exploration=args.epsilon,
        min_visit_threshold=args.min_visits,
        logger=logger
    )
    logger.info("Trainer initialized successfully")
    
    # Load checkpoint if specified
    if args.resume:
        logger.info(f"Attempting to resume training from checkpoint: {args.resume}")
        if Path(args.resume).exists():
            if trainer.load_training_state(args.resume):
                success_msg = f"✅ Resumed training from {args.resume}"
                logger.info(f"Successfully resumed training from {args.resume}")
                print(success_msg)
            else:
                error_msg = f"❌ Failed to load {args.resume}, starting fresh"
                logger.warning(f"Failed to load checkpoint {args.resume}, starting fresh training")
                print(error_msg)
        else:
            error_msg = f"❌ Checkpoint file {args.resume} not found, starting fresh"
            logger.warning(f"Checkpoint file {args.resume} not found, starting fresh training")
            print(error_msg)
    
    # Run training
    training_msg = f"Starting natural game simulation with {args.games} games..."
    logger.info(training_msg)
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
        
        success_msg = f"Training completed successfully in {training_duration/60:.1f} minutes"
        logger.info(success_msg)
        logger.info(f"Games per minute: {results['games_played'] / (training_duration/60):.1f}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Training time: {training_duration/60:.1f} minutes")
        print(f"🎲 Games per minute: {results['games_played'] / (training_duration/60):.1f}")
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        export_msg = f"Exporting results with timestamp {timestamp}..."
        logger.info(export_msg)
        print(f"\n📊 Exporting results...")
        
        scenarios_file = f"natural_scenarios_{timestamp}.csv"
        strategies_file = f"natural_strategies_{timestamp}.csv"
        
        trainer.export_natural_scenarios_csv(scenarios_file)
        trainer.export_strategies_csv(strategies_file)
        
        logger.info(f"Exported natural scenarios to: {scenarios_file}")
        logger.info(f"Exported strategies to: hero_{strategies_file} and villain_{strategies_file}")
        
        # Save final checkpoint (to checkpoints directory)
        final_checkpoint = f"natural_cfr_final_{timestamp}.pkl"
        trainer.save_training_state(final_checkpoint)
        logger.info(f"Saved final checkpoint: checkpoints/{final_checkpoint}")
        
        # Create performance summary
        logger.info("Creating performance summary...")
        performance_file = trainer.create_performance_summary(training_duration, output_format='csv')
        if performance_file:
            logger.info(f"Performance summary created: {performance_file}")
        
        # Create final lookup table
        logger.info("Creating final lookup table...")
        lookup_table_file = trainer.create_final_lookup_table()
        if lookup_table_file:
            logger.info(f"Final lookup table created: {lookup_table_file}")
        
        # Archive old files
        logger.info("Archiving old files...")
        archived_items = trainer.archive_old_files()
        if archived_items:
            logger.info(f"Archived {len(archived_items)} old files/folders")
        
        print(f"\n📁 Results saved:")
        print(f"   📊 Scenarios: {scenarios_file}")
        print(f"   🎯 Hero strategies: hero_{strategies_file}")
        print(f"   🎯 Villain strategies: villain_{strategies_file}")
        print(f"   💾 Final checkpoint: checkpoints/{final_checkpoint}")
        if performance_file:
            print(f"   📈 Performance summary: {performance_file}")
        if lookup_table_file:
            print(f"   📋 Final lookup table: {lookup_table_file}")
        if archived_items:
            print(f"   📦 Archived items: {len(archived_items)} files/folders to archivedfileslocation/")
        
        # Show training summary
        logger.info("TRAINING SUMMARY:")
        logger.info(f"  Total games played: {results['games_played']:,}")
        logger.info(f"  Unique scenarios discovered: {results['unique_scenarios']}")
        logger.info(f"  Hero strategy scenarios: {results['hero_strategy_scenarios']}")
        logger.info(f"  Villain strategy scenarios: {results['villain_strategy_scenarios']}")
        
        print(f"\n📈 TRAINING SUMMARY:")
        print(f"   🎲 Total games played: {results['games_played']:,}")
        print(f"   📊 Unique scenarios discovered: {results['unique_scenarios']}")
        print(f"   🎯 Hero strategy scenarios: {results['hero_strategy_scenarios']}")
        print(f"   🎯 Villain strategy scenarios: {results['villain_strategy_scenarios']}")
        
        # Calculate coverage
        if results['unique_scenarios'] > 0:
            coverage_rate = (results['unique_scenarios'] / 330) * 100  # 330 is theoretical max
            logger.info(f"Scenario space coverage: {coverage_rate:.1f}%")
            print(f"   📈 Scenario space coverage: {coverage_rate:.1f}%")
        
        flush_logs(logger)
        return trainer
        
    except KeyboardInterrupt:
        interrupt_msg = "Training interrupted by user"
        logger.warning(interrupt_msg)
        print(f"\n🛑 Training interrupted by user")
        
        # Save emergency checkpoint
        emergency_checkpoint = f"natural_cfr_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_training_state(emergency_checkpoint)
        checkpoint_msg = f"Emergency checkpoint saved: {emergency_checkpoint}"
        logger.info(checkpoint_msg)
        print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
        
        flush_logs(logger)
        return trainer
        
    except Exception as e:
        error_msg = f"Training failed with error: {e}"
        logger.error(error_msg)
        log_exception(logger, "Training failed")
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        print(f"🔍 Traceback:\n{traceback.format_exc()}")
        
        # Try to save emergency checkpoint
        try:
            emergency_checkpoint = f"natural_cfr_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            trainer.save_training_state(emergency_checkpoint)
            checkpoint_msg = f"Emergency checkpoint saved: {emergency_checkpoint}"
            logger.info(checkpoint_msg)
            print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
        except Exception as save_error:
            save_error_msg = f"Could not save emergency checkpoint: {save_error}"
            logger.error(save_error_msg)
            print("❌ Could not save emergency checkpoint")
        
        flush_logs(logger)
        return None


def run_demo_mode(args):
    """
    Run a quick demo of the natural CFR training system.
    
    Args:
        args: Command line arguments
    """
    # Set up logging for demo mode
    logger = setup_logging("natural_cfr_demo")
    
    logger.info("🎮 Demo Mode: Natural Game CFR Training")
    logger.info("=" * 50)
    logger.info("Demo configuration:")
    logger.info(f"  - Games to simulate: {args.games}")
    logger.info(f"  - Epsilon exploration: 0.2 (higher for demo)")
    logger.info(f"  - Min visit threshold: 3 (lower for demo)")
    logger.info(f"  - Tournament survival penalty: 0.6")
    
    print("🎮 Demo Mode: Natural Game CFR Training")
    print("=" * 50)
    print("This demo will:")
    print("  🎲 Simulate natural poker games")
    print("  📊 Record emerging scenarios")
    print("  🧠 Train both hero and villain strategies")
    print("  🔄 Show co-evolution in action")
    print("")
    
    # Use demo-friendly parameters
    logger.info("Initializing demo trainer with exploration-friendly parameters...")
    demo_trainer = NaturalGameCFRTrainer(
        epsilon_exploration=0.2,  # Higher exploration for demo
        min_visit_threshold=3,    # Lower threshold for demo
        tournament_survival_penalty=0.6,
        logger=logger
    )
    logger.info("Demo trainer initialized successfully")
    
    demo_msg = f"Running demo with {args.games} games..."
    logger.info(demo_msg)
    print(f"🎯 Running demo with {args.games} games...")
    
    # Run short demo training
    demo_results = demo_trainer.train(
        n_games=args.games,
        save_interval=max(args.games // 4, 100),  # Save 4 times during demo
        log_interval=max(args.games // 10, 10)    # Log 10 times during demo
    )
    
    logger.info("Demo training completed successfully")
    print(f"\n🎉 Demo completed!")
    
    # Export demo results
    logger.info("Exporting demo results...")
    demo_trainer.export_natural_scenarios_csv("demo_natural_scenarios.csv")
    demo_trainer.export_strategies_csv("demo_natural_strategies.csv")
    
    # Create demo performance summary
    demo_performance_file = demo_trainer.create_performance_summary(training_duration=0.0, output_format='csv')
    
    # Create demo lookup table
    demo_lookup_file = demo_trainer.create_final_lookup_table("demo_final_lookup_table.csv")
    
    logger.info("Demo results exported successfully")
    
    logger.info("DEMO RESULTS:")
    logger.info(f"  Games played: {demo_results['games_played']:,}")
    logger.info(f"  Scenarios discovered: {demo_results['unique_scenarios']}")
    
    print(f"\n📊 Demo Results:")
    print(f"   🎲 Games played: {demo_results['games_played']:,}")
    print(f"   📊 Scenarios discovered: {demo_results['unique_scenarios']}")
    print(f"   📁 Files created:")
    print(f"      📊 demo_natural_scenarios.csv")
    print(f"      🎯 hero_demo_natural_strategies.csv")
    print(f"      🎯 villain_demo_natural_strategies.csv")
    if demo_performance_file:
        print(f"      📈 {demo_performance_file}")
    if demo_lookup_file:
        print(f"      📋 {demo_lookup_file}")
    
    # Show some interesting statistics
    if demo_trainer.natural_scenarios:
        scenarios = demo_trainer.natural_scenarios
        showdown_rate = sum(s['showdown'] for s in scenarios) / len(scenarios)
        hero_win_rate = sum(s['hero_won'] for s in scenarios) / len(scenarios)
        three_bet_rate = sum(s['is_3bet'] for s in scenarios) / len(scenarios)
        
        logger.info("GAMEPLAY STATISTICS:")
        logger.info(f"  Hero win rate: {hero_win_rate:.1%}")
        logger.info(f"  Showdown rate: {showdown_rate:.1%}")
        logger.info(f"  3-bet rate: {three_bet_rate:.1%}")
        
        print(f"\n🎯 Gameplay Statistics:")
        print(f"   👑 Hero win rate: {hero_win_rate:.1%}")
        print(f"   🎲 Showdown rate: {showdown_rate:.1%}")
        print(f"   🔥 3-bet rate: {three_bet_rate:.1%}")
    
    flush_logs(logger)
    return demo_trainer


def run_analysis_mode(args):
    """
    Run analysis of existing training results.
    
    Args:
        args: Command line arguments
    """
    # Set up logging for analysis mode
    logger = setup_logging("natural_cfr_analysis")
    
    logger.info("📊 Analysis Mode: Natural CFR Results")
    logger.info("=" * 50)
    
    print("📊 Analysis Mode: Natural CFR Results")
    print("=" * 50)
    
    if not args.resume or not Path(args.resume).exists():
        error_msg = f"Checkpoint file required for analysis mode - {args.resume}"
        logger.error(error_msg)
        print(f"❌ Checkpoint file required for analysis mode")
        print(f"   Use --resume <checkpoint.pkl> to specify checkpoint")
        return None
    
    # Load trainer from checkpoint
    logger.info(f"Loading training state from: {args.resume}")
    trainer = NaturalGameCFRTrainer(logger=logger)
    if not trainer.load_training_state(args.resume):
        error_msg = f"Failed to load checkpoint {args.resume}"
        logger.error(error_msg)
        print(f"❌ Failed to load checkpoint {args.resume}")
        return None
    
    logger.info(f"Successfully loaded training state from {args.resume}")
    print(f"✅ Loaded training state from {args.resume}")
    
    # Analyze results
    logger.info("TRAINING ANALYSIS:")
    logger.info(f"  Total games played: {trainer.natural_metrics['games_played']:,}")
    logger.info(f"  Unique scenarios: {trainer.natural_metrics['unique_scenarios']}")
    logger.info(f"  Hero strategies learned: {len(trainer.strategy_sum)}")
    logger.info(f"  Villain strategies learned: {len(trainer.villain_strategy_sum)}")
    
    print(f"\n📈 TRAINING ANALYSIS:")
    print(f"   🎲 Total games played: {trainer.natural_metrics['games_played']:,}")
    print(f"   📊 Unique scenarios: {trainer.natural_metrics['unique_scenarios']}")
    print(f"   🎯 Hero strategies learned: {len(trainer.strategy_sum)}")
    print(f"   🎯 Villain strategies learned: {len(trainer.villain_strategy_sum)}")
    
    if trainer.natural_scenarios:
        scenarios = trainer.natural_scenarios
        hero_win_rate = sum(s['hero_won'] for s in scenarios) / len(scenarios)
        showdown_rate = sum(s['showdown'] for s in scenarios) / len(scenarios)
        three_bet_rate = sum(s['is_3bet'] for s in scenarios) / len(scenarios)
        
        logger.info("GAMEPLAY ANALYSIS:")
        logger.info(f"  Hero win rate: {hero_win_rate:.1%}")
        logger.info(f"  Showdown rate: {showdown_rate:.1%}")
        logger.info(f"  3-bet rate: {three_bet_rate:.1%}")
        
        print(f"\n🎯 Gameplay Analysis:")
        print(f"   👑 Hero win rate: {hero_win_rate:.1%}")
        print(f"   🎲 Showdown rate: {showdown_rate:.1%}")
        print(f"   🔥 3-bet rate: {three_bet_rate:.1%}")
        
        # Analyze hand categories
        hand_categories = {}
        for scenario in scenarios:
            cat = scenario['hand_category']
            hand_categories[cat] = hand_categories.get(cat, 0) + 1
        
        logger.info("HAND CATEGORY DISTRIBUTION:")
        print(f"\n📊 Hand Category Distribution:")
        for cat, count in sorted(hand_categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(scenarios)) * 100
            logger.info(f"  {cat}: {count} games ({percentage:.1f}%)")
            print(f"   {cat:15s}: {count:4d} games ({percentage:5.1f}%)")
    
    # Export analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenarios_file = f"analysis_scenarios_{timestamp}.csv"
    strategies_file = f"analysis_strategies_{timestamp}.csv"
    
    logger.info(f"Exporting analysis results with timestamp {timestamp}...")
    trainer.export_natural_scenarios_csv(scenarios_file)
    trainer.export_strategies_csv(strategies_file)
    
    # Create analysis performance summary
    analysis_performance_file = trainer.create_performance_summary(training_duration=0.0, output_format='csv')
    
    # Create analysis lookup table
    analysis_lookup_file = trainer.create_final_lookup_table(f"analysis_final_lookup_table_{timestamp}.csv")
    
    logger.info(f"Analysis exported to:")
    logger.info(f"  Scenarios: {scenarios_file}")
    logger.info(f"  Hero strategies: hero_{strategies_file}")
    logger.info(f"  Villain strategies: villain_{strategies_file}")
    if analysis_performance_file:
        logger.info(f"  Performance summary: {analysis_performance_file}")
    if analysis_lookup_file:
        logger.info(f"  Lookup table: {analysis_lookup_file}")
    
    print(f"\n📁 Analysis exported to:")
    print(f"   📊 {scenarios_file}")
    print(f"   🎯 hero_{strategies_file}")
    print(f"   🎯 villain_{strategies_file}")
    if analysis_performance_file:
        print(f"   📈 {analysis_performance_file}")
    if analysis_lookup_file:
        print(f"   📋 {analysis_lookup_file}")
    
    flush_logs(logger)
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
    
    # Set up main logger for argument validation and main execution
    logger = setup_logging("natural_cfr_main")
    
    # Log the command line arguments
    logger.info("Natural CFR Training Started")
    logger.info(f"Command line arguments: {vars(args)}")
    
    # Validate arguments
    if args.games <= 0:
        error_msg = "Number of games must be positive"
        logger.error(error_msg)
        print("❌ Number of games must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.epsilon <= 1.0):
        error_msg = "Epsilon must be between 0.0 and 1.0"
        logger.error(error_msg)
        print("❌ Epsilon must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.mode == 'analysis' and not args.resume:
        error_msg = "Analysis mode requires --resume checkpoint file"
        logger.error(error_msg)
        print("❌ Analysis mode requires --resume checkpoint file")
        sys.exit(1)
    
    logger.info("Arguments validated successfully")
    
    # Run appropriate mode
    try:
        logger.info(f"Starting {args.mode} mode...")
        
        if args.mode == 'demo':
            trainer = run_demo_mode(args)
        elif args.mode == 'analysis':
            trainer = run_analysis_mode(args)
        else:  # train mode
            trainer = run_natural_cfr_training(args)
        
        if trainer:
            success_msg = f"Natural CFR {args.mode} session completed successfully!"
            logger.info(success_msg)
            print(f"\n✅ Natural CFR training session completed successfully!")
        else:
            error_msg = f"{args.mode} session failed"
            logger.error(error_msg)
            print(f"\n❌ Training session failed")
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"Unexpected error in {args.mode} mode: {e}"
        logger.error(error_msg)
        log_exception(logger, f"Unexpected error in {args.mode} mode")
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        print(f"🔍 Traceback:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Ensure all logs are flushed
        flush_logs(logger)


if __name__ == "__main__":
    main()