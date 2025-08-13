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
import multiprocessing
import os
import glob
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from natural_game_cfr_trainer import NaturalGameCFRTrainer
from logging_config import setup_logging, log_exception, flush_logs


def find_latest_checkpoint():
    """
    Automatically find the latest checkpoint file in checkpoints/ directory or repo root.
    
    Returns:
        str or None: Path to the latest checkpoint file, or None if none found
    """
    checkpoint_patterns = [
        "checkpoints/*.pkl",  # Check checkpoints directory first
        "*.pkl"               # Then check repo root
    ]
    
    latest_checkpoint = None
    latest_time = 0
    
    for pattern in checkpoint_patterns:
        for checkpoint_file in glob.glob(pattern):
            file_path = Path(checkpoint_file)
            if file_path.is_file():
                file_time = file_path.stat().st_mtime
                if file_time > latest_time:
                    latest_time = file_time
                    latest_checkpoint = str(file_path)
    
    return latest_checkpoint


def multiprocessing_worker(worker_id, games_per_worker, args):
    """
    Worker function for multiprocessing training.
    
    Args:
        worker_id (int): Unique identifier for this worker
        games_per_worker (int): Number of games this worker should process
        args: Command line arguments
        
    Returns:
        dict: Results from this worker including file paths and metrics
    """
    # Set up worker-specific logging
    worker_logger = setup_logging(f"natural_cfr_worker_{worker_id}")
    
    worker_logger.info(f"🚀 Worker {worker_id} starting with {games_per_worker} games")
    
    # Initialize trainer for this worker
    trainer = NaturalGameCFRTrainer(
        enable_pruning=args.enable_pruning,
        regret_pruning_threshold=args.regret_threshold,
        strategy_pruning_threshold=args.strategy_threshold,
        tournament_survival_penalty=args.tournament_penalty,
        epsilon_exploration=args.epsilon,
        min_visit_threshold=args.min_visits,
        logger=worker_logger,
        export_scope=args.export_scope,
        export_window_games=args.export_window_games,
        export_min_visits=args.export_min_visits
    )
    
    # Load checkpoint if this is worker 0 and resume is specified or auto-found
    if worker_id == 0 and hasattr(args, '_resume_checkpoint') and args._resume_checkpoint:
        worker_logger.info(f"Worker {worker_id} loading checkpoint: {args._resume_checkpoint}")
        if Path(args._resume_checkpoint).exists():
            if trainer.load_training_state(args._resume_checkpoint):
                worker_logger.info(f"✅ Worker {worker_id} resumed from {args._resume_checkpoint}")
            else:
                worker_logger.warning(f"❌ Worker {worker_id} failed to load {args._resume_checkpoint}")
        else:
            worker_logger.warning(f"❌ Worker {worker_id} checkpoint file not found: {args._resume_checkpoint}")
    
    # Run training for this worker
    training_start = time.time()
    results = trainer.train(
        n_games=games_per_worker,
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )
    training_end = time.time()
    training_duration = training_end - training_start
    
    # Generate worker-specific filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export worker-specific results
    scenarios_file = f"worker_{worker_id}_natural_scenarios_{timestamp}.csv"
    strategies_file = f"worker_{worker_id}_natural_strategies_{timestamp}.csv"
    checkpoint_file = f"worker_{worker_id}_natural_cfr_{timestamp}.pkl"
    
    trainer.export_natural_scenarios_csv(scenarios_file)
    trainer.export_strategies_csv(strategies_file)
    trainer.save_training_state(checkpoint_file)
    
    worker_logger.info(f"✅ Worker {worker_id} completed {games_per_worker} games in {training_duration/60:.1f} minutes")
    worker_logger.info(f"📁 Worker {worker_id} saved results:")
    worker_logger.info(f"   📊 Scenarios: {scenarios_file}")
    worker_logger.info(f"   🎯 Strategies: hero_{strategies_file} and villain_{strategies_file}")
    worker_logger.info(f"   💾 Checkpoint: checkpoints/{checkpoint_file}")
    
    # Return results for aggregation
    flush_logs(worker_logger)
    return {
        'worker_id': worker_id,
        'games_played': results['games_played'],
        'unique_scenarios': results['unique_scenarios'],
        'hero_strategy_scenarios': results['hero_strategy_scenarios'],
        'villain_strategy_scenarios': results['villain_strategy_scenarios'],
        'training_duration': training_duration,
        'scenarios_file': scenarios_file,
        'strategies_file': strategies_file,
        'checkpoint_file': checkpoint_file,
        'hero_strategies_file': f"hero_{strategies_file}",
        'villain_strategies_file': f"villain_{strategies_file}"
    }


def aggregate_worker_results(worker_results, args):
    """
    Aggregate results from all workers and create summary.
    
    Args:
        worker_results (list): List of worker result dictionaries
        args: Command line arguments
        
    Returns:
        dict: Aggregated results
    """
    logger = setup_logging("natural_cfr_aggregator")
    
    logger.info("📊 Aggregating results from all workers...")
    
    # Calculate totals
    total_games = sum(r['games_played'] for r in worker_results)
    total_unique_scenarios = sum(r['unique_scenarios'] for r in worker_results) 
    total_hero_scenarios = sum(r['hero_strategy_scenarios'] for r in worker_results)
    total_villain_scenarios = sum(r['villain_strategy_scenarios'] for r in worker_results)
    total_duration = max(r['training_duration'] for r in worker_results)  # Use max since parallel
    
    # Create aggregated results dictionary
    aggregated = {
        'total_workers': len(worker_results),
        'total_games': total_games,
        'total_unique_scenarios': total_unique_scenarios,
        'total_hero_scenarios': total_hero_scenarios,
        'total_villain_scenarios': total_villain_scenarios,
        'total_duration': total_duration,
        'games_per_minute': total_games / (total_duration / 60) if total_duration > 0 else 0,
        'worker_files': {
            'scenarios': [r['scenarios_file'] for r in worker_results],
            'hero_strategies': [r['hero_strategies_file'] for r in worker_results],
            'villain_strategies': [r['villain_strategies_file'] for r in worker_results],
            'checkpoints': [r['checkpoint_file'] for r in worker_results]
        }
    }
    
    logger.info("📈 MULTI-WORKER TRAINING SUMMARY:")
    logger.info(f"  Total workers: {aggregated['total_workers']}")
    logger.info(f"  Total games played: {aggregated['total_games']:,}")
    logger.info(f"  Total training time: {aggregated['total_duration']/60:.1f} minutes")
    logger.info(f"  Games per minute: {aggregated['games_per_minute']:.1f}")
    logger.info(f"  Total unique scenarios: {aggregated['total_unique_scenarios']}")
    logger.info(f"  Total hero strategy scenarios: {aggregated['total_hero_scenarios']}")
    logger.info(f"  Total villain strategy scenarios: {aggregated['total_villain_scenarios']}")
    
    # Calculate coverage
    if aggregated['total_unique_scenarios'] > 0:
        coverage_rate = (aggregated['total_unique_scenarios'] / 330) * 100  # 330 is theoretical max
        logger.info(f"  Scenario space coverage: {coverage_rate:.1f}%")
        aggregated['coverage_rate'] = coverage_rate
    
    logger.info("📁 Worker output files:")
    for i, result in enumerate(worker_results):
        logger.info(f"  Worker {result['worker_id']}:")
        logger.info(f"    📊 Scenarios: {result['scenarios_file']}")
        logger.info(f"    🎯 Hero strategies: {result['hero_strategies_file']}")
        logger.info(f"    🎯 Villain strategies: {result['villain_strategies_file']}")
        logger.info(f"    💾 Checkpoint: checkpoints/{result['checkpoint_file']}")
    
    flush_logs(logger)
    return aggregated


def run_natural_cfr_training(args):
    """
    Run natural CFR training with specified parameters.
    Supports both single-process and multi-process training based on --workers argument.
    
    Args:
        args: Command line arguments
        
    Returns:
        NaturalGameCFRTrainer or dict: Trained model (single-process) or aggregated results (multi-process)
    """
    # Set up logging
    logger = setup_logging("natural_cfr_training")
    
    logger.info("🚀 Natural Game CFR Training System")
    logger.info("=" * 60)
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Games to simulate: {args.games:,}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Epsilon exploration: {args.epsilon}")
    logger.info(f"Min visit threshold: {args.min_visits}")
    logger.info(f"Tournament penalty: {args.tournament_penalty}")
    logger.info(f"Save interval: every {args.save_interval} games")
    logger.info(f"Log interval: every {args.log_interval} games")
    
    # Check for automatic checkpoint resume if --resume not specified
    resume_checkpoint = args.resume
    if not resume_checkpoint:
        auto_checkpoint = find_latest_checkpoint()
        if auto_checkpoint:
            resume_checkpoint = auto_checkpoint
            logger.info(f"🔍 Auto-discovered checkpoint: {resume_checkpoint}")
            print(f"🔍 Auto-discovered checkpoint: {resume_checkpoint}")
        else:
            logger.info("🔍 No checkpoints found for auto-resume")
    
    # Store the checkpoint to use (for workers)
    args._resume_checkpoint = resume_checkpoint
    
    # Multi-process training (only for train mode and workers > 1)
    if args.workers > 1 and args.mode == 'train':
        logger.info(f"🔄 Starting multi-process training with {args.workers} workers")
        print(f"🔄 Starting multi-process training with {args.workers} workers")
        
        # Ensure checkpoints directory exists
        os.makedirs("checkpoints", exist_ok=True)
        
        # Calculate games per worker
        games_per_worker = args.games // args.workers
        remaining_games = args.games % args.workers
        
        logger.info(f"📊 Games distribution:")
        logger.info(f"   Games per worker: {games_per_worker}")
        logger.info(f"   Remaining games: {remaining_games} (will be distributed to first workers)")
        
        print(f"📊 Games distribution:")
        print(f"   Games per worker: {games_per_worker}")
        if remaining_games > 0:
            print(f"   Extra games: {remaining_games} (distributed to first workers)")
        
        # Create worker arguments
        worker_args = []
        for worker_id in range(args.workers):
            worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
            worker_args.append((worker_id, worker_games, args))
        
        # Start multi-process training
        training_start = time.time()
        worker_results = []
        
        logger.info(f"🚀 Launching {args.workers} workers...")
        print(f"🚀 Launching {args.workers} workers...")
        
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Submit all worker tasks
                future_to_worker = {
                    executor.submit(multiprocessing_worker, worker_id, worker_games, args): worker_id
                    for worker_id, worker_games, _ in worker_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_worker):
                    worker_id = future_to_worker[future]
                    try:
                        result = future.result()
                        worker_results.append(result)
                        logger.info(f"✅ Worker {worker_id} completed successfully")
                        print(f"✅ Worker {worker_id} completed")
                    except Exception as e:
                        error_msg = f"❌ Worker {worker_id} failed: {e}"
                        logger.error(error_msg)
                        print(error_msg)
                        # Continue with other workers
        
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            print(f"\n🛑 Training interrupted by user")
            return None
        
        training_end = time.time()
        total_duration = training_end - training_start
        
        if not worker_results:
            error_msg = "❌ All workers failed"
            logger.error(error_msg)
            print(error_msg)
            return None
        
        # Aggregate results from all workers
        aggregated_results = aggregate_worker_results(worker_results, args)
        
        print(f"\n🎉 Multi-worker training completed!")
        print(f"⏱️  Total time: {total_duration/60:.1f} minutes")
        print(f"🔄 Workers completed: {len(worker_results)}/{args.workers}")
        print(f"🎲 Total games: {aggregated_results['total_games']:,}")
        print(f"📊 Total unique scenarios: {aggregated_results['total_unique_scenarios']}")
        print(f"⚡ Effective games per minute: {aggregated_results['games_per_minute']:.1f}")
        
        if 'coverage_rate' in aggregated_results:
            print(f"📈 Scenario space coverage: {aggregated_results['coverage_rate']:.1f}%")
        
        print(f"\n📁 Worker output files:")
        for i, result in enumerate(worker_results):
            print(f"   Worker {result['worker_id']}:")
            print(f"      📊 {result['scenarios_file']}")
            print(f"      🎯 {result['hero_strategies_file']}")
            print(f"      🎯 {result['villain_strategies_file']}")
            print(f"      💾 checkpoints/{result['checkpoint_file']}")
        
        logger.info("Multi-process training completed successfully")
        flush_logs(logger)
        return aggregated_results
    
    # Single-process training (original logic)
    else:
        if args.workers > 1:
            logger.info(f"🔄 Falling back to single-process training (mode: {args.mode})")
            print(f"🔄 Single-process mode (mode: {args.mode})")
        
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
        print(f"🎯 Games to simulate: {args.games:,} (each game = multiple hands until bust)")
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
            logger=logger,
            export_scope=args.export_scope,
            export_window_games=args.export_window_games,
            export_min_visits=args.export_min_visits
        )
        logger.info("Trainer initialized successfully")
        
        # Load checkpoint if specified or auto-discovered
        if resume_checkpoint:
            logger.info(f"Attempting to resume training from checkpoint: {resume_checkpoint}")
            if Path(resume_checkpoint).exists():
                if trainer.load_training_state(resume_checkpoint):
                    success_msg = f"✅ Resumed training from {resume_checkpoint}"
                    logger.info(f"Successfully resumed training from {resume_checkpoint}")
                    print(success_msg)
                else:
                    error_msg = f"❌ Failed to load {resume_checkpoint}, starting fresh"
                    logger.warning(f"Failed to load checkpoint {resume_checkpoint}, starting fresh training")
                    print(error_msg)
            else:
                error_msg = f"❌ Checkpoint file {resume_checkpoint} not found, starting fresh"
                logger.warning(f"Checkpoint file {resume_checkpoint} not found, starting fresh training")
                print(error_msg)
        
        # Run training (original single-process logic continues...)
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
        tournament_survival_penalty=0.2,
        logger=logger,
        export_scope=args.export_scope,
        export_window_games=args.export_window_games,
        export_min_visits=args.export_min_visits
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
  # Run 10,000 games training session (single-process) - each game plays multiple hands until bust
  python run_natural_cfr_training.py --games 10000 --workers 1
  
  # Run multi-core training with 4 workers - distributes games across workers
  python run_natural_cfr_training.py --games 20000 --workers 4
  
  # Auto-resume from latest checkpoint with 8 workers (default)
  python run_natural_cfr_training.py --games 50000
  
  # Quick demo with high exploration - plays 1000 complete games
  python run_natural_cfr_training.py --mode demo --games 1000
  
  # Resume from specific checkpoint
  python run_natural_cfr_training.py --resume checkpoint.pkl --games 5000 --workers 2
  
  # Analyze existing results
  python run_natural_cfr_training.py --mode analysis --resume final.pkl
  
  # Custom parameters with multi-core
  python run_natural_cfr_training.py --games 20000 --epsilon 0.05 --tournament-penalty 0.4 --workers 6

Note: The --games parameter now specifies the number of complete poker games to simulate.
Each game consists of multiple hands played with fixed stack and blind sizes until one player is busted.
        """
    )
    
    # Training mode
    parser.add_argument('--mode', choices=['train', 'demo', 'analysis'], default='train',
                       help='Training mode (default: train)')
    
    # Training parameters
    parser.add_argument('--games', type=int, default=10000,
                       help='Number of games to simulate (each game consists of multiple hands until one player is busted) (default: 10000)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers for training (default: 8, use 1 for single-process)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Epsilon exploration rate (default: 0.1)')
    parser.add_argument('--min-visits', type=int, default=5,
                       help='Minimum visits threshold (default: 5)')
    parser.add_argument('--tournament-penalty', type=float, default=0.2,
                       help='Tournament survival penalty factor (default: 0.2)')
    
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
    
    # Export configuration
    parser.add_argument('--export-scope', choices=['cumulative', 'window'], default='cumulative',
                       help='Export scope for scenario lookup table: cumulative (all games) or window (last N games) (default: cumulative)')
    parser.add_argument('--export-window-games', type=int, default=2000,
                       help='Number of recent games to include in window export (default: 2000)')
    parser.add_argument('--export-min-visits', type=int, default=1,
                       help='Minimum visits required for scenario to be included in export (default: 1)')
    
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
    
    if args.workers <= 0:
        error_msg = "Number of workers must be positive"
        logger.error(error_msg)
        print("❌ Number of workers must be positive")
        sys.exit(1)
    
    if args.workers > multiprocessing.cpu_count():
        warning_msg = f"Warning: {args.workers} workers requested but only {multiprocessing.cpu_count()} CPUs available"
        logger.warning(warning_msg)
        print(f"⚠️  {warning_msg}")
    
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
    
    # Validate export arguments
    if args.export_window_games <= 0:
        error_msg = "Export window games must be positive"
        logger.error(error_msg)
        print("❌ Export window games must be positive")
        sys.exit(1)
        
    if args.export_min_visits < 0:
        error_msg = "Export min visits must be non-negative"
        logger.error(error_msg)
        print("❌ Export min visits must be non-negative")
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