#!/usr/bin/env python3
# run_simplified_cfr_training.py - Command-line runner for simplified CFR training

"""
Command-line runner for Simplified CFR Training System

This script provides a simple interface to run the simplified CFR trainer with
direct hole card scenarios and preflop-only simulation.

Features:
- Monte Carlo scenario generation (random hole cards + stack sizes)
- Preflop-only simulation with immediate showdown
- Coverage tracking for all 1326 hole card combinations
- Heads-up match mode option
- Checkpointing and resuming
- Comprehensive logging and reporting

Usage Examples:
  python run_simplified_cfr_training.py --iterations 1000
  python run_simplified_cfr_training.py --iterations 5000 --heads-up --starting-stack 50
  python run_simplified_cfr_training.py --resume checkpoint_1000.pkl --iterations 2000
"""

import argparse
import sys
import time
from pathlib import Path

from simplified_cfr_trainer import SimplifiedCFRTrainer
from simplified_scenario_generator import get_scenario_coverage_stats
from logging_config import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simplified CFR Training System for Preflop Poker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --iterations 1000
  %(prog)s --iterations 5000 --heads-up --starting-stack 50
  %(prog)s --resume checkpoint_1000.pkl --iterations 2000
        """
    )
    
    # Training parameters
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=1000,
        help="Number of training iterations (default: 1000)"
    )
    
    parser.add_argument(
        "--epsilon", 
        type=float, 
        default=0.1,
        help="Exploration rate for epsilon-greedy (default: 0.1)"
    )
    
    parser.add_argument(
        "--min-visits", 
        type=int, 
        default=5,
        help="Minimum visits before exploitation (default: 5)"
    )
    
    # Heads-up mode
    parser.add_argument(
        "--heads-up",
        action="store_true",
        help="Enable heads-up match mode (play until one player is busted)"
    )
    
    parser.add_argument(
        "--starting-stack",
        type=int,
        default=100,
        help="Starting stack size in big blinds for heads-up mode (default: 100)"
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint file"
    )
    
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N iterations (default: 1000)"
    )
    
    # Output
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="simplified_cfr",
        help="Prefix for output files (default: simplified_cfr)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output"
    )
    
    return parser.parse_args()


def print_header():
    """Print training system header."""
    print("🚀 Simplified CFR Training System")
    print("=" * 60)
    print("📊 Features:")
    print("   • Direct hole card scenarios (no hand categories)")
    print("   • Preflop-only simulation with immediate showdown")
    print("   • Monte Carlo scenario generation")
    print("   • Full coverage tracking (1326 hole card combinations)")
    print("   • Heads-up match mode option")
    print("=" * 60)


def print_training_config(args, trainer):
    """Print training configuration."""
    print("⚙️  Training Configuration:")
    print(f"   🎲 Iterations: {args.iterations:,}")
    print(f"   🔍 Epsilon exploration: {args.epsilon}")
    print(f"   📊 Min visits threshold: {args.min_visits}")
    print(f"   🎮 Heads-up mode: {'Enabled' if args.heads_up else 'Disabled'}")
    if args.heads_up:
        print(f"   💰 Starting stacks: {args.starting_stack}bb each")
    print(f"   💾 Save interval: {args.save_interval} iterations")
    print(f"   📁 Output prefix: {args.output_prefix}")
    print()


def print_progress_header():
    """Print progress tracking header."""
    print("📈 Training Progress:")
    print("   Iteration | Scenarios | Coverage | Rate/min | Status")
    print("   " + "-" * 50)


def run_training(args):
    """Run the training process."""
    logger = get_logger("run_simplified_cfr_training")
    
    # Print header
    if not args.quiet:
        print_header()
    
    # Initialize trainer
    trainer = SimplifiedCFRTrainer(
        epsilon_exploration=args.epsilon,
        min_visit_threshold=args.min_visits,
        starting_stack_bb=args.starting_stack,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            # Try in checkpoints directory
            checkpoint_path = Path("checkpoints") / args.resume
            if not checkpoint_path.exists():
                print(f"❌ Checkpoint file not found: {args.resume}")
                return 1
        
        trainer.load_checkpoint(checkpoint_path)
        print(f"📂 Resumed from checkpoint: {checkpoint_path}")
        print(f"   Previous iterations: {trainer.iterations_completed}")
        print(f"   Previous scenarios: {len(trainer.visited_scenarios)}")
        
        if args.heads_up:
            print(f"   Stack state: Hero={trainer.hero_stack}bb, Villain={trainer.villain_stack}bb")
        print()
    
    # Print configuration
    if not args.quiet:
        print_training_config(args, trainer)
        print_progress_header()
    
    # Run training
    start_time = time.time()
    try:
        summary = trainer.train(
            num_iterations=args.iterations,
            heads_up_mode=args.heads_up,
            save_interval=args.save_interval
        )
        
        # Training completed successfully
        training_time = time.time() - start_time
        
        if not args.quiet:
            print()
            print("🎉 Training Completed Successfully!")
            print("=" * 60)
            print(f"⏱️  Total time: {training_time/60:.1f} minutes")
            print(f"🎲 Iterations completed: {summary['iterations_completed']:,}")
            print(f"📊 Total iterations: {summary['total_iterations']:,}")
            print(f"🎯 Scenarios visited: {summary['scenarios_visited']:,}")
            
            coverage = summary['coverage_stats']
            print(f"📈 Coverage: {coverage['coverage_percent']:.1f}% "
                  f"({coverage['unique_hole_cards_visited']}/{coverage['total_possible_combinations']} combinations)")
            
            if args.heads_up and summary['hands_played']:
                print(f"🃏 Hands played: {summary['hands_played']:,}")
                stacks = summary['final_stacks']
                print(f"💰 Final stacks: Hero={stacks['hero']}bb, Villain={stacks['villain']}bb")
                
                if stacks['hero'] > stacks['villain']:
                    print("🏆 Hero wins the heads-up match!")
                elif stacks['villain'] > stacks['hero']:
                    print("🏆 Villain wins the heads-up match!")
                else:
                    print("🤝 Match ended in a tie!")
        
        # Save final checkpoint
        final_checkpoint = trainer.save_checkpoint(f"{args.output_prefix}_final")
        
        # Export results
        exported_files = trainer.export_results(args.output_prefix)
        
        if not args.quiet:
            print()
            print("📁 Output Files:")
            print(f"   💾 Final checkpoint: {final_checkpoint}")
            for file_type, filename in exported_files.items():
                print(f"   📊 {file_type.title()}: {filename}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        
        # Save emergency checkpoint
        emergency_checkpoint = trainer.save_checkpoint(f"{args.output_prefix}_interrupted")
        print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
        
        # Export partial results
        exported_files = trainer.export_results(f"{args.output_prefix}_partial")
        print("📊 Partial results exported")
        
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.iterations <= 0:
        print("❌ Error: iterations must be positive")
        return 1
    
    if not 0 <= args.epsilon <= 1:
        print("❌ Error: epsilon must be between 0 and 1")
        return 1
    
    if args.min_visits < 1:
        print("❌ Error: min-visits must be positive")
        return 1
    
    if args.starting_stack < 10:
        print("❌ Error: starting-stack must be at least 10bb")
        return 1
    
    # Create output directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Run training
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())