#!/usr/bin/env python3
"""
Cross-Domain Training Experiment Runner

This script runs cross-domain training experiments across all Amazon product categories.
Designed to be run in the background for long-running experiments.

Usage:
    python run_cross_domain_experiments.py [options]
    
    # Run in background with nohup:
    nohup python run_cross_domain_experiments.py > experiment_output.log 2>&1 &
    
    # Run with custom parameters:
    python run_cross_domain_experiments.py --epochs 100 --batch-size 32
"""

import argparse
import sys
import os
import signal
import logging
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Train_AutoRec import (
    run_multi_domain_experiments, 
    create_default_config, 
    create_default_param_grid
)

def setup_logging(log_dir="./logs"):
    """Setup logging for the experiment runner."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_runner_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    logger.info("Current experiment will complete before shutdown.")
    sys.exit(0)

def create_domain_pairs(base_data_dir="../../data", subset=None):
    """
    Create all possible domain pairs or a subset.
    
    Args:
        base_data_dir: Base directory containing the CSV files
        subset: Optional list of domain indices to use (e.g., [0, 1, 2] for first 3 domains)
    
    Returns:
        List of (source_path, target_path) tuples
    """
    domains = [
        "Amazon_Instant_Video",
        "Apps_for_Android", 
        "Automotive",
        "Baby",
        "Beauty",
        "Books",
        "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Digital_Music",
        "Electronics",
        "Grocery_and_Gourmet_Food",
        "Health_and_Personal_Care",
        "Home_and_Kitchen",
        "Kindle_Store",
        "Movies_and_TV",
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Sports_and_Outdoors",
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
        "Video_Games"
    ]
    
    # Use subset if specified
    if subset is not None:
        domains = [domains[i] for i in subset if i < len(domains)]
    
    # Generate all possible domain pairs (source -> target)
    domain_pairs = []
    for source_domain in domains:
        for target_domain in domains:
            if source_domain != target_domain:  # Don't pair domain with itself
                source_path = f"{base_data_dir}/ratings_{source_domain}.csv"
                target_path = f"{base_data_dir}/ratings_{target_domain}.csv"
                domain_pairs.append((source_path, target_path))
    
    return domain_pairs, domains

def create_config_from_args(args):
    """Create configuration dictionary from command line arguments."""
    config = create_default_config()
    
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['lr'] = args.learning_rate
    if args.weight_decay:
        config['wd'] = args.weight_decay
    if args.n_factors:
        config['n_factors'] = args.n_factors
    
    return config

def create_param_grid_from_args(args):
    """Create parameter grid from command line arguments for grid search."""
    param_grid = create_default_param_grid()
    
    if args.epochs:
        param_grid['epochs'] = [args.epochs]
    if args.batch_size:
        param_grid['batch_size'] = [args.batch_size]
    if args.learning_rate:
        param_grid['lr'] = [args.learning_rate]
    if args.weight_decay:
        param_grid['wd'] = [args.weight_decay]
    if args.n_factors:
        param_grid['n_factors'] = [args.n_factors]
    
    return param_grid

def main():
    parser = argparse.ArgumentParser(description='Run cross-domain training experiments')
    
    # Experiment parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, help='Weight decay')
    parser.add_argument('--n-factors', type=int, help='Number of latent factors')
    
    # Experiment type
    parser.add_argument('--grid-search', action='store_true', 
                       help='Run grid search instead of single config')
    
    # Data and output paths
    parser.add_argument('--data-dir', default='../../data', 
                       help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='./models', 
                       help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', default='./logs', 
                       help='Directory to save training logs')
    
    # Domain selection
    parser.add_argument('--subset', type=str, 
                       help='Comma-separated indices of domains to use (e.g., "0,1,2,3")')
    parser.add_argument('--max-pairs', type=int,
                       help='Maximum number of domain pairs to process')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                       help='Print domain pairs without running experiments')
    parser.add_argument('--resume-from', type=int, default=0,
                       help='Resume from specific domain pair index')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting cross-domain training experiment runner")
    logger.info(f"Arguments: {vars(args)}")
    
    # Parse subset if provided
    subset = None
    if args.subset:
        try:
            subset = [int(x.strip()) for x in args.subset.split(',')]
            logger.info(f"Using domain subset: {subset}")
        except ValueError:
            logger.error("Invalid subset format. Use comma-separated integers (e.g., '0,1,2')")
            return 1
    
    # Create domain pairs
    try:
        domain_pairs, domains = create_domain_pairs(args.data_dir, subset)
        logger.info(f"Created {len(domain_pairs)} domain pairs from {len(domains)} domains")
        
        # Apply max pairs limit if specified
        if args.max_pairs and args.max_pairs < len(domain_pairs):
            domain_pairs = domain_pairs[:args.max_pairs]
            logger.info(f"Limited to first {args.max_pairs} domain pairs")
        
        # Apply resume offset
        if args.resume_from > 0:
            if args.resume_from >= len(domain_pairs):
                logger.error(f"Resume index {args.resume_from} >= total pairs {len(domain_pairs)}")
                return 1
            domain_pairs = domain_pairs[args.resume_from:]
            logger.info(f"Resuming from pair {args.resume_from}, {len(domain_pairs)} pairs remaining")
        
    except Exception as e:
        logger.error(f"Error creating domain pairs: {e}")
        return 1
    
    # Dry run - just print pairs
    if args.dry_run:
        logger.info("DRY RUN - Domain pairs to be processed:")
        for i, (source, target) in enumerate(domain_pairs):
            source_name = os.path.basename(source).replace('ratings_', '').replace('.csv', '')
            target_name = os.path.basename(target).replace('ratings_', '').replace('.csv', '')
            logger.info(f"{i+1:3d}: {source_name} -> {target_name}")
        return 0
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Run experiments
    try:
        if args.grid_search:
            logger.info("Running grid search experiments")
            param_grid = create_param_grid_from_args(args)
            logger.info(f"Parameter grid: {param_grid}")
            
            results = run_multi_domain_experiments(
                domain_pairs=domain_pairs,
                param_grid=param_grid,
                output_dir=args.output_dir,
                log_dir=args.log_dir
            )
        else:
            logger.info("Running single configuration experiments")
            config = create_config_from_args(args)
            logger.info(f"Configuration: {config}")
            
            results = run_multi_domain_experiments(
                domain_pairs=domain_pairs,
                config=config,
                output_dir=args.output_dir,
                log_dir=args.log_dir
            )
        
        logger.info("All experiments completed successfully!")
        logger.info(f"Results saved to: {results['summary_path']}")
        
    except KeyboardInterrupt:
        logger.info("Experiments interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)