#!/usr/bin/env python3
"""
Command-line interface for DARec experiments.
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from experiment_framework import (
    ExperimentConfig, ExperimentRunner, 
    get_quick_test_config, get_full_grid_search_config,
    get_i_darec_config, get_u_darec_config, get_autorec_config
)
from experiment_framework.utils import DatasetDiscovery


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="DARec Experiment Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  python run_experiments.py --preset quick_test
  
  # Run full grid search
  python run_experiments.py --preset full_grid_search
  
  # Run custom experiment from config file
  python run_experiments.py --config my_experiment.yaml
  
  # List available domains
  python run_experiments.py --list-domains
  
  # Create custom experiment interactively
  python run_experiments.py --interactive
        """
    )
    
    # Main commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Path to experiment configuration file (YAML/JSON)')
    group.add_argument('--preset', choices=['quick_test', 'full_grid_search'], 
                      help='Use a preset configuration')
    group.add_argument('--interactive', action='store_true', 
                      help='Create experiment configuration interactively')
    group.add_argument('--list-domains', action='store_true', 
                      help='List available domains and exit')
    group.add_argument('--analyze', type=str, 
                      help='Analyze results from output directory')
    
    # Experiment options
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                       help='Output directory for results (default: experiment_results)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous incomplete run')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                       help='Device to use for training (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Data options
    parser.add_argument('--data-dir', type=str, default='/home2/dadams/DARec-opt/data',
                       help='Path to data directory')
    parser.add_argument('--max-domain-pairs', type=int,
                       help='Limit number of domain pairs for testing')
    parser.add_argument('--source-domains', nargs='+',
                       help='Specific source domains to test')
    parser.add_argument('--target-domains', nargs='+',
                       help='Specific target domains to test')
    
    # Method options
    parser.add_argument('--methods', nargs='+', 
                       choices=['I_DARec', 'U_DARec', 'I_AutoRec', 'U_AutoRec'],
                       help='Specific methods to run')
    
    # Evaluation options
    parser.add_argument('--metrics', nargs='+', 
                       choices=['rmse', 'mae', 'mape', 'ranking', 'statistical'],
                       default=['rmse', 'mae'],
                       help='Evaluation metrics to use')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save training curve plots')
    
    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output (only errors)')
    
    return parser


def list_domains(data_dir: str):
    """List available domains."""
    discovery = DatasetDiscovery(data_dir)
    discovery.print_summary()


def interactive_config_creation(args) -> ExperimentConfig:
    """Create experiment configuration interactively."""
    print("DARec Experiment Configuration Creator")
    print("=" * 40)
    
    # Basic info
    name = input("Experiment name: ")
    description = input("Description (optional): ")
    
    # Select preset or custom
    print("\nSelect starting point:")
    print("1. Quick test (fast, limited scope)")
    print("2. Full grid search (comprehensive)")
    print("3. Custom configuration")
    
    choice = input("Choice (1-3): ")
    
    if choice == "1":
        config = get_quick_test_config()
    elif choice == "2":
        config = get_full_grid_search_config()
    else:
        config = ExperimentConfig(name=name, description=description)
    
    config.name = name
    config.description = description
    
    # Update with command line args
    if args.output_dir != 'experiment_results':
        config.output_dir = args.output_dir
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.max_domain_pairs:
        config.data.max_domain_pairs = args.max_domain_pairs
    if args.source_domains:
        config.data.source_domains = args.source_domains
    if args.target_domains:
        config.data.target_domains = args.target_domains
    
    # Method selection
    if args.methods:
        method_configs = []
        for method_name in args.methods:
            if method_name == "I_DARec":
                method_configs.append(get_i_darec_config())
            elif method_name == "U_DARec":
                method_configs.append(get_u_darec_config())
            elif method_name == "I_AutoRec":
                method_configs.append(get_autorec_config("I"))
            elif method_name == "U_AutoRec":
                method_configs.append(get_autorec_config("U"))
        config.methods = method_configs
    
    # Save configuration option
    save_config = input("\nSave this configuration to file? (y/n): ")
    if save_config.lower() == 'y':
        config_path = input("Configuration file path (e.g., my_experiment.yaml): ")
        config.to_yaml(config_path)
        print(f"Configuration saved to {config_path}")
    
    return config


def analyze_results(output_dir: str):
    """Analyze results from previous runs."""
    from experiment_framework.utils import ResultManager
    
    result_manager = ResultManager(output_dir)
    
    print(f"Analyzing results from: {output_dir}")
    print("=" * 50)
    
    # Load and summarize results
    results_df = result_manager.create_results_summary()
    
    if results_df.empty:
        print("No results found in the specified directory.")
        return
    
    print(f"Total experiments: {len(results_df)}")
    
    if 'status' in results_df.columns:
        status_counts = results_df['status'].value_counts()
        print(f"Completed: {status_counts.get('completed', 0)}")
        print(f"Failed: {status_counts.get('failed', 0)}")
    
    if 'method' in results_df.columns:
        print(f"\nMethods tested: {results_df['method'].unique().tolist()}")
    
    if 'metrics_rmse' in results_df.columns:
        print(f"\nRMSE Statistics:")
        print(f"  Mean: {results_df['metrics_rmse'].mean():.4f}")
        print(f"  Std:  {results_df['metrics_rmse'].std():.4f}")
        print(f"  Min:  {results_df['metrics_rmse'].min():.4f}")
        print(f"  Max:  {results_df['metrics_rmse'].max():.4f}")
    
    # Generate plots and report
    print("\nGenerating analysis plots and report...")
    result_manager.create_comparison_plots()
    result_manager.generate_report()
    print(f"Analysis complete. Check {output_dir} for plots and report.")


def run_experiment_from_config(config_path: str, args) -> ExperimentConfig:
    """Load and modify configuration from file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        config = ExperimentConfig.from_yaml(config_path)
    elif config_path.suffix.lower() == '.json':
        config = ExperimentConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    # Override with command line arguments
    if args.output_dir != 'experiment_results':
        config.output_dir = args.output_dir
    if args.parallel:
        config.parallel = args.parallel
    if args.max_workers != 4:
        config.max_workers = args.max_workers
    if args.resume:
        config.resume = args.resume
    if args.device != 'cuda':
        config.device = args.device
    if args.seed != 42:
        config.random_seed = args.seed
    
    # Data overrides
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.max_domain_pairs:
        config.data.max_domain_pairs = args.max_domain_pairs
    if args.source_domains:
        config.data.source_domains = args.source_domains
    if args.target_domains:
        config.data.target_domains = args.target_domains
    
    # Evaluation overrides
    if args.metrics != ['rmse', 'mae']:
        config.evaluation.metrics = args.metrics
    if args.save_models:
        config.evaluation.save_predictions = args.save_models
    if not args.save_plots:
        config.evaluation.plot_training_curves = args.save_plots
    
    return config


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_domains:
        list_domains(args.data_dir)
        return
    
    if args.analyze:
        analyze_results(args.analyze)
        return
    
    # Create configuration
    try:
        if args.preset:
            if args.preset == 'quick_test':
                config = get_quick_test_config()
            elif args.preset == 'full_grid_search':
                config = get_full_grid_search_config()
            else:
                raise ValueError(f"Unknown preset: {args.preset}")
            
            # Apply command line overrides
            config.output_dir = args.output_dir
            config.parallel = args.parallel
            config.max_workers = args.max_workers
            config.resume = args.resume
            config.device = args.device
            config.random_seed = args.seed
            
        elif args.config:
            config = run_experiment_from_config(args.config, args)
            
        elif args.interactive:
            config = interactive_config_creation(args)
            
        else:
            parser.print_help()
            return
        
        # Run experiments
        runner = ExperimentRunner(config)
        
        print(f"Starting experiments: {config.name}")
        print("-" * 50)
        
        results = runner.run()
        
        # Print summary
        completed = len([r for r in results if r.get('status') == 'completed'])
        failed = len([r for r in results if r.get('status') == 'failed'])
        
        print(f"\nExperiment run completed!")
        print(f"Results: {completed} successful, {failed} failed")
        print(f"Output directory: {config.output_dir}")
        
        if completed > 0:
            # Find best result
            completed_results = [r for r in results if r.get('status') == 'completed']
            if any('metrics' in r and 'rmse' in r['metrics'] for r in completed_results):
                best_result = min(
                    (r for r in completed_results if 'metrics' in r and 'rmse' in r['metrics']),
                    key=lambda x: x['metrics']['rmse']
                )
                print(f"\nBest result:")
                print(f"  Method: {best_result.get('method', 'Unknown')}")
                print(f"  Domain: {best_result.get('source_domain', '?')} → {best_result.get('target_domain', '?')}")
                print(f"  RMSE: {best_result['metrics']['rmse']:.4f}")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
