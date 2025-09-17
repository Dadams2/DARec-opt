#!/usr/bin/env python3
"""
Example script showing how to use the DARec experiment framework programmatically.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiment_framework import (
    ExperimentConfig, ExperimentRunner, MethodConfig, DataConfig, EvaluationConfig,
    get_i_darec_config, get_u_darec_config
)
from experiment_framework.utils import DatasetDiscovery


def example_quick_test():
    """Example: Quick test with minimal configuration."""
    print("Example 1: Quick Test")
    print("=" * 30)
    
    # Create minimal configuration
    config = ExperimentConfig(
        name="programmatic_quick_test",
        description="Quick test created programmatically",
        data=DataConfig(
            max_domain_pairs=2,
            source_domains=["Amazon_Instant_Video", "Apps_for_Android"],
            target_domains=["Amazon_Instant_Video", "Apps_for_Android", "Automotive"]
        ),
        methods=[
            MethodConfig(
                name="I_DARec",
                module_path="DArec_opt.I_DArec.Train_DArec",
                class_name="I_DArec",
                hyperparameters={
                    "epochs": 5,
                    "batch_size": 32,
                    "lr": 1e-3,
                    "wd": 1e-4,
                    "n_factors": 200,
                    "RPE_hidden_size": 200
                },
                grid_search_params={
                    "lr": [1e-3, 1e-2]  # Just 2 values
                }
            )
        ],
        output_dir="programmatic_test_results"
    )
    
    # Run experiment
    runner = ExperimentRunner(config)
    runner.print_experiment_summary()
    
    # For demo, just validate without running
    is_valid = runner.validate_configuration()
    print(f"Configuration valid: {is_valid}")
    
    # Uncomment to actually run:
    # results = runner.run()
    # print(f"Completed {len(results)} experiments")


def example_custom_method():
    """Example: Adding a custom method configuration."""
    print("\nExample 2: Custom Method Configuration")
    print("=" * 40)
    
    # Create custom AutoRec configuration with specific hyperparameters
    custom_autorec = MethodConfig(
        name="Custom_I_AutoRec",
        module_path="DArec_opt.I_DArec.Train_AutoRec", 
        class_name="I_AutoRec",
        hyperparameters={
            "epochs": 30,
            "batch_size": 128,  # Larger batch size
            "lr": 5e-4,         # Lower learning rate
            "wd": 1e-5,         # Lower weight decay
            "n_factors": 300,   # More factors
            "train_S": False
        },
        grid_search_params={
            "n_factors": [200, 300, 400],
            "lr": [1e-4, 5e-4, 1e-3]
        }
    )
    
    config = ExperimentConfig(
        name="custom_method_test",
        description="Testing custom method configuration",
        methods=[custom_autorec],
        data=DataConfig(max_domain_pairs=2),
        output_dir="custom_method_results"
    )
    
    runner = ExperimentRunner(config)
    runner.print_experiment_summary()


def example_domain_analysis():
    """Example: Analyze available domains."""
    print("\nExample 3: Domain Analysis")
    print("=" * 30)
    
    # Discover domains
    discovery = DatasetDiscovery("/home2/dadams/DARec-opt/data")
    domains = discovery.get_available_domains()
    
    print(f"Found {len(domains)} domains:")
    for domain in domains[:10]:  # Show first 10
        print(f"  - {domain}")
    
    if len(domains) > 10:
        print(f"  ... and {len(domains) - 10} more")
    
    # Get domain pair statistics
    domain_pairs = discovery.get_domain_pairs(max_pairs=20)
    print(f"\nExample domain pairs (first 5):")
    for source, target in domain_pairs[:5]:
        print(f"  {source} → {target}")
    
    # Check data statistics
    print("\nGetting data statistics...")
    stats = discovery.get_data_stats()
    
    # Show stats for first 3 domains
    for domain, stat in list(stats.items())[:3]:
        if 'error' not in stat:
            print(f"\n{domain}:")
            print(f"  Users: {stat['num_users']:,}")
            print(f"  Items: {stat['num_items']:,}")
            print(f"  Ratings: {stat['num_ratings']:,}")
            print(f"  File size: {stat['file_size_mb']:.1f} MB")


def example_evaluation_framework():
    """Example: Using the evaluation framework."""
    print("\nExample 4: Evaluation Framework")
    print("=" * 35)
    
    from experiment_framework.evaluation import EvaluationFramework
    import torch
    
    # Create mock data for demonstration
    predictions = torch.randn(100, 50) * 5 + 3  # 100 users, 50 items
    targets = torch.randn(100, 50) * 5 + 3
    mask = torch.randint(0, 2, (100, 50))  # Random mask
    
    # Create evaluation framework
    evaluator = EvaluationFramework()
    
    # Add additional evaluators
    evaluator.add_ranking_evaluator(k=10)
    evaluator.add_coverage_evaluator(total_items=50)
    
    # Evaluate
    metrics = evaluator.evaluate(predictions, targets, mask)
    
    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def example_result_analysis():
    """Example: Analyzing results from previous runs."""
    print("\nExample 5: Result Analysis")
    print("=" * 30)
    
    from experiment_framework.utils import ResultManager
    
    # This would work if you have previous results
    output_dir = "experiment_results"  # Change to actual results directory
    
    if os.path.exists(output_dir):
        result_manager = ResultManager(output_dir)
        
        # Load results
        results = result_manager.load_experiment_results()
        print(f"Found {len(results)} previous experiment results")
        
        if results:
            # Create summary
            summary_df = result_manager.create_results_summary()
            print(f"Summary shape: {summary_df.shape}")
            
            # Generate plots and report
            result_manager.create_comparison_plots()
            result_manager.generate_report()
            print("Analysis plots and report generated")
        else:
            print("No results found to analyze")
    else:
        print(f"Results directory {output_dir} not found")
        print("Run some experiments first to generate results")


def main():
    """Run all examples."""
    print("DARec Experiment Framework Examples")
    print("=" * 50)
    
    try:
        example_quick_test()
        example_custom_method()
        example_domain_analysis()
        example_evaluation_framework()
        example_result_analysis()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nTo run actual experiments:")
        print("1. Use the CLI: python run_experiments.py --preset quick_test")
        print("2. Modify the example configurations in examples/")
        print("3. Create your own configuration files")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
