"""
Main experiment runner for DARec experiments.
"""
import os
import sys
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .config import ExperimentConfig, MethodConfig
from .methods import method_registry, BaseMethod
from .evaluation import EvaluationFramework
from .utils import DatasetDiscovery, ResultManager, ExperimentTracker


class ExperimentRunner:
    """Main runner for DARec experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result_manager = ResultManager(config.output_dir)
        self.evaluation_framework = EvaluationFramework()
        self.tracker = ExperimentTracker(config.output_dir)
        
        # Set random seeds for reproducibility
        self.set_random_seeds(config.random_seed)
        
        # Setup device
        self.device = self._setup_device(config.device)
        
        # Discover datasets
        self.dataset_discovery = DatasetDiscovery(config.data.data_dir)
        
        self.result_manager.logger.info(f"Initialized experiment runner: {config.name}")
        self.result_manager.logger.info(f"Output directory: {config.output_dir}")
        self.result_manager.logger.info(f"Device: {self.device}")
    
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_device(self, device_str: str) -> str:
        """Setup and validate device."""
        if device_str == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device_str == "cuda" and not torch.cuda.is_available():
            self.result_manager.logger.warning("CUDA requested but not available, using CPU")
            return "cpu"
        else:
            return device_str
    
    def _generate_experiment_id(self, method_name: str, source_domain: str, target_domain: str, 
                              hyperparams: Dict[str, Any]) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_hash = hash(str(sorted(hyperparams.items()))) % 10000
        return f"{method_name}_{source_domain}_{target_domain}_{param_hash}_{timestamp}"
    
    def run_single_experiment(self, method_config: MethodConfig, source_domain: str, 
                            target_domain: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given configuration."""
        experiment_id = self._generate_experiment_id(
            method_config.name, source_domain, target_domain, hyperparams
        )
        
        self.result_manager.logger.info(f"Starting experiment: {experiment_id}")
        
        experiment_result = {
            "experiment_id": experiment_id,
            "method": method_config.name,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "hyperparameters": hyperparams,
            "config": {
                "data": self.config.data.__dict__,
                "evaluation": self.config.evaluation.__dict__
            },
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            # Get method instance
            method_hyperparams = hyperparams.copy()
            method_hyperparams.update({"device": self.device})
            
            method = method_registry.get_method(
                method_config.class_name, method_hyperparams, source_domain, target_domain
            )
            
            # Prepare data
            self.result_manager.logger.info(f"Preparing data for {source_domain} -> {target_domain}")
            train_loader, test_loader = method.prepare_data(self.config.data.__dict__)
            
            # Build model
            self.result_manager.logger.info("Building model")
            model = method.build_model()
            
            # Train model
            self.result_manager.logger.info("Training model")
            training_history = method.train(train_loader, test_loader)
            
            # Evaluate model
            self.result_manager.logger.info("Evaluating model")
            evaluation_metrics = method.evaluate(test_loader)
            
            # Additional evaluation with framework
            if hasattr(self.evaluation_framework, 'evaluate_method'):
                framework_metrics = self.evaluation_framework.evaluate_method(
                    method, test_loader, self.config.evaluation.metrics
                )
                evaluation_metrics.update(framework_metrics)
            
            # Update result
            experiment_result.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "training_history": training_history,
                "metrics": evaluation_metrics,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            })
            
            # Save results
            self.result_manager.save_experiment_result(experiment_result, experiment_id)
            self.result_manager.save_training_history(training_history, experiment_id)
            
            if self.config.evaluation.save_predictions:
                # Save model if requested
                self.result_manager.save_model(model, experiment_id)
            
            if self.config.evaluation.plot_training_curves:
                # Plot training curves
                self.result_manager.plot_training_curves(experiment_id)
            
            self.tracker.mark_completed(experiment_id)
            self.result_manager.logger.info(f"Completed experiment: {experiment_id}")
            
            return experiment_result
            
        except Exception as e:
            error_msg = f"Experiment {experiment_id} failed: {str(e)}"
            self.result_manager.logger.error(error_msg)
            self.result_manager.logger.error(traceback.format_exc())
            
            experiment_result.update({
                "status": "failed",
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            self.result_manager.save_experiment_result(experiment_result, experiment_id)
            self.tracker.mark_failed(experiment_id, str(e))
            
            return experiment_result
    
    def run_grid_search(self, method_config: MethodConfig, domain_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Run grid search for a method across domain pairs."""
        results = []
        param_combinations = method_config.get_param_combinations()
        
        self.result_manager.logger.info(
            f"Starting grid search for {method_config.name}: "
            f"{len(param_combinations)} param combinations × {len(domain_pairs)} domain pairs = "
            f"{len(param_combinations) * len(domain_pairs)} experiments"
        )
        
        for source_domain, target_domain in domain_pairs:
            for hyperparams in param_combinations:
                result = self.run_single_experiment(
                    method_config, source_domain, target_domain, hyperparams
                )
                results.append(result)
                
                # Print progress
                completed = len([r for r in results if r['status'] == 'completed'])
                failed = len([r for r in results if r['status'] == 'failed'])
                total = len(results)
                
                self.result_manager.logger.info(
                    f"Progress: {total}/{len(param_combinations) * len(domain_pairs)} "
                    f"({completed} completed, {failed} failed)"
                )
        
        return results
    
    def run_parallel_experiments(self, experiments: List[Tuple], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run experiments in parallel."""
        if max_workers is None:
            max_workers = min(self.config.max_workers, mp.cpu_count())
        
        self.result_manager.logger.info(f"Running {len(experiments)} experiments with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_experiment = {
                executor.submit(
                    self.run_single_experiment,
                    method_config, source_domain, target_domain, hyperparams
                ): (method_config, source_domain, target_domain, hyperparams)
                for method_config, source_domain, target_domain, hyperparams in experiments
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_experiment):
                method_config, source_domain, target_domain, hyperparams = future_to_experiment[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed = len([r for r in results if r['status'] == 'completed'])
                    failed = len([r for r in results if r['status'] == 'failed'])
                    
                    self.result_manager.logger.info(
                        f"Progress: {len(results)}/{len(experiments)} "
                        f"({completed} completed, {failed} failed)"
                    )
                    
                except Exception as e:
                    self.result_manager.logger.error(
                        f"Failed to get result for {method_config.name} "
                        f"{source_domain}->{target_domain}: {e}"
                    )
        
        return results
    
    def run_all_experiments(self) -> List[Dict[str, Any]]:
        """Run all experiments defined in the configuration."""
        # Get domain pairs
        domain_pairs = self.config.get_domain_pairs()
        total_experiments = self.config.get_total_experiments()
        
        self.result_manager.logger.info(f"Starting experiment run: {self.config.name}")
        self.result_manager.logger.info(f"Total experiments: {total_experiments}")
        self.result_manager.logger.info(f"Domain pairs: {len(domain_pairs)}")
        self.result_manager.logger.info(f"Methods: {[m.name for m in self.config.methods]}")
        
        # Save experiment configuration
        self.result_manager.save_experiment_config(self.config, self.config.name)
        
        # Start tracking
        self.tracker.start_experiment(self.config.name, total_experiments)
        
        all_results = []
        
        if self.config.parallel and len(self.config.methods) > 1:
            # Prepare all experiments for parallel execution
            experiments = []
            for method_config in self.config.methods:
                param_combinations = method_config.get_param_combinations()
                for source_domain, target_domain in domain_pairs:
                    for hyperparams in param_combinations:
                        experiments.append((method_config, source_domain, target_domain, hyperparams))
            
            # Run in parallel
            all_results = self.run_parallel_experiments(experiments)
            
        else:
            # Run sequentially
            for method_config in self.config.methods:
                method_results = self.run_grid_search(method_config, domain_pairs)
                all_results.extend(method_results)
        
        # Generate summary and plots
        self.result_manager.logger.info("Generating analysis and plots")
        self.result_manager.create_comparison_plots()
        self.result_manager.generate_report()
        
        completed = len([r for r in all_results if r['status'] == 'completed'])
        failed = len([r for r in all_results if r['status'] == 'failed'])
        
        self.result_manager.logger.info(f"Experiment run completed: {completed} successful, {failed} failed")
        
        return all_results
    
    def resume_experiments(self) -> List[Dict[str, Any]]:
        """Resume experiments from previous run."""
        if not self.config.resume:
            self.result_manager.logger.warning("Resume not enabled in config")
            return []
        
        # Load previous results
        previous_results = self.result_manager.load_experiment_results()
        completed_experiments = set()
        
        for result in previous_results:
            if result.get('status') == 'completed':
                exp_key = (
                    result.get('method'),
                    result.get('source_domain'),
                    result.get('target_domain'),
                    str(sorted(result.get('hyperparameters', {}).items()))
                )
                completed_experiments.add(exp_key)
        
        self.result_manager.logger.info(f"Found {len(completed_experiments)} completed experiments")
        
        # Generate remaining experiments
        domain_pairs = self.config.get_domain_pairs()
        remaining_experiments = []
        
        for method_config in self.config.methods:
            param_combinations = method_config.get_param_combinations()
            for source_domain, target_domain in domain_pairs:
                for hyperparams in param_combinations:
                    exp_key = (
                        method_config.name,
                        source_domain,
                        target_domain,
                        str(sorted(hyperparams.items()))
                    )
                    
                    if exp_key not in completed_experiments:
                        remaining_experiments.append((method_config, source_domain, target_domain, hyperparams))
        
        self.result_manager.logger.info(f"Running {len(remaining_experiments)} remaining experiments")
        
        if remaining_experiments:
            if self.config.parallel:
                new_results = self.run_parallel_experiments(remaining_experiments)
            else:
                new_results = []
                for method_config, source_domain, target_domain, hyperparams in remaining_experiments:
                    result = self.run_single_experiment(method_config, source_domain, target_domain, hyperparams)
                    new_results.append(result)
            
            # Combine with previous results
            all_results = previous_results + new_results
        else:
            all_results = previous_results
        
        return all_results
    
    def validate_configuration(self) -> bool:
        """Validate experiment configuration."""
        errors = []
        
        # Check data directory
        if not os.path.exists(self.config.data.data_dir):
            errors.append(f"Data directory does not exist: {self.config.data.data_dir}")
        
        # Check domain availability
        available_domains = self.dataset_discovery.get_available_domains()
        for domain in self.config.data.source_domains + self.config.data.target_domains:
            if domain not in available_domains:
                errors.append(f"Domain not found: {domain}")
        
        # Check methods
        available_methods = method_registry.list_methods()
        for method_config in self.config.methods:
            if method_config.class_name not in available_methods:
                errors.append(f"Method not available: {method_config.name}")
            
            # Check pretrained weights if required
            if method_config.pretrained_required:
                for weight_name, weight_path in method_config.pretrained_paths.items():
                    if not os.path.exists(weight_path):
                        errors.append(f"Pretrained weights not found: {weight_path}")
        
        # Check output directory permissions
        try:
            test_file = Path(self.config.output_dir) / "test_write.tmp"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Cannot write to output directory: {e}")
        
        if errors:
            for error in errors:
                self.result_manager.logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def print_experiment_summary(self):
        """Print a summary of the planned experiments."""
        domain_pairs = self.config.get_domain_pairs()
        total_experiments = self.config.get_total_experiments()
        
        print(f"Experiment Summary: {self.config.name}")
        print("=" * 50)
        print(f"Description: {self.config.description}")
        print(f"Output directory: {self.config.output_dir}")
        print(f"Device: {self.device}")
        print(f"Parallel execution: {self.config.parallel}")
        print()
        
        print(f"Data Configuration:")
        print(f"  Data directory: {self.config.data.data_dir}")
        print(f"  Batch size: {self.config.data.batch_size}")
        print(f"  Train ratio: {self.config.data.train_ratio}")
        print(f"  Test ratio: {self.config.data.test_ratio}")
        print()
        
        print(f"Domain Pairs ({len(domain_pairs)}):")
        for i, (source, target) in enumerate(domain_pairs[:10]):  # Show first 10
            print(f"  {source} → {target}")
        if len(domain_pairs) > 10:
            print(f"  ... and {len(domain_pairs) - 10} more")
        print()
        
        print(f"Methods ({len(self.config.methods)}):")
        for method_config in self.config.methods:
            param_combinations = len(method_config.get_param_combinations())
            print(f"  {method_config.name}: {param_combinations} parameter combinations")
            if method_config.grid_search_params:
                print(f"    Grid search parameters: {list(method_config.grid_search_params.keys())}")
        print()
        
        print(f"Total experiments: {total_experiments}")
        
        # Estimate time
        avg_time_per_experiment = 300  # 5 minutes estimate
        total_time_hours = (total_experiments * avg_time_per_experiment) / 3600
        print(f"Estimated time: {total_time_hours:.1f} hours")
        print()
    
    def run(self) -> List[Dict[str, Any]]:
        """Main entry point to run experiments."""
        if not self.validate_configuration():
            raise ValueError("Configuration validation failed")
        
        self.print_experiment_summary()
        
        if self.config.resume:
            return self.resume_experiments()
        else:
            return self.run_all_experiments()
