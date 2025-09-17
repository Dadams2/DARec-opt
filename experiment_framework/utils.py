"""
Utility functions for the DARec experiment framework.
"""
import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetDiscovery:
    """Discover and manage datasets in the data directory."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._domains = None
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains from CSV files."""
        if self._domains is None:
            self._discover_domains()
        return self._domains
    
    def _discover_domains(self):
        """Discover domains from the data directory."""
        csv_files = list(self.data_dir.glob("ratings_*.csv"))
        domains = []
        
        for file in csv_files:
            # Extract domain name from filename
            match = re.match(r"ratings_(.+)\.csv", file.name)
            if match:
                domain = match.group(1)
                domains.append(domain)
        
        self._domains = sorted(domains)
    
    def get_domain_pairs(self, max_pairs: Optional[int] = None) -> List[Tuple[str, str]]:
        """Get all possible domain pairs."""
        domains = self.get_available_domains()
        pairs = []
        
        for source in domains:
            for target in domains:
                if source != target:
                    pairs.append((source, target))
        
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        return pairs
    
    def check_preprocessed_data(self, source_domain: str, target_domain: str, variant: str = "I") -> bool:
        """Check if preprocessed data exists for a domain pair."""
        if variant == "I":
            pattern = f"I_ratings_{source_domain}.csv_ratings_{target_domain}.csv.npy"
        else:
            pattern = f"U_ratings_{source_domain}.csv_ratings_{target_domain}.csv.npy"
        
        return (self.data_dir / pattern).exists()
    
    def get_data_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about available datasets."""
        stats = {}
        domains = self.get_available_domains()
        
        for domain in domains:
            csv_path = self.data_dir / f"ratings_{domain}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, header=None)
                    stats[domain] = {
                        'num_ratings': len(df),
                        'num_users': df.iloc[:, 0].nunique(),
                        'num_items': df.iloc[:, 1].nunique(),
                        'rating_range': (df.iloc[:, 2].min(), df.iloc[:, 2].max()),
                        'file_size_mb': csv_path.stat().st_size / (1024 * 1024)
                    }
                except Exception as e:
                    stats[domain] = {'error': str(e)}
        
        return stats
    
    def print_summary(self):
        """Print a summary of available datasets."""
        domains = self.get_available_domains()
        stats = self.get_data_stats()
        
        print(f"Found {len(domains)} domains:")
        print("-" * 60)
        
        for domain in domains:
            if domain in stats and 'error' not in stats[domain]:
                s = stats[domain]
                print(f"{domain:30} | Users: {s['num_users']:6} | Items: {s['num_items']:6} | Ratings: {s['num_ratings']:8}")
            else:
                print(f"{domain:30} | Error loading data")
        
        print("-" * 60)
        print(f"Total possible domain pairs: {len(domains) * (len(domains) - 1)}")


class ResultManager:
    """Manage experiment results including saving, loading, and analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for experiments."""
        log_file = self.output_dir / "logs" / "experiment.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def save_experiment_config(self, config, experiment_id: str):
        """Save experiment configuration."""
        config_file = self.output_dir / "results" / f"{experiment_id}_config.json"
        
        # Convert config to dict if needed
        if hasattr(config, '__dict__'):
            config_dict = self._config_to_dict(config)
        else:
            config_dict = config
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._config_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [self._config_to_dict(item) if hasattr(item, '__dict__') else item for item in value]
                else:
                    result[key] = value
            return result
        return config
    
    def save_experiment_result(self, result: Dict[str, Any], experiment_id: str):
        """Save individual experiment result."""
        result_file = self.output_dir / "results" / f"{experiment_id}_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def save_model(self, model, experiment_id: str):
        """Save trained model."""
        model_file = self.output_dir / "models" / f"{experiment_id}_model.pth"
        torch.save(model.state_dict(), model_file)
    
    def save_training_history(self, history: Dict[str, List[float]], experiment_id: str):
        """Save training history."""
        history_file = self.output_dir / "results" / f"{experiment_id}_history.pkl"
        
        with open(history_file, 'wb') as f:
            pickle.dump(history, f)
    
    def load_experiment_results(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """Load all experiment results matching pattern."""
        results = []
        result_files = (self.output_dir / "results").glob(pattern)
        
        for file in result_files:
            if file.name.endswith("_result.json"):
                try:
                    with open(file, 'r') as f:
                        result = json.load(f)
                        result['experiment_id'] = file.stem.replace('_result', '')
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Could not load result file {file}: {e}")
        
        return results
    
    def create_results_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of all results."""
        results = self.load_experiment_results()
        
        if not results:
            return pd.DataFrame()
        
        # Flatten results for DataFrame
        flattened = []
        for result in results:
            flat_result = self._flatten_dict(result)
            flattened.append(flat_result)
        
        return pd.DataFrame(flattened)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def plot_training_curves(self, experiment_id: str, save: bool = True):
        """Plot training curves for an experiment."""
        history_file = self.output_dir / "results" / f"{experiment_id}_history.pkl"
        
        if not history_file.exists():
            self.logger.warning(f"No training history found for {experiment_id}")
            return
        
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        
        plt.figure(figsize=(10, 6))
        
        if 'train_rmse' in history:
            plt.plot(history['train_rmse'], label='Train RMSE', marker='o')
        if 'test_rmse' in history:
            plt.plot(history['test_rmse'], label='Test RMSE', marker='s')
        
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title(f'Training Curves - {experiment_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plot_file = self.output_dir / "plots" / f"{experiment_id}_training_curves.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_comparison_plots(self):
        """Create comparison plots across all experiments."""
        results_df = self.create_results_summary()
        
        if results_df.empty:
            self.logger.warning("No results found for comparison plots")
            return
        
        # Plot 1: Method performance comparison
        if 'method' in results_df.columns and 'metrics_rmse' in results_df.columns:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=results_df, x='method', y='metrics_rmse')
            plt.title('RMSE Comparison Across Methods')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "method_comparison.png", dpi=300)
            plt.close()
        
        # Plot 2: Domain pair difficulty
        if 'source_domain' in results_df.columns and 'target_domain' in results_df.columns:
            results_df['domain_pair'] = results_df['source_domain'] + ' → ' + results_df['target_domain']
            
            # Create heatmap of performance by domain pairs
            if 'metrics_rmse' in results_df.columns:
                pivot_data = results_df.groupby(['source_domain', 'target_domain'])['metrics_rmse'].mean().unstack()
                
                plt.figure(figsize=(14, 10))
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis_r')
                plt.title('Average RMSE by Domain Pair')
                plt.xlabel('Target Domain')
                plt.ylabel('Source Domain')
                plt.tight_layout()
                plt.savefig(self.output_dir / "plots" / "domain_pair_heatmap.png", dpi=300)
                plt.close()
        
        # Plot 3: Hyperparameter effects
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols if col.startswith('hyperparameters_')]
        
        if param_cols and 'metrics_rmse' in results_df.columns:
            n_params = len(param_cols)
            fig, axes = plt.subplots(1, min(n_params, 3), figsize=(15, 5))
            if n_params == 1:
                axes = [axes]
            
            for i, param_col in enumerate(param_cols[:3]):
                sns.scatterplot(data=results_df, x=param_col, y='metrics_rmse', ax=axes[i])
                axes[i].set_title(f'RMSE vs {param_col.replace("hyperparameters_", "")}')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "hyperparameter_effects.png", dpi=300)
            plt.close()
    
    def generate_report(self):
        """Generate a comprehensive experiment report."""
        results_df = self.create_results_summary()
        
        if results_df.empty:
            self.logger.warning("No results found for report generation")
            return
        
        report_file = self.output_dir / "experiment_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("DARec Experiment Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments: {len(results_df)}\n\n")
            
            # Method summary
            if 'method' in results_df.columns:
                f.write("Methods tested:\n")
                method_counts = results_df['method'].value_counts()
                for method, count in method_counts.items():
                    f.write(f"  {method}: {count} experiments\n")
                f.write("\n")
            
            # Performance summary
            if 'metrics_rmse' in results_df.columns:
                f.write("Performance Summary (RMSE):\n")
                f.write(f"  Overall mean: {results_df['metrics_rmse'].mean():.4f}\n")
                f.write(f"  Overall std:  {results_df['metrics_rmse'].std():.4f}\n")
                f.write(f"  Best result:  {results_df['metrics_rmse'].min():.4f}\n")
                f.write(f"  Worst result: {results_df['metrics_rmse'].max():.4f}\n\n")
                
                # Method-wise performance
                if 'method' in results_df.columns:
                    f.write("Performance by Method:\n")
                    method_perf = results_df.groupby('method')['metrics_rmse'].agg(['mean', 'std', 'min', 'max'])
                    f.write(method_perf.to_string())
                    f.write("\n\n")
            
            # Domain pair summary
            if 'source_domain' in results_df.columns and 'target_domain' in results_df.columns:
                f.write("Domain Pairs Tested:\n")
                domain_pairs = results_df.groupby(['source_domain', 'target_domain']).size()
                f.write(f"  Total unique pairs: {len(domain_pairs)}\n")
                f.write("  Top 10 most tested pairs:\n")
                for (source, target), count in domain_pairs.nlargest(10).items():
                    f.write(f"    {source} → {target}: {count} experiments\n")
                f.write("\n")
        
        self.logger.info(f"Report generated: {report_file}")
    
    def cleanup_old_results(self, days: int = 30):
        """Clean up old result files."""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        
        for result_dir in [self.output_dir / "results", self.output_dir / "models", self.output_dir / "plots"]:
            for file in result_dir.glob("*"):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    self.logger.info(f"Cleaned up old file: {file}")


class ExperimentTracker:
    """Track running experiments and manage progress."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / "experiment_progress.json"
        self.load_progress()
    
    def load_progress(self):
        """Load experiment progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'completed': [],
                'failed': [],
                'running': [],
                'total': 0,
                'start_time': None
            }
    
    def save_progress(self):
        """Save experiment progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2, default=str)
    
    def start_experiment(self, experiment_id: str, total_experiments: int):
        """Mark start of experiment run."""
        self.progress['total'] = total_experiments
        self.progress['start_time'] = datetime.now().isoformat()
        self.progress['running'] = [experiment_id]
        self.save_progress()
    
    def mark_completed(self, experiment_id: str):
        """Mark experiment as completed."""
        if experiment_id in self.progress['running']:
            self.progress['running'].remove(experiment_id)
        if experiment_id not in self.progress['completed']:
            self.progress['completed'].append(experiment_id)
        self.save_progress()
    
    def mark_failed(self, experiment_id: str, error: str):
        """Mark experiment as failed."""
        if experiment_id in self.progress['running']:
            self.progress['running'].remove(experiment_id)
        if experiment_id not in self.progress['failed']:
            self.progress['failed'].append({'id': experiment_id, 'error': error})
        self.save_progress()
    
    def get_completion_rate(self) -> float:
        """Get completion rate."""
        if self.progress['total'] == 0:
            return 0.0
        return len(self.progress['completed']) / self.progress['total']
    
    def print_status(self):
        """Print current experiment status."""
        completed = len(self.progress['completed'])
        failed = len(self.progress['failed'])
        running = len(self.progress['running'])
        total = self.progress['total']
        
        print(f"Experiment Status:")
        print(f"  Completed: {completed}/{total} ({completed/total*100:.1f}%)")
        print(f"  Failed: {failed}")
        print(f"  Running: {running}")
        
        if self.progress['start_time']:
            start_time = datetime.fromisoformat(self.progress['start_time'])
            elapsed = datetime.now() - start_time
            print(f"  Elapsed time: {elapsed}")
            
            if completed > 0:
                avg_time = elapsed / completed
                remaining = total - completed
                estimated_remaining = avg_time * remaining
                print(f"  Estimated remaining: {estimated_remaining}")
