"""
Evaluation framework for DARec experiments.
"""
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings


class BaseEvaluator(ABC):
    """Base class for evaluators."""
    
    @abstractmethod
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate predictions against targets with given mask."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get evaluator name."""
        pass


class RMSEEvaluator(BaseEvaluator):
    """Root Mean Square Error evaluator."""
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate RMSE."""
        masked_pred = predictions * mask.float()
        masked_target = targets * mask.float()
        
        mse = torch.sum((masked_pred - masked_target) ** 2) / torch.sum(mask)
        rmse = torch.sqrt(mse).item()
        
        return {"rmse": rmse}
    
    def get_name(self) -> str:
        return "RMSE"


class MAEEvaluator(BaseEvaluator):
    """Mean Absolute Error evaluator."""
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate MAE."""
        masked_pred = predictions * mask.float()
        masked_target = targets * mask.float()
        
        mae = torch.sum(torch.abs(masked_pred - masked_target)) / torch.sum(mask)
        
        return {"mae": mae.item()}
    
    def get_name(self) -> str:
        return "MAE"


class MAPEEvaluator(BaseEvaluator):
    """Mean Absolute Percentage Error evaluator."""
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate MAPE."""
        # Avoid division by zero
        nonzero_mask = (targets != 0) & mask
        
        if torch.sum(nonzero_mask) == 0:
            return {"mape": float('inf')}
        
        masked_pred = predictions * nonzero_mask.float()
        masked_target = targets * nonzero_mask.float()
        
        mape = torch.sum(torch.abs((masked_target - masked_pred) / masked_target)) / torch.sum(nonzero_mask) * 100
        
        return {"mape": mape.item()}
    
    def get_name(self) -> str:
        return "MAPE"


class PrecisionRecallEvaluator(BaseEvaluator):
    """Precision and Recall evaluator."""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate precision and recall@k."""
        # Convert to numpy for easier manipulation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        
        precision_scores = []
        recall_scores = []
        
        for user_idx in range(pred_np.shape[0]):
            user_pred = pred_np[user_idx]
            user_target = target_np[user_idx]
            user_mask = mask_np[user_idx]
            
            if np.sum(user_mask) == 0:
                continue
            
            # Get top-k recommendations
            masked_pred = user_pred.copy()
            masked_pred[user_mask == 0] = -np.inf
            
            top_k_indices = np.argsort(masked_pred)[::-1][:self.k]
            
            # Calculate relevant items (items that user actually rated positively)
            # Assuming ratings > 3 are relevant (4 and 5 star ratings)
            relevant_items = set(np.where(user_target > 3)[0])
            recommended_items = set(top_k_indices)
            
            # Calculate precision and recall
            if len(recommended_items) > 0:
                true_positives = len(relevant_items.intersection(recommended_items))
                precision = true_positives / len(recommended_items)
                precision_scores.append(precision)
                
                if len(relevant_items) > 0:
                    recall = true_positives / len(relevant_items)
                    recall_scores.append(recall)
        
        metrics = {}
        if precision_scores:
            metrics[f"precision@{self.k}"] = np.mean(precision_scores)
        if recall_scores:
            metrics[f"recall@{self.k}"] = np.mean(recall_scores)
        
        return metrics
    
    def get_name(self) -> str:
        return f"PrecisionRecall@{self.k}"


class RankingEvaluator(BaseEvaluator):
    """Ranking-based evaluation metrics."""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate ranking metrics."""
        # Convert to numpy for easier manipulation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        
        metrics = {}
        
        # Calculate per-user metrics
        ndcg_scores = []
        hr_scores = []
        
        for user_idx in range(pred_np.shape[0]):
            user_pred = pred_np[user_idx]
            user_target = target_np[user_idx]
            user_mask = mask_np[user_idx]
            
            if np.sum(user_mask) == 0:
                continue
            
            # Get top-k recommendations
            masked_pred = user_pred.copy()
            masked_pred[user_mask == 0] = -np.inf
            
            top_k_indices = np.argsort(masked_pred)[::-1][:self.k]
            top_k_targets = user_target[top_k_indices]
            
            # Calculate NDCG@k
            dcg = self._calculate_dcg(top_k_targets)
            ideal_targets = np.sort(user_target[user_target > 3])[::-1][:self.k]  # Only consider ratings > 3
            idcg = self._calculate_dcg(ideal_targets)
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            
            # Calculate HR@k (Hit Rate)
            # Using rating > 3 as relevant (consistent with precision/recall)
            hr = np.sum(top_k_targets > 3) / min(self.k, np.sum(user_target > 3))
            hr_scores.append(hr)
        
        if ndcg_scores:
            metrics[f"ndcg@{self.k}"] = np.mean(ndcg_scores)
        if hr_scores:
            metrics[f"hr@{self.k}"] = np.mean(hr_scores)
        
        return metrics
    
    def _calculate_dcg(self, relevances: np.ndarray) -> float:
        """Calculate Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rel in enumerate(relevances):
            if rel > 3:  # Only count ratings > 3 as relevant
                dcg += rel / np.log2(i + 2)
        return dcg
    
    def get_name(self) -> str:
        return f"Ranking@{self.k}"


class CoverageEvaluator(BaseEvaluator):
    """Item coverage evaluator."""
    
    def __init__(self, total_items: int, threshold: float = 0.0):
        self.total_items = total_items
        self.threshold = threshold
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate item coverage."""
        # Items recommended above threshold
        recommended_items = torch.sum(predictions > self.threshold, dim=0)
        covered_items = torch.sum(recommended_items > 0).item()
        
        coverage = covered_items / self.total_items
        
        return {"item_coverage": coverage}
    
    def get_name(self) -> str:
        return "Coverage"


class NoveltyEvaluator(BaseEvaluator):
    """Novelty evaluator based on item popularity."""
    
    def __init__(self, item_popularity: torch.Tensor):
        self.item_popularity = item_popularity
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate novelty score."""
        # Get top recommendations per user
        top_items = torch.argmax(predictions, dim=1)
        
        # Calculate average novelty (inverse popularity)
        novelty_scores = []
        for item_idx in top_items:
            if item_idx < len(self.item_popularity):
                popularity = self.item_popularity[item_idx].item()
                novelty = -np.log2(popularity + 1e-8)  # Higher for less popular items
                novelty_scores.append(novelty)
        
        return {"novelty": np.mean(novelty_scores) if novelty_scores else 0.0}
    
    def get_name(self) -> str:
        return "Novelty"


class DiversityEvaluator(BaseEvaluator):
    """Diversity evaluator based on item similarity."""
    
    def __init__(self, item_similarity_matrix: torch.Tensor, k: int = 10):
        self.item_similarity = item_similarity_matrix
        self.k = k
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate intra-list diversity."""
        diversity_scores = []
        
        for user_idx in range(predictions.shape[0]):
            user_pred = predictions[user_idx]
            user_mask = mask[user_idx]
            
            # Get top-k items
            masked_pred = user_pred.clone()
            masked_pred[user_mask == 0] = -np.inf
            top_k_items = torch.topk(masked_pred, min(self.k, torch.sum(user_mask).item())).indices
            
            if len(top_k_items) < 2:
                continue
            
            # Calculate pairwise diversity
            diversity = 0.0
            count = 0
            for i in range(len(top_k_items)):
                for j in range(i + 1, len(top_k_items)):
                    item_i, item_j = top_k_items[i], top_k_items[j]
                    if item_i < self.item_similarity.shape[0] and item_j < self.item_similarity.shape[1]:
                        similarity = self.item_similarity[item_i, item_j].item()
                        diversity += 1 - similarity
                        count += 1
            
            if count > 0:
                diversity_scores.append(diversity / count)
        
        return {"diversity": np.mean(diversity_scores) if diversity_scores else 0.0}
    
    def get_name(self) -> str:
        return f"Diversity@{self.k}"


class StatisticalEvaluator(BaseEvaluator):
    """Statistical analysis evaluator."""
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Calculate statistical metrics."""
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        mask_np = mask.detach().cpu().numpy().flatten()
        
        # Filter by mask
        valid_pred = pred_np[mask_np == 1]
        valid_target = target_np[mask_np == 1]
        
        if len(valid_pred) == 0:
            return {}
        
        metrics = {}
        
        # Correlation
        if len(valid_pred) > 1:
            correlation, p_value = stats.pearsonr(valid_pred, valid_target)
            metrics["correlation"] = correlation
            metrics["correlation_p_value"] = p_value
        
        # Bias (mean error)
        bias = np.mean(valid_pred - valid_target)
        metrics["bias"] = bias
        
        # Standard deviation of errors
        error_std = np.std(valid_pred - valid_target)
        metrics["error_std"] = error_std
        
        # R-squared
        ss_res = np.sum((valid_target - valid_pred) ** 2)
        ss_tot = np.sum((valid_target - np.mean(valid_target)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        metrics["r_squared"] = r_squared
        
        return metrics
    
    def get_name(self) -> str:
        return "Statistical"


class EvaluationFramework:
    """Main evaluation framework that coordinates multiple evaluators."""
    
    def __init__(self):
        self.evaluators: Dict[str, BaseEvaluator] = {}
        self.register_default_evaluators()
    
    def register_default_evaluators(self):
        """Register default evaluators."""
        self.register_evaluator("rmse", RMSEEvaluator())
        self.register_evaluator("mae", MAEEvaluator())
        self.register_evaluator("mape", MAPEEvaluator())
        self.register_evaluator("statistical", StatisticalEvaluator())
    
    def register_evaluator(self, name: str, evaluator: BaseEvaluator):
        """Register a new evaluator."""
        self.evaluators[name] = evaluator
    
    def add_ranking_evaluator(self, k: int = 10):
        """Add ranking evaluator with specific k."""
        self.register_evaluator(f"ranking_{k}", RankingEvaluator(k))
    
    def add_precision_recall_evaluator(self, k: int = 5):
        """Add precision/recall evaluator with specific k."""
        self.register_evaluator(f"precision_recall_{k}", PrecisionRecallEvaluator(k))
    
    def add_coverage_evaluator(self, total_items: int, threshold: float = 0.0):
        """Add coverage evaluator."""
        self.register_evaluator("coverage", CoverageEvaluator(total_items, threshold))
    
    def add_novelty_evaluator(self, item_popularity: torch.Tensor):
        """Add novelty evaluator."""
        self.register_evaluator("novelty", NoveltyEvaluator(item_popularity))
    
    def add_diversity_evaluator(self, item_similarity_matrix: torch.Tensor, k: int = 10):
        """Add diversity evaluator."""
        self.register_evaluator("diversity", DiversityEvaluator(item_similarity_matrix, k))
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, 
                 evaluator_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Run evaluation with specified evaluators."""
        if evaluator_names is None:
            evaluator_names = list(self.evaluators.keys())
        
        results = {}
        
        for name in evaluator_names:
            if name in self.evaluators:
                try:
                    evaluator_results = self.evaluators[name].evaluate(predictions, targets, mask)
                    results.update(evaluator_results)
                except Exception as e:
                    warnings.warn(f"Evaluator {name} failed: {str(e)}")
            else:
                warnings.warn(f"Evaluator {name} not found")
        
        return results
    
    def evaluate_method(self, method, test_loader, evaluator_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate a trained method on test data."""
        method.model.eval()
        
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # Handle different data formats
                if len(data) == 4:  # DARec format
                    source_rating, target_rating, _, _ = data
                    # Use target rating for evaluation
                    rating = target_rating.to(method.device)
                    
                    # Get predictions
                    if hasattr(method, 'variant') and method.variant == "U":
                        _, _, pred = method.model(rating, 1.0, False)
                    else:
                        _, _, pred = method.model(rating, False)
                        
                elif len(data) == 2:  # AutoRec format
                    rating, _ = data
                    rating = rating.to(method.device)
                    _, pred = method.model(rating)
                else:
                    rating = data[0].to(method.device)
                    _, pred = method.model(rating)
                
                # Create mask (non-zero entries)
                mask = (rating != 0)
                
                all_predictions.append(pred)
                all_targets.append(rating)
                all_masks.append(mask)
        
        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0)
        
        return self.evaluate(predictions, targets, masks, evaluator_names)
    
    def create_evaluation_report(self, results: Dict[str, float], method_name: str, 
                                domain_pair: Tuple[str, str]) -> str:
        """Create a formatted evaluation report."""
        source, target = domain_pair
        
        report = f"Evaluation Report: {method_name}\n"
        report += f"Domain Transfer: {source} → {target}\n"
        report += "=" * 50 + "\n\n"
        
        # Group metrics by type
        basic_metrics = ["rmse", "mae", "mape"]
        ranking_metrics = [k for k in results.keys() if k.startswith(("ndcg", "hr"))]
        other_metrics = [k for k in results.keys() if k not in basic_metrics and k not in ranking_metrics]
        
        if any(metric in results for metric in basic_metrics):
            report += "Basic Metrics:\n"
            for metric in basic_metrics:
                if metric in results:
                    report += f"  {metric.upper()}: {results[metric]:.4f}\n"
            report += "\n"
        
        if ranking_metrics:
            report += "Ranking Metrics:\n"
            for metric in ranking_metrics:
                report += f"  {metric.upper()}: {results[metric]:.4f}\n"
            report += "\n"
        
        if other_metrics:
            report += "Additional Metrics:\n"
            for metric in other_metrics:
                report += f"  {metric}: {results[metric]:.4f}\n"
            report += "\n"
        
        return report
    
    def compare_methods(self, results_list: List[Dict[str, Any]], 
                       metric: str = "rmse") -> pd.DataFrame:
        """Compare multiple methods on a specific metric."""
        comparison_data = []
        
        for result in results_list:
            if metric in result.get('metrics', {}):
                comparison_data.append({
                    'method': result.get('method', 'Unknown'),
                    'source_domain': result.get('source_domain', 'Unknown'),
                    'target_domain': result.get('target_domain', 'Unknown'),
                    'metric_value': result['metrics'][metric],
                    'hyperparameters': result.get('hyperparameters', {})
                })
        
        return pd.DataFrame(comparison_data)
    
    def plot_metric_comparison(self, comparison_df: pd.DataFrame, metric: str, 
                              save_path: Optional[str] = None):
        """Plot metric comparison across methods."""
        plt.figure(figsize=(12, 8))
        
        if 'method' in comparison_df.columns:
            sns.boxplot(data=comparison_df, x='method', y='metric_value')
            plt.title(f'{metric.upper()} Comparison Across Methods')
            plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def statistical_significance_test(self, results1: List[float], results2: List[float], 
                                     test_type: str = "ttest") -> Dict[str, float]:
        """Perform statistical significance test between two sets of results."""
        if test_type == "ttest":
            statistic, p_value = stats.ttest_ind(results1, results2)
        elif test_type == "wilcoxon":
            statistic, p_value = stats.ranksums(results1, results2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
