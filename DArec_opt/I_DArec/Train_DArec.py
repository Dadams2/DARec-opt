import numpy as np
import torch.optim as optim
import torch.utils.data
from I_DArec import *
from torch.utils.data import DataLoader
from Data_Preprocessing import Mydata
from evaluation import EvaluationFramework, RMSEEvaluator, MAEEvaluator, RankingEvaluator, PrecisionRecallEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
import os
import datetime
import itertools
import json
import traceback
import glob
import re

def convert_to_json_serializable(obj):
    """Convert numpy/torch types to JSON serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj):
            return "NaN"
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # PyTorch tensors
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return "NaN" if np.isnan(val) else ("Infinity" if val > 0 else "-Infinity")
        return val
    elif isinstance(obj, (np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return "NaN" if np.isnan(obj) else ("Infinity" if obj > 0 else "-Infinity")
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return "NaN" if np.isnan(obj) else ("Infinity" if obj > 0 else "-Infinity")
        return obj
    else:
        return obj


def find_latest_autorec_models(models_dir, source_domain_name, target_domain_name):
    """
    Find the most recent timestamped AutoRec models for source and target domains.
    
    Args:
        models_dir: Directory containing saved AutoRec models
        source_domain_name: Name of source domain (e.g., 'Amazon_Instant_Video')
        target_domain_name: Name of target domain (e.g., 'Beauty')
    
    Returns:
        Tuple of (source_model_path, target_model_path) or (None, None) if not found
    """
    # Pattern for timestamped AutoRec models
    s_pattern = f"S_AutoRec_{source_domain_name}_{target_domain_name}_*_*.pkl"
    t_pattern = f"T_AutoRec_{source_domain_name}_{target_domain_name}_*_*.pkl"
    
    s_files = glob.glob(os.path.join(models_dir, s_pattern))
    t_files = glob.glob(os.path.join(models_dir, t_pattern))
    
    def extract_timestamp(filename):
        # Extract timestamp from filename like 'S_AutoRec_domain1_domain2_50_20250917_143025.pkl'
        match = re.search(r'_(\d{8}_\d{6})\.pkl$', filename)
        return match.group(1) if match else '00000000_000000'
    
    if s_files and t_files:
        # Find most recent files
        latest_s = max(s_files, key=extract_timestamp)
        latest_t = max(t_files, key=extract_timestamp)
        return latest_s, latest_t
    else:
        print(f"Warning: Could not find AutoRec models for {source_domain_name} -> {target_domain_name}")
        return None, None


def train_darec(source_domain_path, target_domain_path, config, 
                output_dir="./results", log_dir="./logs", models_dir="./models",
                s_pretrained_path=None, t_pretrained_path=None):
    """
    Train DARec model with configurable parameters.
    
    Args:
        source_domain_path: Path to source domain data CSV
        target_domain_path: Path to target domain data CSV
        config: Dictionary with training parameters
        output_dir: Directory to save trained DARec models
        log_dir: Directory to save training logs
        models_dir: Directory to search for AutoRec pretrained models
        s_pretrained_path: Manual override for source AutoRec model path
        t_pretrained_path: Manual override for target AutoRec model path
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    log_filename = f"darec_training_log_{source_name}_to_{target_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    print(f"Training DARec for {source_name} -> {target_name}")
    
    # Load datasets
    train_dataset = Mydata(source_domain_path, target_domain_path, train=True, preprocessed=True)
    test_dataset = Mydata(source_domain_path, target_domain_path, train=False, preprocessed=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Get data dimensions
    n_users = train_dataset.S_data.shape[1]
    S_n_items, T_n_items = train_dataset.S_data.shape[0], train_dataset.T_data.shape[0]
    
    print(f"Data loaded - Users: {n_users}, Source items: {S_n_items}, Target items: {T_n_items}")
    
    # Find or use provided AutoRec models
    if s_pretrained_path is None or t_pretrained_path is None:
        auto_s_path, auto_t_path = find_latest_autorec_models(models_dir, source_name, target_name)
        s_pretrained_path = s_pretrained_path or auto_s_path
        t_pretrained_path = t_pretrained_path or auto_t_path
    
    if s_pretrained_path is None or t_pretrained_path is None:
        raise FileNotFoundError(f"Could not find pretrained AutoRec models. Please provide paths manually or ensure AutoRec models exist in {models_dir}")
    
    print(f"Using pretrained models:")
    print(f"  Source: {s_pretrained_path}")
    print(f"  Target: {t_pretrained_path}")
    
    # Create DARec args object
    class DArec_Args:
        pass
    
    args = DArec_Args()
    args.epochs = config['epochs']
    args.batch_size = config['batch_size']
    args.lr = config['lr']
    args.wd = config['wd']
    args.n_factors = config['n_factors']
    args.n_users = n_users
    args.S_n_items = S_n_items
    args.T_n_items = T_n_items
    args.RPE_hidden_size = config.get('RPE_hidden_size', 200)
    args.S_pretrained_weights = s_pretrained_path
    args.T_pretrained_weights = t_pretrained_path
    
    # Create model
    net = I_DArec(args)
    net.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
    net.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
    net.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                          weight_decay=args.wd, lr=args.lr)
    RMSE = MRMSELoss().cuda()
    criterion = DArec_Loss().cuda()
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework()
    evaluator.add_ranking_evaluator(k=5)
    evaluator.add_precision_recall_evaluator(k=5)

    def train_epoch(epoch):
        net.train()
        total_loss = 0
        source_total_rmse = 0
        source_total_mask = 0
        target_total_rmse = 0
        target_total_mask = 0
        
        for idx, (source_rating, target_rating, source_labels, target_labels) in enumerate(train_loader):
            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()
            target_labels = target_labels.squeeze(1).long().cuda()

            optimizer.zero_grad()
            
            # Source domain forward pass
            is_source = True
            class_output, source_prediction, target_prediction = net(source_rating, is_source)
            source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                    source_rating, target_rating, source_labels)
            source_rmse, _ = RMSE(source_prediction, source_rating)
            source_total_rmse += source_rmse.item()
            source_total_mask += torch.sum(source_mask).item()
            loss = source_loss
            
            # Target domain forward pass
            is_source = False
            class_output, source_prediction, target_prediction = net(target_rating, is_source)
            target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                              source_rating, target_rating, target_labels)
            target_rmse, _ = RMSE(target_prediction, target_rating)
            target_total_rmse += target_rmse.item()
            target_total_mask += torch.sum(target_mask).item()
            loss += target_loss
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        source_rmse = math.sqrt(source_total_rmse / source_total_mask) if source_total_mask > 0 else float('inf')
        target_rmse = math.sqrt(target_total_rmse / target_total_mask) if target_total_mask > 0 else float('inf')
        
        return avg_loss, source_rmse, target_rmse

    def test_epoch():
        net.eval()
        total_loss = 0
        source_total_rmse = 0
        source_total_mask = 0
        target_total_rmse = 0
        target_total_mask = 0
        
        # For comprehensive evaluation
        all_source_preds = []
        all_source_targets = []
        all_source_masks = []
        all_target_preds = []
        all_target_targets = []
        all_target_masks = []
        
        with torch.no_grad():
            for idx, (source_rating, target_rating, source_labels, target_labels) in enumerate(test_loader):
                source_rating = source_rating.cuda()
                target_rating = target_rating.cuda()
                source_labels = source_labels.squeeze(1).long().cuda()
                target_labels = target_labels.squeeze(1).long().cuda()
                
                # Source domain evaluation
                is_source = True
                class_output, source_prediction, target_prediction = net(source_rating, is_source)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                        source_rating, target_rating, source_labels)
                source_rmse, _ = RMSE(source_prediction, source_rating)
                source_total_rmse += source_rmse.item()
                source_total_mask += torch.sum(source_mask).item()
                
                # Store for detailed evaluation
                all_source_preds.append(source_prediction)
                all_source_targets.append(source_rating)
                all_source_masks.append(source_mask)
                
                # Target domain evaluation
                is_source = False
                class_output_t, source_prediction_t, target_prediction_t = net(target_rating, is_source)
                target_loss, source_mask_t, target_mask_t = criterion(class_output_t, source_prediction_t, target_prediction_t,
                                                              source_rating, target_rating, target_labels)
                target_rmse, _ = RMSE(target_prediction_t, target_rating)
                target_total_rmse += target_rmse.item()
                target_total_mask += torch.sum(target_mask_t).item()
                
                # Store for detailed evaluation
                all_target_preds.append(target_prediction_t)
                all_target_targets.append(target_rating)
                all_target_masks.append(target_mask_t)
                
                total_loss += (source_loss + target_loss).item()
        
        # Calculate basic RMSE
        source_rmse = math.sqrt(source_total_rmse / source_total_mask) if source_total_mask > 0 else float('inf')
        target_rmse = math.sqrt(target_total_rmse / target_total_mask) if target_total_mask > 0 else float('inf')
        avg_loss = total_loss / len(test_loader)
        
        # Comprehensive evaluation
        source_metrics = {}
        target_metrics = {}
        
        if all_source_preds:
            source_preds_cat = torch.cat(all_source_preds, dim=0)
            source_targets_cat = torch.cat(all_source_targets, dim=0)
            source_masks_cat = torch.cat(all_source_masks, dim=0)
            source_metrics = evaluator.evaluate(source_preds_cat, source_targets_cat, source_masks_cat)
        
        if all_target_preds:
            target_preds_cat = torch.cat(all_target_preds, dim=0)
            target_targets_cat = torch.cat(all_target_targets, dim=0)
            target_masks_cat = torch.cat(all_target_masks, dim=0)
            target_metrics = evaluator.evaluate(target_preds_cat, target_targets_cat, target_masks_cat)
        
        return avg_loss, source_rmse, target_rmse, source_metrics, target_metrics

    # Training loop
    training_log = {
        'config': config,
        'source_domain': source_name,
        'target_domain': target_name,
        'pretrained_models': {
            'source': s_pretrained_path,
            'target': t_pretrained_path
        },
        'epochs': []
    }
    
    train_losses = []
    source_train_rmse = []
    source_test_rmse = []
    target_train_rmse = []
    target_test_rmse = []
    
    for epoch in tqdm(range(config['epochs']), desc=f"Training DARec {source_name}->{target_name}"):
        train_loss, s_tr_rmse, t_tr_rmse = train_epoch(epoch)
        test_loss, s_te_rmse, t_te_rmse, s_metrics, t_metrics = test_epoch()
        
        train_losses.append(train_loss)
        source_train_rmse.append(s_tr_rmse)
        source_test_rmse.append(s_te_rmse)
        target_train_rmse.append(t_tr_rmse)
        target_test_rmse.append(t_te_rmse)
        
        # Log epoch results
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'source_train_rmse': s_tr_rmse,
            'source_test_rmse': s_te_rmse,
            'target_train_rmse': t_tr_rmse,
            'target_test_rmse': t_te_rmse,
            'source_metrics': s_metrics,
            'target_metrics': t_metrics
        }
        training_log['epochs'].append(epoch_log)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config['epochs']}")
            print(f"  Source RMSE: {s_te_rmse:.4f}, Target RMSE: {t_te_rmse:.4f}")
            if 'ndcg@5' in s_metrics:
                print(f"  Source NDCG@5: {s_metrics['ndcg@5']:.4f}")
            if 'ndcg@5' in t_metrics:
                print(f"  Target NDCG@5: {t_metrics['ndcg@5']:.4f}")
            if 'precision@5' in s_metrics:
                print(f"  Source Precision@5: {s_metrics['precision@5']:.4f}")
            if 'precision@5' in t_metrics:
                print(f"  Target Precision@5: {t_metrics['precision@5']:.4f}")
            if 'recall@5' in s_metrics:
                print(f"  Source Recall@5: {s_metrics['recall@5']:.4f}")
            if 'recall@5' in t_metrics:
                print(f"  Target Recall@5: {t_metrics['recall@5']:.4f}")
            if 'hr@5' in s_metrics:
                print(f"  Source HR@5: {s_metrics['hr@5']:.4f}")
            if 'hr@5' in t_metrics:
                print(f"  Target HR@5: {t_metrics['hr@5']:.4f}")
    
    # Save final model
    model_path = os.path.join(output_dir, f"DARec_{source_name}_{target_name}_{timestamp}.pkl")
    torch.save(net.state_dict(), model_path)
    
    # Save training log
    training_log_serializable = convert_to_json_serializable(training_log)
    with open(log_path, 'w') as f:
        json.dump(training_log_serializable, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best source test RMSE: {min(source_test_rmse):.4f}")
    print(f"Best target test RMSE: {min(target_test_rmse):.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")
    
    return {
        'source_best_rmse': min(source_test_rmse),
        'target_best_rmse': min(target_test_rmse),
        'model_path': model_path,
        'log_path': log_path,
        'final_source_metrics': training_log['epochs'][-1]['source_metrics'],
        'final_target_metrics': training_log['epochs'][-1]['target_metrics']
    }


def grid_search_darec_hyperparameters(source_domain_path, target_domain_path, param_grid, 
                                     output_dir="./results", log_dir="./logs", models_dir="./models"):
    """
    Perform grid search over DARec hyperparameters.
    
    Args:
        source_domain_path: Path to source domain data CSV
        target_domain_path: Path to target domain data CSV
        param_grid: Dictionary with parameter names as keys and lists of values to try
        output_dir: Directory to save trained models
        log_dir: Directory to save training logs
        models_dir: Directory to search for AutoRec pretrained models
    
    Example param_grid:
    {
        'epochs': [50, 70],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400],
        'RPE_hidden_size': [200, 300]
    }
    """
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    
    print(f"Starting DARec grid search with {len(param_combinations)} combinations for {source_name} -> {target_name}")
    
    for i, param_combo in enumerate(param_combinations):
        config = dict(zip(param_names, param_combo))
        print(f"\nDARec Grid search {i+1}/{len(param_combinations)}: {config}")
        
        try:
            result = train_darec(
                source_domain_path, 
                target_domain_path, 
                config, 
                output_dir, 
                log_dir,
                models_dir
            )
            result['config'] = config
            result['combination_id'] = i + 1
            results.append(result)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error with DARec configuration {config}: {e}")
            print(f"Full traceback:\n{error_traceback}")
            results.append({
                'config': config,
                'combination_id': i + 1,
                'error': str(e),
                'error_traceback': error_traceback,
                'source_best_rmse': float('inf'),
                'target_best_rmse': float('inf')
            })
    
    # Save grid search results
    grid_search_log = {
        'source_domain': source_name,
        'target_domain': target_name,
        'param_grid': param_grid,
        'results': results,
        'timestamp': timestamp
    }
    
    grid_search_path = os.path.join(log_dir, f"darec_grid_search_{source_name}_to_{target_name}_{timestamp}.json")
    grid_search_log_serializable = convert_to_json_serializable(grid_search_log)
    with open(grid_search_path, 'w') as f:
        json.dump(grid_search_log_serializable, f, indent=2)
    
    # Find best configurations
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_source = min(valid_results, key=lambda x: x['source_best_rmse'])
        best_target = min(valid_results, key=lambda x: x['target_best_rmse'])
        
        print(f"\nDARec grid search completed!")
        print(f"Best source RMSE: {best_source['source_best_rmse']:.4f} with config: {best_source['config']}")
        print(f"Best target RMSE: {best_target['target_best_rmse']:.4f} with config: {best_target['config']}")
        print(f"Grid search results saved to: {grid_search_path}")
        
        return {
            'best_source_config': best_source,
            'best_target_config': best_target,
            'all_results': results,
            'grid_search_path': grid_search_path
        }
    else:
        print("All DARec configurations failed!")
        return {'all_results': results, 'grid_search_path': grid_search_path}


def run_multi_domain_darec_experiments(domain_pairs, config=None, param_grid=None, 
                                       output_dir="./results", log_dir="./logs", models_dir="./models"):
    """
    Run DARec experiments across multiple domain pairs.
    
    Args:
        domain_pairs: List of tuples (source_path, target_path) for domain combinations
        config: Single configuration dict for simple training (mutually exclusive with param_grid)
        param_grid: Parameter grid for grid search (mutually exclusive with config)
        output_dir: Directory to save trained models
        log_dir: Directory to save training logs
        models_dir: Directory to search for AutoRec pretrained models
    """
    if config is not None and param_grid is not None:
        raise ValueError("Provide either config or param_grid, not both")
    if config is None and param_grid is None:
        raise ValueError("Must provide either config or param_grid")
    
    all_results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, (source_path, target_path) in enumerate(domain_pairs):
        print(f"\n{'='*60}")
        print(f"DARec Domain pair {i+1}/{len(domain_pairs)}")
        print(f"Source: {source_path}")
        print(f"Target: {target_path}")
        print(f"{'='*60}")
        
        try:
            if config is not None:
                # Single configuration training
                result = train_darec(source_path, target_path, config, output_dir, log_dir, models_dir)
                result['domain_pair'] = (source_path, target_path)
                all_results.append(result)
            else:
                # Grid search
                result = grid_search_darec_hyperparameters(source_path, target_path, param_grid, 
                                                          output_dir, log_dir, models_dir)
                result['domain_pair'] = (source_path, target_path)
                all_results.append(result)
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error processing DARec domain pair {source_path} -> {target_path}: {e}")
            print(f"Full traceback:\n{error_traceback}")
            all_results.append({
                'domain_pair': (source_path, target_path),
                'error': str(e),
                'error_traceback': error_traceback
            })
    
    # Save overall experiment results
    experiment_summary = {
        'timestamp': timestamp,
        'domain_pairs': domain_pairs,
        'experiment_type': 'single_config' if config is not None else 'grid_search',
        'config': config,
        'param_grid': param_grid,
        'results': all_results
    }
    
    summary_path = os.path.join(log_dir, f"darec_multi_domain_experiment_{timestamp}.json")
    experiment_summary_serializable = convert_to_json_serializable(experiment_summary)
    with open(summary_path, 'w') as f:
        json.dump(experiment_summary_serializable, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DARec multi-domain experiment completed!")
    print(f"Processed {len(domain_pairs)} domain pairs")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}")
    
    return {
        'experiment_summary': experiment_summary,
        'summary_path': summary_path
    }


# Example usage functions
def create_default_darec_config():
    """Create a default DARec configuration dictionary."""
    return {
        'epochs': 70,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200,
        'RPE_hidden_size': 200
    }


def create_default_darec_param_grid():
    """Create a default parameter grid for DARec grid search."""
    return {
        'epochs': [50, 70],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400],
        'RPE_hidden_size': [200, 300]
    }


if __name__ == "__main__":
    
    # Define domain pairs 
    base_data_dir = "../../data" 
    domain_pairs = [
        (f"{base_data_dir}/ratings_Apps_for_Android.csv", f"{base_data_dir}/ratings_Video_Games.csv"),
        (f"{base_data_dir}/ratings_Toys_and_Games.csv", f"{base_data_dir}/ratings_Automotive.csv"),
        # Add more domain pairs as needed
    ]
    
    # Option 1: Single configuration training
    config = create_default_darec_config()
    results = run_multi_domain_darec_experiments(
        domain_pairs=domain_pairs,
        config=config,
        output_dir="./darec_models",
        log_dir="./darec_logs",
        models_dir="./models"  # Directory where AutoRec models are saved
    )
    
    # Option 2: Grid search (comment out the above and uncomment below)
    # param_grid = create_default_darec_param_grid()
    # results = run_multi_domain_darec_experiments(
    #     domain_pairs=domain_pairs,
    #     param_grid=param_grid,
    #     output_dir="./darec_models",
    #     log_dir="./darec_logs",
    #     models_dir="./models"  # Directory where AutoRec models are saved
    # )
    
    # Option 3: Single domain pair with manual AutoRec model paths
    # result = train_darec(
    #     source_domain_path=f"{base_data_dir}/ratings_Amazon_Instant_Video.csv",
    #     target_domain_path=f"{base_data_dir}/ratings_Beauty.csv",
    #     config=config,
    #     output_dir="./darec_models",
    #     log_dir="./darec_logs",
    #     s_pretrained_path="./models/S_AutoRec_Amazon_Instant_Video_Beauty_50_20250917_143025.pkl",
    #     t_pretrained_path="./models/T_AutoRec_Amazon_Instant_Video_Beauty_50_20250917_143025.pkl"
    # )