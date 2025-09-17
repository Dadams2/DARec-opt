from U_DArec import *
from torch.utils.data import DataLoader
from Data_Preprocessing import Mydata
from evaluation import EvaluationFramework, RMSEEvaluator, MAEEvaluator, RankingEvaluator, PrecisionRecallEvaluator
from function import MRMSELoss, DArec_Loss
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import math
import os
import datetime
import itertools
import json
import traceback
import glob

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
        # Extract timestamp from filename like "S_AutoRec_domain1_domain2_20240917_143022_20.pkl"
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 6:
            return parts[4] + '_' + parts[5]  # timestamp part
        return '00000000_000000'
    
    if s_files and t_files:
        s_latest = max(s_files, key=extract_timestamp)
        t_latest = max(t_files, key=extract_timestamp)
        return s_latest, t_latest
    else:
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
    
    # Load datasets
    train_dataset = Mydata(source_domain_path, target_domain_path, train=True, preprocessed=True)
    test_dataset = Mydata(source_domain_path, target_domain_path, train=False, preprocessed=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Data loaded for DARec training: {source_name} -> {target_name}")
    
    # Set up model arguments
    class Args:
        def __init__(self):
            self.n_users = train_dataset.S_data.shape[0]
            self.S_n_items = train_dataset.S_data.shape[1]
            self.T_n_items = train_dataset.T_data.shape[1]
            self.n_factors = config['n_factors']
            self.RPE_hidden_size = config['RPE_hidden_size']
    
    args = Args()
    
    # Find or use provided AutoRec models
    if s_pretrained_path is None or t_pretrained_path is None:
        print(f"Searching for AutoRec models in {models_dir}...")
        s_model_path, t_model_path = find_latest_autorec_models(models_dir, source_name, target_name)
        
        if s_model_path is None or t_model_path is None:
            raise FileNotFoundError(f"Could not find AutoRec models for {source_name} -> {target_name} in {models_dir}")
        
        print(f"Found source AutoRec model: {s_model_path}")
        print(f"Found target AutoRec model: {t_model_path}")
    else:
        s_model_path = s_pretrained_path
        t_model_path = t_pretrained_path
        print(f"Using provided source AutoRec model: {s_model_path}")
        print(f"Using provided target AutoRec model: {t_model_path}")
    
    # Create DARec model
    net = U_DArec(args)
    
    # Load pretrained AutoRec weights
    try:
        net.S_autorec.load_state_dict(torch.load(s_model_path))
        net.T_autorec.load_state_dict(torch.load(t_model_path))
        print("AutoRec models loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load AutoRec models: {str(e)}")
    
    net.cuda()
    
    # Set up optimizer and loss functions
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                          weight_decay=config['wd'], lr=config['lr'])
    RMSE = MRMSELoss().cuda()
    criterion = DArec_Loss(lamda=config.get('lamda', 0.001), 
                          u=config.get('u', 1.0), 
                          beta=config.get('beta', 0.001)).cuda()
    
    # Set up evaluation framework
    eval_framework = EvaluationFramework()
    eval_framework.register_default_evaluators()
    eval_framework.add_precision_recall_evaluator(k=5)
    eval_framework.add_ranking_evaluator(k=5)
    
    def train_epoch(epoch):
        net.train()
        Total_RMSE = 0
        Total_MASK = 0
        total_loss = 0
        
        for idx, d in enumerate(train_loader):
            # alpha parameter for domain adaptation (similar to DANN)
            p = float(idx + epoch * len(train_loader)) / config['epochs'] / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            source_rating, target_rating, source_labels, target_labels = d
            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()
            target_labels = target_labels.squeeze(1).long().cuda()

            optimizer.zero_grad()
            
            # Forward pass on source data
            class_output_s, source_prediction_s, target_prediction_s = net(source_rating, alpha, True)
            
            # Forward pass on target data  
            class_output_t, source_prediction_t, target_prediction_t = net(target_rating, alpha, False)
            
            # Combine predictions and labels
            all_class_output = torch.cat([class_output_s, class_output_t], dim=0)
            all_source_pred = torch.cat([source_prediction_s, source_prediction_t], dim=0)
            all_target_pred = torch.cat([target_prediction_s, target_prediction_t], dim=0)
            all_source_rating = torch.cat([source_rating, source_rating], dim=0) 
            all_target_rating = torch.cat([target_rating, target_rating], dim=0)
            all_labels = torch.cat([source_labels, target_labels], dim=0)
            
            # Calculate loss
            loss, source_mask, target_mask = criterion(
                all_class_output, all_source_pred, all_target_pred,
                all_source_rating, all_target_rating, all_labels
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track RMSE on target domain
            target_rmse_loss, target_rmse_mask = RMSE(target_prediction_t, target_rating)
            Total_RMSE += target_rmse_loss.item()
            Total_MASK += torch.sum(target_rmse_mask).item()

        avg_loss = total_loss / len(train_loader)
        target_rmse = math.sqrt(Total_RMSE / Total_MASK) if Total_MASK > 0 else float('inf')
        return avg_loss, target_rmse

    def test_epoch():
        net.eval()
        Total_RMSE = 0
        Total_MASK = 0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for idx, d in enumerate(test_loader):
                source_rating, target_rating, source_labels, target_labels = d
                source_rating = source_rating.cuda()
                target_rating = target_rating.cuda()
                
                alpha = 1.0  # Fixed alpha for testing
                
                # Test on target data (main evaluation)
                class_output, source_prediction, target_prediction = net(target_rating, alpha, False)
                
                # Calculate RMSE
                rmse_loss, rmse_mask = RMSE(target_prediction, target_rating)
                Total_RMSE += rmse_loss.item()
                Total_MASK += torch.sum(rmse_mask).item()
                
                # Store for comprehensive evaluation
                all_predictions.append(target_prediction.cpu())
                all_targets.append(target_rating.cpu())
                all_masks.append(rmse_mask.cpu())

        target_rmse = math.sqrt(Total_RMSE / Total_MASK) if Total_MASK > 0 else float('inf')
        
        # Comprehensive evaluation
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        eval_results = eval_framework.evaluate(all_predictions, all_targets, all_masks)
        eval_results['target_rmse'] = target_rmse
        
        return eval_results

    # Training loop
    training_log = {
        'config': config,
        'source_domain': source_name,
        'target_domain': target_name,
        'source_model_path': s_model_path,
        'target_model_path': t_model_path,
        'epochs': []
    }
    
    train_losses = []
    train_rmses = []
    test_results = []
    
    best_rmse = float('inf')
    best_epoch = -1
    
    for epoch in tqdm(range(config['epochs'])):
        train_loss, train_rmse = train_epoch(epoch)
        test_eval = test_epoch()
        
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)
        test_results.append(test_eval)
        
        # Track best model
        current_rmse = test_eval['target_rmse']
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_epoch = epoch
            
            # Save best model
            best_model_path = os.path.join(output_dir, f"best_darec_{source_name}_to_{target_name}_{timestamp}.pkl")
            torch.save(net.state_dict(), best_model_path)
        
        # Log epoch results
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'test_evaluation': test_eval,
            'timestamp': datetime.datetime.now().isoformat()
        }
        training_log['epochs'].append(epoch_log)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
                  f"Test RMSE: {current_rmse:.4f}, Test NDCG@5: {test_eval.get('ndcg@5', 'N/A'):.4f}")
    
    # Final model save
    final_model_path = os.path.join(output_dir, f"final_darec_{source_name}_to_{target_name}_{timestamp}.pkl")
    torch.save(net.state_dict(), final_model_path)
    
    # Add summary to training log
    training_log['summary'] = {
        'best_epoch': best_epoch + 1,
        'best_rmse': best_rmse,
        'final_rmse': test_results[-1]['target_rmse'],
        'final_evaluation': test_results[-1],
        'best_model_path': best_model_path,
        'final_model_path': final_model_path
    }
    
    # Save training log
    with open(log_path, 'w') as f:
        json.dump(convert_to_json_serializable(training_log), f, indent=2)
    
    print(f"\nDARec training completed!")
    print(f"Best RMSE: {best_rmse:.4f} at epoch {best_epoch + 1}")
    print(f"Final RMSE: {test_results[-1]['target_rmse']:.4f}")
    print(f"Training log saved to: {log_path}")
    print(f"Best model saved to: {best_model_path}")
    
    return {
        'train_losses': train_losses,
        'train_rmses': train_rmses,
        'test_results': test_results,
        'best_rmse': best_rmse,
        'best_epoch': best_epoch,
        'final_evaluation': test_results[-1],
        'log_path': log_path,
        'best_model_path': best_model_path
    }


def grid_search_darec_hyperparameters(source_domain_path, target_domain_path, param_grid, 
                                     output_dir="./results", log_dir="./logs", models_dir="./models"):
    """
    Perform grid search over DARec hyperparameters.
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    
    print(f"Starting DARec grid search with {len(param_combinations)} parameter combinations")
    print(f"Parameters: {param_names}")
    
    results = []
    for i, param_combo in enumerate(param_combinations):
        config = dict(zip(param_names, param_combo))
        print(f"\nDARec Configuration {i+1}/{len(param_combinations)}: {config}")
        
        try:
            result = train_darec(source_domain_path, target_domain_path, config, 
                               output_dir, log_dir, models_dir)
            result['config'] = config
            result['config_id'] = i + 1
            results.append(result)
        except Exception as e:
            print(f"Error in DARec configuration {i+1}: {str(e)}")
            traceback.print_exc()
            results.append({
                'config': config,
                'config_id': i + 1,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    # Save grid search results
    grid_search_log = {
        'experiment_type': 'darec_grid_search',
        'param_grid': param_grid,
        'total_combinations': len(param_combinations),
        'results': results,
        'timestamp': timestamp
    }
    
    grid_search_path = os.path.join(log_dir, f"darec_grid_search_{source_name}_to_{target_name}_{timestamp}.json")
    with open(grid_search_path, 'w') as f:
        json.dump(convert_to_json_serializable(grid_search_log), f, indent=2)
    
    # Find best configurations
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_rmse = min(valid_results, key=lambda x: x['best_rmse'])
        best_ndcg = max(valid_results, key=lambda x: x['final_evaluation'].get('ndcg@5', 0))
        
        print(f"\n{'='*60}")
        print(f"DARec grid search completed! Results saved to: {grid_search_path}")
        print(f"Best RMSE: {best_rmse['best_rmse']:.4f} with config: {best_rmse['config']}")
        print(f"Best NDCG@5: {best_ndcg['final_evaluation'].get('ndcg@5', 0):.4f} with config: {best_ndcg['config']}")
    else:
        print("No valid DARec configurations completed successfully!")
    
    return results


def run_multi_domain_darec_experiments(domain_pairs, config=None, param_grid=None, 
                                       output_dir="./results", log_dir="./logs", models_dir="./models"):
    """
    Run DARec experiments across multiple domain pairs.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting multi-domain DARec experiments across {len(domain_pairs)} domain pairs")
    
    all_results = []
    for i, (source_path, target_path) in enumerate(domain_pairs):
        source_name = os.path.basename(source_path).replace('.csv', '').replace('ratings_', '')
        target_name = os.path.basename(target_path).replace('.csv', '').replace('ratings_', '')
        
        print(f"\n{'='*60}")
        print(f"DARec Domain pair {i+1}/{len(domain_pairs)}: {source_name} -> {target_name}")
        print(f"{'='*60}")
        
        try:
            if param_grid is not None:
                # Grid search for this domain pair
                results = grid_search_darec_hyperparameters(source_path, target_path, param_grid, 
                                                           output_dir, log_dir, models_dir)
            else:
                # Single configuration for this domain pair
                if config is None:
                    config = create_default_darec_config()
                results = train_darec(source_path, target_path, config, output_dir, log_dir, models_dir)
            
            all_results.append({
                'source_domain': source_name,
                'target_domain': target_name,
                'source_path': source_path,
                'target_path': target_path,
                'results': results,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error in DARec domain pair {source_name} -> {target_name}: {str(e)}")
            traceback.print_exc()
            all_results.append({
                'source_domain': source_name,
                'target_domain': target_name,
                'source_path': source_path,
                'target_path': target_path,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.datetime.now().isoformat()
            })
    
    # Save overall experiment summary
    experiment_summary = {
        'experiment_type': 'multi_domain_darec',
        'total_domain_pairs': len(domain_pairs),
        'config': config if config else None,
        'param_grid': param_grid if param_grid else None,
        'timestamp': timestamp,
        'results': all_results
    }
    
    summary_path = os.path.join(log_dir, f"multi_domain_darec_experiment_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_to_json_serializable(experiment_summary), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Multi-domain DARec experiment completed!")
    print(f"Processed {len(domain_pairs)} domain pairs")
    print(f"Results saved to: {summary_path}")
    
    return all_results


# Example usage functions
def create_default_darec_config():
    return {
        'epochs': 20,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-5,
        'n_factors': 200,
        'RPE_hidden_size': 200,
        'lamda': 0.001,
        'u': 1.0,
        'beta': 0.001
    }


def create_default_darec_param_grid():
    return {
        'epochs': [20],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400],
        'RPE_hidden_size': [200, 400],
        'beta': [0.001, 0.01]
    }


if __name__ == "__main__":
    # Define domain pairs 
    base_data_dir = "../../data" 
    domain_pairs = [
        (f"{base_data_dir}/ratings_Amazon_Instant_Video.csv", f"{base_data_dir}/ratings_Apps_for_Android.csv"),
        (f"{base_data_dir}/ratings_Amazon_Instant_Video.csv", f"{base_data_dir}/ratings_Beauty.csv"),
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