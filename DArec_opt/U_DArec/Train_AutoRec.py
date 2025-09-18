
from torch import nn, optim
from AutoRec import *
from Data_Preprocessing import Mydata
from function import MRMSELoss
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import datetime
import itertools
import json
import numpy as np
import traceback

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

def train_autoencoders(source_domain_path, target_domain_path, config, output_dir="./results", log_dir="./logs"):
    """
    Train both source and target autoencoders simultaneously.
    
    Args:
        source_domain_path: Path to source domain data CSV
        target_domain_path: Path to target domain data CSV  
        config: Dictionary with training parameters (epochs, batch_size, lr, wd, n_factors)
        output_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    log_filename = f"training_log_{source_name}_to_{target_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    # Load datasets
    train_dataset = Mydata(source_domain_path, target_domain_path, train=True, preprocessed=True)
    test_dataset = Mydata(source_domain_path, target_domain_path, train=False, preprocessed=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Data loaded for {source_name} -> {target_name}")
    
    # Get dimensions for both source and target
    S_n_users, S_n_items = train_dataset.S_data.shape[0], train_dataset.S_data.shape[1]
    T_n_users, T_n_items = train_dataset.T_data.shape[0], train_dataset.T_data.shape[1]

    # Create models for both source and target
    S_model = U_AutoRec(n_users=S_n_users, n_items=S_n_items, n_factors=config['n_factors']).cuda()
    T_model = U_AutoRec(n_users=T_n_users, n_items=T_n_items, n_factors=config['n_factors']).cuda()
    
    criterion = MRMSELoss().cuda()

    # Create optimizers for both models
    S_optimizer = optim.Adam(S_model.parameters(), weight_decay=config['wd'], lr=config['lr'])
    T_optimizer = optim.Adam(T_model.parameters(), weight_decay=config['wd'], lr=config['lr'])

    def train_epoch(epoch):
        S_model.train()
        T_model.train()
        S_Total_RMSE = 0
        S_Total_MASK = 0
        T_Total_RMSE = 0
        T_Total_MASK = 0
        
        for idx, d in enumerate(train_loader):
            S_data, T_data = d[0].cuda(), d[1].cuda()
            
            # Train source model
            S_optimizer.zero_grad()
            _, S_pred = S_model(S_data)
            S_loss, S_mask = criterion(S_pred, S_data)
            S_Total_RMSE += S_loss.item()
            S_Total_MASK += torch.sum(S_mask).item()
            S_loss.backward()
            S_optimizer.step()
            
            # Train target model
            T_optimizer.zero_grad()
            _, T_pred = T_model(T_data)
            T_loss, T_mask = criterion(T_pred, T_data)
            T_Total_RMSE += T_loss.item()
            T_Total_MASK += torch.sum(T_mask).item()
            T_loss.backward()
            T_optimizer.step()

        S_train_rmse = math.sqrt(S_Total_RMSE / S_Total_MASK)
        T_train_rmse = math.sqrt(T_Total_RMSE / T_Total_MASK)
        return S_train_rmse, T_train_rmse

    def test_epoch():
        S_model.eval()
        T_model.eval()
        S_Total_RMSE = 0
        S_Total_MASK = 0
        T_Total_RMSE = 0
        T_Total_MASK = 0
        
        with torch.no_grad():
            for idx, d in enumerate(test_loader):
                S_data, T_data = d[0].cuda(), d[1].cuda()
                
                # Test source model
                _, S_pred = S_model(S_data)
                S_loss, S_mask = criterion(S_pred, S_data)
                S_Total_RMSE += S_loss.item()
                S_Total_MASK += torch.sum(S_mask).item()
                
                # Test target model
                _, T_pred = T_model(T_data)
                T_loss, T_mask = criterion(T_pred, T_data)
                T_Total_RMSE += T_loss.item()
                T_Total_MASK += torch.sum(T_mask).item()

        S_test_rmse = math.sqrt(S_Total_RMSE / S_Total_MASK)
        T_test_rmse = math.sqrt(T_Total_RMSE / T_Total_MASK)
        return S_test_rmse, T_test_rmse

    # Training loop
    training_log = {
        'config': config,
        'source_domain': source_name,
        'target_domain': target_name,
        'epochs': []
    }
    
    S_train_rmse = []
    S_test_rmse = []
    T_train_rmse = []
    T_test_rmse = []
    
    for epoch in tqdm(range(config['epochs'])):
        S_train, T_train = train_epoch(epoch)
        S_test, T_test = test_epoch()
        
        S_train_rmse.append(S_train)
        S_test_rmse.append(S_test)
        T_train_rmse.append(T_train)
        T_test_rmse.append(T_test)
        
        # Log epoch results
        epoch_log = {
            'epoch': epoch + 1,
            'source_train_rmse': S_train,
            'source_test_rmse': S_test,
            'target_train_rmse': T_train,
            'target_test_rmse': T_test,
            'timestamp': datetime.datetime.now().isoformat()
        }
        training_log['epochs'].append(epoch_log)
        
        # Save models at final epoch
        if epoch == config['epochs'] - 1:
            S_model_path = os.path.join(output_dir, f"S_AutoRec_{source_name}_{target_name}_{timestamp}_{epoch+1}.pkl")
            T_model_path = os.path.join(output_dir, f"T_AutoRec_{source_name}_{target_name}_{timestamp}_{epoch+1}.pkl")
            torch.save(S_model.state_dict(), S_model_path)
            torch.save(T_model.state_dict(), T_model_path)
    
    # Save training log
    with open(log_path, 'w') as f:
        json.dump(convert_to_json_serializable(training_log), f, indent=2)
    
    print(f"Source best test RMSE: {min(S_test_rmse):.4f}")
    print(f"Target best test RMSE: {min(T_test_rmse):.4f}")
    print(f"Training log saved to: {log_path}")
    
    return {
        'source_train_rmse': S_train_rmse,
        'source_test_rmse': S_test_rmse,
        'target_train_rmse': T_train_rmse,
        'target_test_rmse': T_test_rmse,
        'best_source_rmse': min(S_test_rmse),
        'best_target_rmse': min(T_test_rmse),
        'log_path': log_path
    }

def grid_search_hyperparameters(source_domain_path, target_domain_path, param_grid, output_dir="./results", log_dir="./logs"):
    """
    Perform grid search over hyperparameters.
    
    Args:
        source_domain_path: Path to source domain data CSV
        target_domain_path: Path to target domain data CSV
        param_grid: Dictionary with parameter names as keys and lists of values to try
        output_dir: Directory to save models
        log_dir: Directory to save logs
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    
    print(f"Starting grid search with {len(param_combinations)} parameter combinations")
    print(f"Parameters: {param_names}")
    
    results = []
    for i, param_combo in enumerate(param_combinations):
        config = dict(zip(param_names, param_combo))
        print(f"\nConfiguration {i+1}/{len(param_combinations)}: {config}")
        
        try:
            result = train_autoencoders(source_domain_path, target_domain_path, config, output_dir, log_dir)
            result['config'] = config
            result['config_id'] = i + 1
            results.append(result)
        except Exception as e:
            print(f"Error in configuration {i+1}: {str(e)}")
            traceback.print_exc()
            results.append({
                'config': config,
                'config_id': i + 1,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    # Save grid search results
    grid_search_log = {
        'param_grid': param_grid,
        'total_combinations': len(param_combinations),
        'results': results,
        'timestamp': timestamp
    }
    
    grid_search_path = os.path.join(log_dir, f"grid_search_{source_name}_to_{target_name}_{timestamp}.json")
    with open(grid_search_path, 'w') as f:
        json.dump(convert_to_json_serializable(grid_search_log), f, indent=2)
    
    # Find best configurations
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_source = min(valid_results, key=lambda x: x['best_source_rmse'])
        best_target = min(valid_results, key=lambda x: x['best_target_rmse'])
        
        print(f"\n{'='*60}")
        print(f"Grid search completed! Results saved to: {grid_search_path}")
        print(f"Best source RMSE: {best_source['best_source_rmse']:.4f} with config: {best_source['config']}")
        print(f"Best target RMSE: {best_target['best_target_rmse']:.4f} with config: {best_target['config']}")
    else:
        print("No valid configurations completed successfully!")
    
    return results

def run_multi_domain_experiments(domain_pairs, config=None, param_grid=None, output_dir="./results", log_dir="./logs"):
    """
    Run experiments across multiple domain pairs.
    
    Args:
        domain_pairs: List of tuples (source_path, target_path)
        config: Single configuration dict (if not doing grid search)
        param_grid: Parameter grid dict (if doing grid search)
        output_dir: Directory to save models
        log_dir: Directory to save logs
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting multi-domain experiments across {len(domain_pairs)} domain pairs")
    
    all_results = []
    for i, (source_path, target_path) in enumerate(domain_pairs):
        source_name = os.path.basename(source_path).replace('.csv', '').replace('ratings_', '')
        target_name = os.path.basename(target_path).replace('.csv', '').replace('ratings_', '')
        
        print(f"\n{'='*60}")
        print(f"Domain pair {i+1}/{len(domain_pairs)}: {source_name} -> {target_name}")
        print(f"{'='*60}")
        
        try:
            if param_grid is not None:
                # Grid search for this domain pair
                results = grid_search_hyperparameters(source_path, target_path, param_grid, output_dir, log_dir)
            else:
                # Single configuration for this domain pair
                if config is None:
                    config = create_default_config()
                results = train_autoencoders(source_path, target_path, config, output_dir, log_dir)
            
            all_results.append({
                'source_domain': source_name,
                'target_domain': target_name,
                'source_path': source_path,
                'target_path': target_path,
                'results': results,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error in domain pair {source_name} -> {target_name}: {str(e)}")
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
        'experiment_type': 'multi_domain_autorec',
        'total_domain_pairs': len(domain_pairs),
        'config': config if config else None,
        'param_grid': param_grid if param_grid else None,
        'timestamp': timestamp,
        'results': all_results
    }
    
    summary_path = os.path.join(log_dir, f"multi_domain_experiment_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_to_json_serializable(experiment_summary), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Multi-domain experiment completed!")
    print(f"Processed {len(domain_pairs)} domain pairs")
    print(f"Results saved to: {summary_path}")
    
    return all_results

# Example usage functions
def create_default_config():
    return {
        'epochs': 20,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200
    }

def create_default_param_grid():
    return {
        'epochs': [20],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400]
    }

if __name__ == "__main__":
    # Simple example usage - for full experiments, use run_cross_domain_udarec_experiments.py
    base_data_dir = "../../data"
    
    domain_pairs = [
        (f"{base_data_dir}/ratings_Apps_for_Android.csv", f"{base_data_dir}/ratings_Video_Games.csv"),
        (f"{base_data_dir}/ratings_Toys_and_Games.csv", f"{base_data_dir}/ratings_Automotive.csv"),
    ]
    
    # Option 1: Single configuration training
    config = create_default_config()
    results = run_multi_domain_experiments(
        domain_pairs=domain_pairs,
        config=config,
        output_dir="./models",
        log_dir="./logs"
    )
    
