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
    S_n_items, S_n_users = train_dataset.S_data.shape[0], train_dataset.S_data.shape[1]
    T_n_items, T_n_users = train_dataset.T_data.shape[0], train_dataset.T_data.shape[1]

    # print number of items and users
    print(f"Source domain ({source_name}): {S_n_items} items, {S_n_users} users")
    print(f"Target domain ({target_name}): {T_n_items} items, {T_n_users} users")

    # Create models for both source and target
    S_model = I_AutoRec(n_users=S_n_users, n_items=S_n_items, n_factors=config['n_factors']).cuda()
    T_model = I_AutoRec(n_users=T_n_users, n_items=T_n_items, n_factors=config['n_factors']).cuda()
    
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
        
        # Get OT configuration
        enable_ot = config.get('enable_ot', False)
        gw_weight = config.get('gw_weight', 0.1)
        w_weight = config.get('w_weight', 0.1)
        
        for idx, (S_data, T_data, S_y, T_y) in enumerate(train_loader):
            S_data = S_data.cuda()
            T_data = T_data.cuda()
            
            # Train source model
            S_optimizer.zero_grad()
            if enable_ot:
                _, S_pred, S_ot_loss = S_model(S_data, other_domain_x=T_data, enable_ot=True, 
                                              gw_weight=gw_weight, w_weight=w_weight)
            else:
                result = S_model(S_data)
                if len(result) == 3:  # Handle both old and new return formats
                    _, S_pred, S_ot_loss = result
                else:
                    _, S_pred = result
                    S_ot_loss = None
            
            S_pred.cuda()
            S_loss, S_mask = criterion(S_pred, S_data)
            
            # Add OT loss if enabled and available
            if enable_ot and S_ot_loss is not None:
                S_loss = S_loss + S_ot_loss
            
            S_Total_RMSE += S_loss.item()
            S_Total_MASK += torch.sum(S_mask).item()
            S_loss.backward()
            S_optimizer.step()
            
            # Train target model
            T_optimizer.zero_grad()
            if enable_ot:
                _, T_pred, T_ot_loss = T_model(T_data, other_domain_x=S_data, enable_ot=True,
                                              gw_weight=gw_weight, w_weight=w_weight)
            else:
                result = T_model(T_data)
                if len(result) == 3:  # Handle both old and new return formats
                    _, T_pred, T_ot_loss = result
                else:
                    _, T_pred = result
                    T_ot_loss = None
            
            T_pred.cuda()
            T_loss, T_mask = criterion(T_pred, T_data)
            
            # Add OT loss if enabled and available
            if enable_ot and T_ot_loss is not None:
                T_loss = T_loss + T_ot_loss
            
            T_Total_RMSE += T_loss.item()
            T_Total_MASK += torch.sum(T_mask).item()
            T_loss.backward()
            T_optimizer.step()

        S_rmse = math.sqrt(S_Total_RMSE / S_Total_MASK)
        T_rmse = math.sqrt(T_Total_RMSE / T_Total_MASK)
        return S_rmse, T_rmse

    def test_epoch():
        S_model.eval()
        T_model.eval()
        S_Total_RMSE = 0
        S_Total_MASK = 0
        T_Total_RMSE = 0
        T_Total_MASK = 0
        
        with torch.no_grad():
            for idx, (S_data, T_data, S_y, T_y) in enumerate(test_loader):
                S_data = S_data.cuda()
                T_data = T_data.cuda()
                
                # Test source model
                result = S_model(S_data)
                if len(result) == 3:  # Handle both old and new return formats
                    _, S_pred, _ = result
                else:
                    _, S_pred = result
                
                S_pred.cuda()
                S_loss, S_mask = criterion(S_pred, S_data)
                S_Total_RMSE += S_loss.item()
                S_Total_MASK += torch.sum(S_mask).item()
                
                # Test target model
                result_t = T_model(T_data)
                if len(result_t) == 3:  # Handle both old and new return formats
                    _, T_pred, _ = result_t
                else:
                    _, T_pred = result_t
                
                T_pred.cuda()
                T_loss, T_mask = criterion(T_pred, T_data)
                T_Total_RMSE += T_loss.item()
                T_Total_MASK += torch.sum(T_mask).item()

        S_rmse = math.sqrt(S_Total_RMSE / S_Total_MASK)
        T_rmse = math.sqrt(T_Total_RMSE / T_Total_MASK)
        return S_rmse, T_rmse

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
    
    for epoch in tqdm(range(config['epochs']), desc=f"Training {source_name}->{target_name}"):
        S_tr_rmse, T_tr_rmse = train_epoch(epoch)
        S_te_rmse, T_te_rmse = test_epoch()
        
        S_train_rmse.append(S_tr_rmse)
        S_test_rmse.append(S_te_rmse)
        T_train_rmse.append(T_tr_rmse)
        T_test_rmse.append(T_te_rmse)
        
        # Log epoch results
        epoch_log = {
            'epoch': epoch + 1,
            'source_train_rmse': S_tr_rmse,
            'source_test_rmse': S_te_rmse,
            'target_train_rmse': T_tr_rmse,
            'target_test_rmse': T_te_rmse
        }
        training_log['epochs'].append(epoch_log)
        
        # Save models at final epoch
        if epoch == config['epochs'] - 1:
            S_model_path = os.path.join(output_dir, f"S_AutoRec_{source_name}_{target_name}_{epoch+1}_{timestamp}.pkl")
            T_model_path = os.path.join(output_dir, f"T_AutoRec_{source_name}_{target_name}_{epoch+1}_{timestamp}.pkl")
            torch.save(S_model.state_dict(), S_model_path)
            torch.save(T_model.state_dict(), T_model_path)
    
    # Save training log
    with open(log_path, 'w') as f:
        json.dump(convert_to_json_serializable(training_log), f, indent=2)
    
    print(f"Source best test RMSE: {min(S_test_rmse):.4f}")
    print(f"Target best test RMSE: {min(T_test_rmse):.4f}")
    print(f"Training log saved to: {log_path}")
    
    return {
        'source_best_rmse': min(S_test_rmse),
        'target_best_rmse': min(T_test_rmse),
        'log_path': log_path,
        'source_model_path': S_model_path,
        'target_model_path': T_model_path
    }


def grid_search_hyperparameters(source_domain_path, target_domain_path, param_grid, output_dir="./results", log_dir="./logs"):
    """
    Perform grid search over hyperparameters.
    
    Args:
        source_domain_path: Path to source domain data CSV
        target_domain_path: Path to target domain data CSV
        param_grid: Dictionary with parameter names as keys and lists of values to try
        output_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
    
    Example param_grid:
    {
        'epochs': [50, 100],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400]
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
    
    print(f"Starting grid search with {len(param_combinations)} combinations for {source_name} -> {target_name}")
    
    for i, param_combo in enumerate(param_combinations):
        config = dict(zip(param_names, param_combo))
        print(f"\nGrid search {i+1}/{len(param_combinations)}: {config}")
        
        try:
            result = train_autoencoders(
                source_domain_path, 
                target_domain_path, 
                config, 
                output_dir, 
                log_dir
            )
            result['config'] = config
            result['combination_id'] = i + 1
            results.append(result)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error with configuration {config}: {e}")
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
    
    grid_search_path = os.path.join(log_dir, f"grid_search_{source_name}_to_{target_name}_{timestamp}.json")
    with open(grid_search_path, 'w') as f:
        json.dump(convert_to_json_serializable(grid_search_log), f, indent=2)
    
    # Find best configurations
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_source = min(valid_results, key=lambda x: x['source_best_rmse'])
        best_target = min(valid_results, key=lambda x: x['target_best_rmse'])
        
        print(f"\nGrid search completed!")
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
        print("All configurations failed!")
        return {'all_results': results, 'grid_search_path': grid_search_path}


def run_multi_domain_experiments(domain_pairs, config=None, param_grid=None, output_dir="./results", log_dir="./logs"):
    """
    Run experiments across multiple domain pairs.
    
    Args:
        domain_pairs: List of tuples (source_path, target_path) for domain combinations
        config: Single configuration dict for simple training (mutually exclusive with param_grid)
        param_grid: Parameter grid for grid search (mutually exclusive with config)
        output_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
    """
    if config is not None and param_grid is not None:
        raise ValueError("Provide either config or param_grid, not both")
    if config is None and param_grid is None:
        raise ValueError("Must provide either config or param_grid")
    
    all_results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, (source_path, target_path) in enumerate(domain_pairs):
        print(f"\n{'='*60}")
        print(f"Domain pair {i+1}/{len(domain_pairs)}")
        print(f"Source: {source_path}")
        print(f"Target: {target_path}")
        print(f"{'='*60}")
        
        try:
            if config is not None:
                # Single configuration training
                result = train_autoencoders(source_path, target_path, config, output_dir, log_dir)
                result['domain_pair'] = (source_path, target_path)
                all_results.append(result)
            else:
                # Grid search
                result = grid_search_hyperparameters(source_path, target_path, param_grid, output_dir, log_dir)
                result['domain_pair'] = (source_path, target_path)
                all_results.append(result)
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error processing domain pair {source_path} -> {target_path}: {e}")
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
    
    summary_path = os.path.join(log_dir, f"multi_domain_experiment_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_to_json_serializable(experiment_summary), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Multi-domain experiment completed!")
    print(f"Processed {len(domain_pairs)} domain pairs")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}")
    
    return {
        'experiment_summary': experiment_summary,
        'summary_path': summary_path
    }


# Example usage functions
def create_default_config():
    """Create a default configuration dictionary."""
    return {
        'epochs': 50,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200,
        'enable_ot': False,     # Enable full optimal transport loss (GW + W distance)
        'gw_weight': 0.1,       # Weight for GW distance component
        'w_weight': 0.1         # Weight for Wasserstein distance component
    }


def create_ot_config():
    """Create a configuration with full optimal transport loss enabled."""
    return {
        'epochs': 50,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200,
        'enable_ot': True,      # Enable full optimal transport loss (GW + W distance)
        'gw_weight': 0.1,       # Weight for GW distance component
        'w_weight': 0.1         # Weight for Wasserstein distance component
    }


def create_gw_config():
    """Create a configuration with only GW loss enabled."""
    return {
        'epochs': 50,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200,
        'enable_ot': True,      # Enable optimal transport loss
        'gw_weight': 0.1,       # Weight for GW distance component
        'w_weight': 0.0         # No Wasserstein distance component
    }


def create_w_config():
    """Create a configuration with only Wasserstein distance enabled."""
    return {
        'epochs': 50,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200,
        'enable_ot': True,      # Enable optimal transport loss
        'gw_weight': 0.0,       # No GW distance component
        'w_weight': 0.1         # Weight for Wasserstein distance component
    }

def create_default_param_grid():
    """Create a default parameter grid for grid search."""
    return {
        'epochs': [50],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400],
        'enable_ot': [False, True],         # Test both with and without OT loss
        'gw_weight': [0.05, 0.1, 0.2],     # Different GW loss weights
        'w_weight': [0.05, 0.1, 0.2]       # Different W distance weights
    }

if __name__ == "__main__":
    
    # Simple example usage - for full experiments, use run_cross_domain_experiments.py
    base_data_dir = "../../data"
    
    # Example: Train on a few domain pairs
    domain_pairs = [
        (f"{base_data_dir}/ratings_Apps_for_Android.csv", f"{base_data_dir}/ratings_Video_Games.csv"),
        (f"{base_data_dir}/ratings_Toys_and_Games.csv", f"{base_data_dir}/ratings_Automotive.csv"),
    ]
    
    # Single configuration training options:
    
    # Option 1: Default config (no OT loss)
    # config = create_default_config()
    
    # Option 2: Full OT loss (GW + W distance)
    # config = create_ot_config()
    
    # Option 3: Only GW loss
    # config = create_gw_config()
    
    # Option 4: Only Wasserstein distance
    # config = create_w_config()
    
    # Current: Use GW only configuration
    config = create_gw_config()
    results = run_multi_domain_experiments(
        domain_pairs=domain_pairs,
        config=config,
        output_dir="./ot_models",
        log_dir="./ot_logs"
    )
    
    # Option: Grid search (comment out the above and uncomment below)
    # param_grid = create_default_param_grid()
    # results = run_multi_domain_experiments(
    #     domain_pairs=domain_pairs,
    #     param_grid=param_grid,
    #     output_dir="./models",
    #     log_dir="./logs"
    # )
    #     log_dir="./logs"
    # )
