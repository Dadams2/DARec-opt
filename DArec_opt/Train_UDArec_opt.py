import numpy as np
import torch.optim as optim
import torch.utils.data
from U_DArec_opt import *
from torch.utils.data import DataLoader
from Data_Preprocessing import Mydata
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
import os
import datetime
import itertools
import json
import traceback

def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):
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

def train_udarec_opt(source_domain_path, target_domain_path, config, output_dir="./results", log_dir="./logs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    log_filename = f"udarec_training_log_{source_name}_to_{target_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    train_dataset = Mydata(source_domain_path, target_domain_path, train=True, preprocessed=True)
    test_dataset = Mydata(source_domain_path, target_domain_path, train=False, preprocessed=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Note: For U_DArec, the data dimensions are transposed compared to I_DArec
    # U_DArec processes user vectors (items as features), while I_DArec processes item vectors (users as features)
    config['n_users'] = train_dataset.S_data.shape[0]  # Number of users
    config['S_n_items'], config['T_n_items'] = train_dataset.S_data.shape[1], train_dataset.T_data.shape[1]  # Number of items
    
    print(f"Data loaded for {source_name} -> {target_name}")
    print(f"Users: {config['n_users']}, Source items: {config['S_n_items']}, Target items: {config['T_n_items']}")
    
    # Create args for U_DArec
    args = create_args(
        n_users=config['n_users'],
        S_n_items=config['S_n_items'],
        T_n_items=config['T_n_items'],
        n_factors=config['n_factors'],
        RPE_hidden_size=config['RPE_hidden_size']
    )
    
    if 'S_pretrained_weights' in config and config['S_pretrained_weights']:
        args.S_pretrained_weights = config['S_pretrained_weights']
    if 'T_pretrained_weights' in config and config['T_pretrained_weights']:
        args.T_pretrained_weights = config['T_pretrained_weights']
    
    net = U_DArec(args)
    
    # Load pretrained weights if available
    if hasattr(args, 'S_pretrained_weights') and args.S_pretrained_weights:
        net.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
    if hasattr(args, 'T_pretrained_weights') and args.T_pretrained_weights:
        net.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
    
    net.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=config['wd'], lr=config['lr'])
    RMSE = MRMSELoss().cuda()
    criterion = DArec_Loss().cuda()
    
    def train_epoch(epoch):
        Total_RMSE = 0
        Total_MASK = 0
        gw_weight = config.get('gw_weight', 0.1)
        
        for idx, d in enumerate(train_loader):
            source_rating, target_rating, source_labels, target_labels = d
            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()
            target_labels = target_labels.squeeze(1).long().cuda()
            
            optimizer.zero_grad()
            
            # Forward pass for source data
            class_output_s, source_prediction_s, target_prediction_s, gw_loss_s = net(
                source_rating, True, source_rating_matrix=source_rating, target_rating_matrix=target_rating)
            source_loss, source_mask, target_mask = criterion(class_output_s, source_prediction_s, target_prediction_s,
                                                            source_rating, target_rating, source_labels)
            rmse, _ = RMSE(source_prediction_s, source_rating)
            Total_RMSE += rmse.item()
            Total_MASK += torch.sum(target_mask).item()
            
            # Forward pass for target data
            class_output_t, source_prediction_t, target_prediction_t, gw_loss_t = net(
                target_rating, False, source_rating_matrix=source_rating, target_rating_matrix=target_rating)
            target_loss, source_mask, target_mask = criterion(class_output_t, source_prediction_t, target_prediction_t,
                                                              source_rating, target_rating, target_labels)
            
            # Combine losses
            loss = source_loss + target_loss
            
            # Add GW losses if available
            gw_losses = []
            if gw_loss_s is not None:
                gw_losses.append(gw_loss_s)
            if gw_loss_t is not None:
                gw_losses.append(gw_loss_t)
            if gw_losses:
                loss = loss + gw_weight * sum(gw_losses) / len(gw_losses)
            
            loss.backward()
            optimizer.step()
        
        return math.sqrt(Total_RMSE / Total_MASK)
    
    def test_epoch():
        Total_RMSE = 0
        Total_MASK = 0
        
        with torch.no_grad():
            for idx, d in enumerate(test_loader):
                source_rating, target_rating, source_labels, target_labels = d
                source_rating = source_rating.cuda()
                target_rating = target_rating.cuda()
                source_labels = source_labels.squeeze(1).long().cuda()
                target_labels = target_labels.squeeze(1).long().cuda()
                
                # Test on source data
                class_output_s, source_prediction_s, target_prediction_s, _ = net(
                    source_rating, True, source_rating_matrix=source_rating, target_rating_matrix=target_rating)
                source_loss, source_mask, target_mask = criterion(class_output_s, source_prediction_s, target_prediction_s,
                                                                source_rating, target_rating, source_labels)
                rmse, _ = RMSE(source_prediction_s, source_rating)
                Total_RMSE += rmse.item()
                Total_MASK += torch.sum(target_mask).item()
                
                # Test on target data
                class_output_t, source_prediction_t, target_prediction_t, _ = net(
                    target_rating, False, source_rating_matrix=source_rating, target_rating_matrix=target_rating)
                target_loss, source_mask, target_mask = criterion(class_output_t, source_prediction_t, target_prediction_t,
                                                                  source_rating, target_rating, target_labels)
        
        return math.sqrt(Total_RMSE / Total_MASK)
    
    # Training loop
    training_log = {
        'config': config,
        'source_domain': source_name,
        'target_domain': target_name,
        'epochs': []
    }
    
    train_rmse_history = []
    test_rmse_history = []
    
    for epoch in tqdm(range(config['epochs'])):
        try:
            train_rmse = train_epoch(epoch)
            test_rmse = test_epoch()
            
            train_rmse_history.append(train_rmse)
            test_rmse_history.append(test_rmse)
            
            # Log epoch results
            epoch_log = {
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'timestamp': datetime.datetime.now().isoformat()
            }
            training_log['epochs'].append(epoch_log)
            
            print(f"Epoch {epoch+1}/{config['epochs']}: Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {str(e)}")
            traceback.print_exc()
            epoch_log = {
                'epoch': epoch + 1,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            training_log['epochs'].append(epoch_log)
    
    # Save final model
    if train_rmse_history:
        model_path = os.path.join(output_dir, f"UDArec_opt_{source_name}_{target_name}_{timestamp}.pkl")
        torch.save(net.state_dict(), model_path)
        training_log['model_path'] = model_path
        training_log['final_train_rmse'] = train_rmse_history[-1]
        training_log['final_test_rmse'] = test_rmse_history[-1]
        training_log['best_test_rmse'] = min(test_rmse_history)
    
    # Save training log
    with open(log_path, 'w') as f:
        json.dump(convert_to_json_serializable(training_log), f, indent=2)
    
    print(f"Training completed!")
    if test_rmse_history:
        print(f"Best test RMSE: {min(test_rmse_history):.4f}")
    print(f"Training log saved to: {log_path}")
    
    return {
        'train_rmse_history': train_rmse_history,
        'test_rmse_history': test_rmse_history,
        'best_test_rmse': min(test_rmse_history) if test_rmse_history else float('inf'),
        'log_path': log_path,
        'model_path': model_path if 'model_path' in locals() else None
    }

def grid_search_udarec_opt(source_domain_path, target_domain_path, param_grid, output_dir="./results", log_dir="./logs"):
    """
    Perform grid search over hyperparameters for U_DArec_opt.
    
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
    
    print(f"Starting U_DArec_opt grid search with {len(param_combinations)} parameter combinations")
    print(f"Parameters: {param_names}")
    
    results = []
    for i, param_combo in enumerate(param_combinations):
        config = dict(zip(param_names, param_combo))
        print(f"\nConfiguration {i+1}/{len(param_combinations)}: {config}")
        
        try:
            result = train_udarec_opt(source_domain_path, target_domain_path, config, output_dir, log_dir)
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
    
    grid_search_path = os.path.join(log_dir, f"udarec_opt_grid_search_{source_name}_to_{target_name}_{timestamp}.json")
    with open(grid_search_path, 'w') as f:
        json.dump(convert_to_json_serializable(grid_search_log), f, indent=2)
    
    # Find best configuration
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_test_rmse'])
        
        print(f"\n{'='*60}")
        print(f"Grid search completed! Results saved to: {grid_search_path}")
        print(f"Best test RMSE: {best_result['best_test_rmse']:.4f} with config: {best_result['config']}")
    else:
        print("No valid configurations completed successfully!")
    
    return results

def create_default_udarec_config():
    return {
        'epochs': 50,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 400,
        'RPE_hidden_size': 200,
        'gw_weight': 0.1
    }

def create_default_udarec_param_grid():
    return {
        'epochs': [30, 50],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400],
        'RPE_hidden_size': [100, 200],
        'gw_weight': [0.05, 0.1, 0.2]
    }

if __name__ == "__main__":
    # Example usage
    base_data_dir = "../../data"
    
    # Example single training
    source_path = f"{base_data_dir}/ratings_Apps_for_Android.csv"
    target_path = f"{base_data_dir}/ratings_Video_Games.csv"
    
    config = create_default_udarec_config()
    
    print("Training U_DArec with optimal transport...")
    result = train_udarec_opt(source_path, target_path, config, "./models", "./logs")
    print(f"Training completed with best RMSE: {result['best_test_rmse']:.4f}")
    
    # Example grid search
    # param_grid = create_default_udarec_param_grid()
    # results = grid_search_udarec_opt(source_path, target_path, param_grid, "./models", "./logs")