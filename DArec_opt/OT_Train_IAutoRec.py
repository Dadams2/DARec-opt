
import torch
from torch import nn, optim
from DArec_opt.OT_IAutoRec import OT_AutoRec
from Data_Preprocessing import Mydata
from function import MRMSELoss
from torch.utils.data import DataLoader
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

def train_ot_autorec(source_domain_path, target_domain_path, config, output_dir="./results", log_dir="./logs"):
    """
    Train both source and target OT-AutoRec models simultaneously.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    log_filename = f"ot_training_log_{source_name}_to_{target_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)

    train_dataset = Mydata(source_domain_path, target_domain_path, train=True, preprocessed=True)
    test_dataset = Mydata(source_domain_path, target_domain_path, train=False, preprocessed=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Data loaded for {source_name} -> {target_name}")

    S_n_items, S_n_users = train_dataset.S_data.shape[0], train_dataset.S_data.shape[1]
    T_n_items, T_n_users = train_dataset.T_data.shape[0], train_dataset.T_data.shape[1]

    source_model = OT_AutoRec(n_users=S_n_users, n_items=S_n_items, n_factors=config['n_factors'], ot_weight=config['ot_weight']).cuda()
    target_model = OT_AutoRec(n_users=T_n_users, n_items=T_n_items, n_factors=config['n_factors'], ot_weight=config['ot_weight']).cuda()

    criterion = MRMSELoss().cuda()
    optimizer = optim.Adam(list(source_model.parameters()) + list(target_model.parameters()), weight_decay=config['wd'], lr=config['lr'])

    def train_epoch(epoch):
        source_model.train()
        target_model.train()
        Total_RMSE = 0
        Total_MASK = 0
        for idx, (S_data, T_data, S_y, T_y) in enumerate(train_loader):
            S_data = S_data.cuda()
            T_data = T_data.cuda()
            optimizer.zero_grad()
            _, pred_s, ot_loss_s = source_model(S_data, x_other=T_data)
            _, pred_t, ot_loss_t = target_model(T_data, x_other=S_data)
            loss_s, mask_s = criterion(pred_s, S_data)
            loss_t, mask_t = criterion(pred_t, T_data)
            ot_losses = []
            if ot_loss_s is not None:
                ot_losses.append(ot_loss_s)
            if ot_loss_t is not None:
                ot_losses.append(ot_loss_t)
            ot_loss = sum(ot_losses) / len(ot_losses) if ot_losses else 0.0
            total_loss = loss_s + loss_t + config['ot_weight'] * ot_loss
            Total_RMSE += loss_s.item() + loss_t.item()
            Total_MASK += torch.sum(mask_s).item() + torch.sum(mask_t).item()
            total_loss.backward()
            optimizer.step()
        return math.sqrt(Total_RMSE / Total_MASK)

    def test_epoch():
        source_model.eval()
        target_model.eval()
        Total_RMSE = 0
        Total_MASK = 0
        with torch.no_grad():
            for idx, (S_data, T_data, S_y, T_y) in enumerate(test_loader):
                S_data = S_data.cuda()
                T_data = T_data.cuda()
                _, pred_s, _ = source_model(S_data)
                _, pred_t, _ = target_model(T_data)
                loss_s, mask_s = criterion(pred_s, S_data)
                loss_t, mask_t = criterion(pred_t, T_data)
                Total_RMSE += loss_s.item() + loss_t.item()
                Total_MASK += torch.sum(mask_s).item() + torch.sum(mask_t).item()
        return math.sqrt(Total_RMSE / Total_MASK)

    training_log = {
        'config': config,
        'source_domain': source_name,
        'target_domain': target_name,
        'epochs': []
    }

    train_rmse = []
    test_rmse = []

    for epoch in tqdm(range(config['epochs']), desc=f"Training {source_name}->{target_name}"):
        tr_rmse = train_epoch(epoch)
        te_rmse = test_epoch()
        train_rmse.append(tr_rmse)
        test_rmse.append(te_rmse)
        epoch_log = {
            'epoch': epoch + 1,
            'train_rmse': tr_rmse,
            'test_rmse': te_rmse
        }
        training_log['epochs'].append(epoch_log)
        if epoch == config['epochs'] - 1:
            S_model_path = os.path.join(output_dir, f"S_OT_AutoRec_{source_name}_{target_name}_{epoch+1}_{timestamp}.pkl")
            T_model_path = os.path.join(output_dir, f"T_OT_AutoRec_{source_name}_{target_name}_{epoch+1}_{timestamp}.pkl")
            torch.save(source_model.state_dict(), S_model_path)
            torch.save(target_model.state_dict(), T_model_path)

    with open(log_path, 'w') as f:
        json.dump(convert_to_json_serializable(training_log), f, indent=2)

    print(f"Source best test RMSE: {min(train_rmse):.4f}")
    print(f"Target best test RMSE: {min(test_rmse):.4f}")
    print(f"Training log saved to: {log_path}")

    return {
        'source_best_rmse': min(train_rmse),
        'target_best_rmse': min(test_rmse),
        'log_path': log_path,
        'source_model_path': S_model_path,
        'target_model_path': T_model_path
    }

def grid_search_ot_hyperparameters(source_domain_path, target_domain_path, param_grid, output_dir="./results", log_dir="./logs"):
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
            result = train_ot_autorec(
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
    grid_search_log = {
        'source_domain': source_name,
        'target_domain': target_name,
        'param_grid': param_grid,
        'results': results,
        'timestamp': timestamp
    }
    grid_search_path = os.path.join(log_dir, f"ot_grid_search_{source_name}_to_{target_name}_{timestamp}.json")
    with open(grid_search_path, 'w') as f:
        json.dump(convert_to_json_serializable(grid_search_log), f, indent=2)
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

def run_multi_domain_ot_experiments(domain_pairs, config=None, param_grid=None, output_dir="./results", log_dir="./logs"):
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
                result = train_ot_autorec(source_path, target_path, config, output_dir, log_dir)
                result['domain_pair'] = (source_path, target_path)
                all_results.append(result)
            else:
                result = grid_search_ot_hyperparameters(source_path, target_path, param_grid, output_dir, log_dir)
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
    experiment_summary = {
        'timestamp': timestamp,
        'domain_pairs': domain_pairs,
        'experiment_type': 'single_config' if config is not None else 'grid_search',
        'config': config,
        'param_grid': param_grid,
        'results': all_results
    }
    summary_path = os.path.join(log_dir, f"multi_domain_ot_experiment_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_to_json_serializable(experiment_summary), f, indent=2)
    print(f"\n{'='*60}")
    print(f"Multi-domain OT-AutoRec experiment completed!")
    print(f"Processed {len(domain_pairs)} domain pairs")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}")
    return {
        'experiment_summary': experiment_summary,
        'summary_path': summary_path
    }

# Example usage functions
def create_default_ot_config():
    return {
        'epochs': 50,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200,
        'ot_weight': 0.1
    }

def create_default_ot_param_grid():
    return {
        'epochs': [50],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400],
        'ot_weight': [0.05, 0.1, 0.2]
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
    config = create_default_ot_config()
    results = run_multi_domain_ot_experiments(
        domain_pairs=domain_pairs,
        config=config,
        output_dir="./ot_models",
        log_dir="./ot_logs"
    )
    # Option 2: Grid search (comment out the above and uncomment below)
    # param_grid = create_default_ot_param_grid()
    # results = run_multi_domain_ot_experiments(
    #     domain_pairs=domain_pairs,
    #     param_grid=param_grid,
    #     output_dir="./ot_models",
    #     log_dir="./ot_logs"
    # )