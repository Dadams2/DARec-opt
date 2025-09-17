

import numpy as np
import torch.optim as optim
import torch.utils.data
from I_DArec_opt import *
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

def train_idarec_opt(source_domain_path, target_domain_path, config, output_dir="./results", log_dir="./logs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_domain_path).replace('.csv', '').replace('ratings_', '')
    target_name = os.path.basename(target_domain_path).replace('.csv', '').replace('ratings_', '')
    log_filename = f"idarec_training_log_{source_name}_to_{target_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)
    train_dataset = Mydata(source_domain_path, target_domain_path, train=True, preprocessed=True)
    test_dataset = Mydata(source_domain_path, target_domain_path, train=False, preprocessed=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    config['n_users'] = train_dataset.S_data.shape[1]
    config['S_n_items'], config['T_n_items'] = train_dataset.S_data.shape[0], train_dataset.T_data.shape[0]
    print(f"Data loaded for {source_name} -> {target_name}")
    net = I_DArec(config)
    net.S_autorec.load_state_dict(torch.load(config['S_pretrained_weights']))
    net.T_autorec.load_state_dict(torch.load(config['T_pretrained_weights']))
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
            target_labels= target_labels.squeeze(1).long().cuda()
            optimizer.zero_grad()
            class_output_s, source_prediction_s, target_prediction_s, gw_loss_s = net(
                source_rating, True, source_rating_matrix=source_rating, target_rating_matrix=target_rating)
            source_loss, source_mask, target_mask = criterion(class_output_s, source_prediction_s, target_prediction_s,
                                                    source_rating, target_rating, source_labels)
            rmse, _ = RMSE(source_prediction_s, source_rating)
            Total_RMSE += rmse.item()
            Total_MASK += torch.sum(target_mask).item()
            class_output_t, source_prediction_t, target_prediction_t, gw_loss_t = net(
                target_rating, False, source_rating_matrix=source_rating, target_rating_matrix=target_rating)
            target_loss, source_mask, target_mask = criterion(class_output_t, source_prediction_t, target_prediction_t,
                                                              source_rating, target_rating, target_labels)
            loss = source_loss + target_loss
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
                class_output, source_prediction, target_prediction, _ = net(source_rating, True, source_rating_matrix=source_rating, target_rating_matrix=target_rating)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                        source_rating, target_rating, source_labels)
                rmse, _ = RMSE(source_prediction, source_rating)
                Total_RMSE += rmse.item()
                Total_MASK += torch.sum(target_mask).item()
        return math.sqrt(Total_RMSE / Total_MASK)
    training_log = {
        'config': config,
        'source_domain': source_name,
        'target_domain': target_name,
        'epochs': []
    }
    train_rmse = []
    test_rmse = []
    wdir = r"I-darec"
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
            model_path = os.path.join(output_dir, f"I_DArec_opt_{source_name}_{target_name}_{epoch+1}_{timestamp}.pkl")
            torch.save(net.state_dict(), model_path)
    with open(log_path, 'w') as f:
        json.dump(convert_to_json_serializable(training_log), f, indent=2)
    print(f"Best test RMSE: {min(test_rmse):.4f}")
    print(f"Training log saved to: {log_path}")
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    plt.plot(range(config['epochs']), train_rmse, label='Train RMSE', marker='o', linewidth=2)
    plt.plot(range(config['epochs']), test_rmse, label='Test RMSE', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.title('Training and Test RMSE over Epochs', fontsize=16)
    plt.xticks(range(0, config['epochs'], max(1, config['epochs'] // 20)))
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return {
        'best_rmse': min(test_rmse),
        'log_path': log_path,
        'model_path': model_path
    }

def grid_search_idarec_opt_hyperparameters(source_domain_path, target_domain_path, param_grid, output_dir="./results", log_dir="./logs"):
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
        config['S_pretrained_weights'] = 'Pretrained_ParametersS_AutoRec_50.pkl'
        config['T_pretrained_weights'] = 'Pretrained_ParametersT_AutoRec_50.pkl'
        print(f"\nGrid search {i+1}/{len(param_combinations)}: {config}")
        try:
            result = train_idarec_opt(
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
                'best_rmse': float('inf')
            })
    grid_search_log = {
        'source_domain': source_name,
        'target_domain': target_name,
        'param_grid': param_grid,
        'results': results,
        'timestamp': timestamp
    }
    grid_search_path = os.path.join(log_dir, f"idarec_grid_search_{source_name}_to_{target_name}_{timestamp}.json")
    with open(grid_search_path, 'w') as f:
        json.dump(convert_to_json_serializable(grid_search_log), f, indent=2)
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_rmse'])
        print(f"\nGrid search completed!")
        print(f"Best RMSE: {best_result['best_rmse']:.4f} with config: {best_result['config']}")
        print(f"Grid search results saved to: {grid_search_path}")
        return {
            'best_config': best_result,
            'all_results': results,
            'grid_search_path': grid_search_path
        }
    else:
        print("All configurations failed!")
        return {'all_results': results, 'grid_search_path': grid_search_path}

def run_multi_domain_idarec_opt_experiments(domain_pairs, config=None, param_grid=None, output_dir="./results", log_dir="./logs"):
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
                config['S_pretrained_weights'] = 'Pretrained_ParametersS_AutoRec_50.pkl'
                config['T_pretrained_weights'] = 'Pretrained_ParametersT_AutoRec_50.pkl'
                result = train_idarec_opt(source_path, target_path, config, output_dir, log_dir)
                result['domain_pair'] = (source_path, target_path)
                all_results.append(result)
            else:
                result = grid_search_idarec_opt_hyperparameters(source_path, target_path, param_grid, output_dir, log_dir)
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
    summary_path = os.path.join(log_dir, f"multi_domain_idarec_opt_experiment_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_to_json_serializable(experiment_summary), f, indent=2)
    print(f"\n{'='*60}")
    print(f"Multi-domain I-DArec-opt experiment completed!")
    print(f"Processed {len(domain_pairs)} domain pairs")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}")
    return {
        'experiment_summary': experiment_summary,
        'summary_path': summary_path
    }

# Example usage functions
def create_default_idarec_opt_config():
    return {
        'epochs': 70,
        'batch_size': 64,
        'lr': 1e-3,
        'wd': 1e-4,
        'n_factors': 200,
        'RPE_hidden_size': 200,
        'S_pretrained_weights': 'Pretrained_ParametersS_AutoRec_50.pkl',
        'T_pretrained_weights': 'Pretrained_ParametersT_AutoRec_50.pkl',
        'gw_weight': 0.1
    }

def create_default_idarec_opt_param_grid():
    return {
        'epochs': [70],
        'batch_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'wd': [1e-4, 1e-5],
        'n_factors': [200, 400],
        'RPE_hidden_size': [200],
        'gw_weight': [0.05, 0.1, 0.2]
    }

if __name__ == "__main__":
    base_data_dir = "../../data"
    domain_pairs = [
        (f"{base_data_dir}/ratings_Amazon_Instant_Video.csv", f"{base_data_dir}/ratings_Apps_for_Android.csv"),
        (f"{base_data_dir}/ratings_Amazon_Instant_Video.csv", f"{base_data_dir}/ratings_Beauty.csv"),
        # Add more domain pairs as needed
    ]
    config = create_default_idarec_opt_config()
    results = run_multi_domain_idarec_opt_experiments(
        domain_pairs=domain_pairs,
        config=config,
        output_dir="./idarec_opt_models",
        log_dir="./idarec_opt_logs"
    )
    # Option 2: Grid search (comment out the above and uncomment below)
    # param_grid = create_default_idarec_opt_param_grid()
    # results = run_multi_domain_idarec_opt_experiments(
    #     domain_pairs=domain_pairs,
    #     param_grid=param_grid,
    #     output_dir="./idarec_opt_models",
    #     log_dir="./idarec_opt_logs"
    # )
