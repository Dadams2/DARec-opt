#!/usr/bin/env python3
"""
Comprehensive Cross-Domain Training Experiment Runner

This script runs comprehensive experiments comparing different combinations of:
- Regular DARec vs OT-DARec
- Regular AutoRec vs OT-AutoRec pretrained models

The four combinations tested are:
1. DARec with Regular AutoRec
2. DARec with OT AutoRec 
3. OT-DARec with Regular AutoRec
4. OT-DARec with OT AutoRec

Usage:
    python run_comprehensive_experiments.py [options]
    
    # Run in background with nohup:
    nohup python run_comprehensive_experiments.py > comprehensive_experiment_output.log 2>&1 &
    
    # Run with custom parameters:
    python run_comprehensive_experiments.py --epochs 100 --batch-size 32 --subset "0,1,2"
"""

import argparse
import sys
import os
import signal
import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import glob
import re
import traceback

# Try to import pandas, but make it optional
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Some table features may be limited.")

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Import training functions
from Train_AutoRec import train_autoencoders, create_default_config as create_autorec_config
from Train_DArec import train_darec, create_default_darec_config

def setup_logging(log_dir="./logs"):
    """Setup logging for the experiment runner."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"comprehensive_experiment_runner_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    logger.info("Current experiment will complete before shutdown.")
    sys.exit(0)

def validate_domain_name(domain_name, data_dir="../../data"):
    """
    Validate that a domain name corresponds to an existing CSV file.
    
    Args:
        domain_name: Name of the domain (e.g., 'Amazon_Instant_Video')
        data_dir: Directory containing the CSV files
    
    Returns:
        True if valid, False otherwise
    """
    csv_path = f"{data_dir}/ratings_{domain_name}.csv"
    return os.path.exists(csv_path)

def get_available_domains(data_dir="../../data"):
    """
    Get list of available domain names from CSV files in data directory.
    
    Args:
        data_dir: Directory containing the CSV files
    
    Returns:
        List of available domain names
    """
    csv_files = glob.glob(os.path.join(data_dir, "ratings_*.csv"))
    domains = []
    for csv_file in csv_files:
        # Extract domain name from filename
        filename = os.path.basename(csv_file)
        if filename.startswith("ratings_") and filename.endswith(".csv"):
            domain_name = filename[8:-4]  # Remove "ratings_" and ".csv"
            domains.append(domain_name)
    return sorted(domains)

def create_domain_pairs(base_data_dir="../../data", subset=None):
    """
    Create all possible domain pairs or a subset.
    
    Args:
        base_data_dir: Base directory containing the CSV files
        subset: Optional list of domain indices to use (e.g., [0, 1, 2] for first 3 domains)
    
    Returns:
        List of (source_path, target_path) tuples
    """
    domains = [
        "Amazon_Instant_Video",
        "Apps_for_Android", 
        "Automotive",
        "Baby",
        "Beauty",
        "Books",
        "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Digital_Music",
        "Electronics",
        "Grocery_and_Gourmet_Food",
        "Health_and_Personal_Care",
        "Home_and_Kitchen",
        "Kindle_Store",
        "Movies_and_TV",
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Sports_and_Outdoors",
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
        "Video_Games"
    ]
    
    # Use subset if specified
    if subset is not None:
        domains = [domains[i] for i in subset if i < len(domains)]
    
    # Generate all possible domain pairs (source -> target)
    domain_pairs = []
    for source_domain in domains:
        for target_domain in domains:
            if source_domain != target_domain:  # Don't pair domain with itself
                source_path = f"{base_data_dir}/ratings_{source_domain}.csv"
                target_path = f"{base_data_dir}/ratings_{target_domain}.csv"
                domain_pairs.append((source_path, target_path))
    
    return domain_pairs, domains

def find_latest_models(models_dir, source_domain_name, target_domain_name, model_type="AutoRec"):
    """
    Find the most recent timestamped models for source and target domains.
    
    Args:
        models_dir: Directory containing saved models
        source_domain_name: Name of source domain (e.g., 'Amazon_Instant_Video')
        target_domain_name: Name of target domain (e.g., 'Beauty')
        model_type: Type of model to find ("AutoRec")
    
    Returns:
        Tuple of (source_model_path, target_model_path) or (None, None) if not found
    """
    # Both regular and OT models use the same naming pattern now
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
        return None, None

def run_autorec_training(domain_pairs, config, models_dir, log_dir, enable_ot=False):
    """
    Run AutoRec training (regular or with OT/GW) for all domain pairs.
    
    Args:
        domain_pairs: List of (source_path, target_path) tuples
        config: Training configuration
        models_dir: Directory to save models
        log_dir: Directory to save logs
        enable_ot: Whether to enable OT/Gromov-Wasserstein functionality
    """
    logger = logging.getLogger(__name__)
    model_type = "OT-enhanced" if enable_ot else "regular"
    logger.info(f"Starting {model_type} AutoRec training for {len(domain_pairs)} domain pairs")
    
    # Create config with OT settings
    autorec_config = config.copy()
    autorec_config['enable_gw'] = enable_ot
    if enable_ot and 'gw_weight' not in autorec_config:
        autorec_config['gw_weight'] = 0.1
    
    results = []
    for i, (source_path, target_path) in enumerate(domain_pairs):
        try:
            source_name = os.path.basename(source_path).replace('.csv', '').replace('ratings_', '')
            target_name = os.path.basename(target_path).replace('.csv', '').replace('ratings_', '')
            logger.info(f"Training {model_type} AutoRec {i+1}/{len(domain_pairs)}: {source_name} -> {target_name}")
            
            result = train_autoencoders(
                source_path, target_path, autorec_config, 
                output_dir=models_dir, log_dir=log_dir
            )
            
            result['domain_pair'] = (source_path, target_path)
            result['enable_ot'] = enable_ot
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error training {model_type} AutoRec for {source_path} -> {target_path}: {e}")
            results.append({
                'domain_pair': (source_path, target_path),
                'enable_ot': enable_ot,
                'error': str(e),
                'error_traceback': traceback.format_exc()
            })
    
    return results

def run_darec_training(domain_pairs, config, models_dir, log_dir, enable_ot=False, autorec_ot=False):
    """
    Run DARec training (regular or with OT/GW) with specified AutoRec models.
    
    Args:
        domain_pairs: List of (source_path, target_path) tuples
        config: Training configuration
        models_dir: Directory to find AutoRec models and save DARec models
        log_dir: Directory to save logs
        enable_ot: Whether to enable OT/GW in DARec training
        autorec_ot: Whether to use OT-enhanced AutoRec models
    """
    logger = logging.getLogger(__name__)
    darec_type = "OT-enhanced" if enable_ot else "regular"
    autorec_type = "OT-enhanced" if autorec_ot else "regular"
    logger.info(f"Starting {darec_type} DARec training with {autorec_type} AutoRec models for {len(domain_pairs)} domain pairs")
    
    # Create config with OT settings
    darec_config = config.copy()
    darec_config['enable_gw'] = enable_ot
    if enable_ot and 'gw_weight' not in darec_config:
        darec_config['gw_weight'] = 0.1
    
    results = []
    for i, (source_path, target_path) in enumerate(domain_pairs):
        try:
            source_name = os.path.basename(source_path).replace('.csv', '').replace('ratings_', '')
            target_name = os.path.basename(target_path).replace('.csv', '').replace('ratings_', '')
            
            logger.info(f"Training {darec_type} DARec {i+1}/{len(domain_pairs)}: {source_name} -> {target_name}")
            
            # Find AutoRec models - search in the appropriate subdirectory
            if autorec_ot:
                autorec_models_search_dir = os.path.join(os.path.dirname(models_dir), "ot_autorec")
            else:
                autorec_models_search_dir = os.path.join(os.path.dirname(models_dir), "regular_autorec")
            
            s_autorec_path, t_autorec_path = find_latest_models(
                autorec_models_search_dir, source_name, target_name, "AutoRec"
            )
            
            if s_autorec_path is None or t_autorec_path is None:
                raise FileNotFoundError(f"Could not find AutoRec models for {source_name} -> {target_name}")
            
            logger.info(f"Using {autorec_type} AutoRec models:")
            logger.info(f"  Source: {s_autorec_path}")
            logger.info(f"  Target: {t_autorec_path}")
            
            # Create DARec output directory
            exp_name = f"{darec_type}_darec_{autorec_type}_autorec".replace("-", "_").replace(" ", "_")
            darec_output_dir = os.path.join(os.path.dirname(models_dir), exp_name + "_models")
            darec_log_dir = os.path.join(log_dir, exp_name)
            os.makedirs(darec_output_dir, exist_ok=True)
            os.makedirs(darec_log_dir, exist_ok=True)
            
            result = train_darec(
                source_path, target_path, darec_config,
                output_dir=darec_output_dir,
                log_dir=darec_log_dir,
                models_dir=models_dir,
                s_pretrained_path=s_autorec_path,
                t_pretrained_path=t_autorec_path
            )
            
            result['domain_pair'] = (source_path, target_path)
            result['darec_ot'] = enable_ot
            result['autorec_ot'] = autorec_ot
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error training {darec_type} DARec with {autorec_type} AutoRec for {source_path} -> {target_path}: {e}")
            results.append({
                'domain_pair': (source_path, target_path),
                'darec_ot': enable_ot,
                'autorec_ot': autorec_ot,
                'error': str(e),
                'error_traceback': traceback.format_exc()
            })
    
    return results

def parse_training_logs(log_dir, experiment_type):
    """
    Parse training logs from a specific experiment type.
    
    Args:
        log_dir: Directory containing training logs
        experiment_type: Type of experiment (e.g., "regular_autorec", "ot_autorec", etc.)
    
    Returns:
        List of parsed log data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing training logs for {experiment_type}")
    
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    parsed_logs = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Extract key information
            if 'epochs' in log_data and log_data['epochs']:
                last_epoch = log_data['epochs'][-1]
                
                parsed_entry = {
                    'experiment_type': experiment_type,
                    'source_domain': log_data.get('source_domain', 'Unknown'),
                    'target_domain': log_data.get('target_domain', 'Unknown'),
                    'config': log_data.get('config', {}),
                    'log_file': log_file
                }
                
                # Extract metrics based on log structure
                if 'source_train_rmse' in last_epoch:
                    # AutoRec logs
                    parsed_entry.update({
                        'source_train_rmse': last_epoch['source_train_rmse'],
                        'source_test_rmse': last_epoch['source_test_rmse'],
                        'target_train_rmse': last_epoch['target_train_rmse'],
                        'target_test_rmse': last_epoch['target_test_rmse']
                    })
                elif 'train_rmse' in last_epoch:
                    # OT AutoRec logs
                    parsed_entry.update({
                        'train_rmse': last_epoch['train_rmse'],
                        'test_rmse': last_epoch['test_rmse']
                    })
                elif 'source_metrics' in last_epoch:
                    # DARec logs with comprehensive metrics
                    parsed_entry.update({
                        'source_train_rmse': last_epoch.get('source_train_rmse'),
                        'source_test_rmse': last_epoch.get('source_test_rmse'),
                        'target_train_rmse': last_epoch.get('target_train_rmse'),
                        'target_test_rmse': last_epoch.get('target_test_rmse'),
                        'source_metrics': last_epoch.get('source_metrics', {}),
                        'target_metrics': last_epoch.get('target_metrics', {})
                    })
                
                # Extract best RMSE across all epochs
                if 'epochs' in log_data:
                    epochs_data = log_data['epochs']
                    
                    if 'source_test_rmse' in epochs_data[0]:
                        source_test_rmses = [ep['source_test_rmse'] for ep in epochs_data if 'source_test_rmse' in ep]
                        target_test_rmses = [ep['target_test_rmse'] for ep in epochs_data if 'target_test_rmse' in ep]
                        if source_test_rmses:
                            parsed_entry['best_source_test_rmse'] = min(source_test_rmses)
                        if target_test_rmses:
                            parsed_entry['best_target_test_rmse'] = min(target_test_rmses)
                    elif 'test_rmse' in epochs_data[0]:
                        test_rmses = [ep['test_rmse'] for ep in epochs_data if 'test_rmse' in ep]
                        if test_rmses:
                            parsed_entry['best_test_rmse'] = min(test_rmses)
                
                parsed_logs.append(parsed_entry)
                
        except Exception as e:
            logger.warning(f"Could not parse log file {log_file}: {e}")
    
    logger.info(f"Parsed {len(parsed_logs)} log files for {experiment_type}")
    return parsed_logs

def generate_comparison_tables(all_logs, output_dir):
    """
    Generate comparison tables from all experiment logs.
    
    Args:
        all_logs: Dictionary with experiment types as keys and parsed logs as values
        output_dir: Directory to save comparison tables
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating comparison tables")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize data by domain pairs
    domain_pair_results = {}
    
    for exp_type, logs in all_logs.items():
        for log_entry in logs:
            if 'error' in log_entry:
                continue
                
            domain_key = f"{log_entry['source_domain']}_to_{log_entry['target_domain']}"
            
            if domain_key not in domain_pair_results:
                domain_pair_results[domain_key] = {
                    'source_domain': log_entry['source_domain'],
                    'target_domain': log_entry['target_domain']
                }
            
            domain_pair_results[domain_key][exp_type] = log_entry
    
    # Create summary table
    summary_data = []
    for domain_key, results in domain_pair_results.items():
        row = {
            'source_domain': results['source_domain'],
            'target_domain': results['target_domain']
        }
        
        # Extract metrics for each experiment type
        for exp_type in ['regular_autorec', 'ot_autorec', 
                        'regular_darec_regular_autorec', 'regular_darec_ot_autorec',
                        'ot_darec_regular_autorec', 'ot_darec_ot_autorec']:
            if exp_type in results:
                entry = results[exp_type]
                
                if exp_type in ['regular_autorec']:
                    row[f'{exp_type}_source_rmse'] = entry.get('best_source_test_rmse')
                    row[f'{exp_type}_target_rmse'] = entry.get('best_target_test_rmse')
                elif exp_type in ['ot_autorec']:
                    row[f'{exp_type}_rmse'] = entry.get('best_test_rmse')
                else:  # DARec experiments
                    row[f'{exp_type}_source_rmse'] = entry.get('best_source_test_rmse')
                    row[f'{exp_type}_target_rmse'] = entry.get('best_target_test_rmse')
                    
                    # Add comprehensive metrics if available
                    if 'source_metrics' in entry:
                        source_metrics = entry['source_metrics']
                        target_metrics = entry['target_metrics']
                        
                        for metric_name in ['rmse', 'mae', 'ndcg_5', 'precision_5', 'recall_5']:
                            if metric_name in source_metrics:
                                row[f'{exp_type}_source_{metric_name}'] = source_metrics[metric_name]
                            if metric_name in target_metrics:
                                row[f'{exp_type}_target_{metric_name}'] = target_metrics[metric_name]
        
        summary_data.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"experiment_comparison_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed table with all metrics
    detailed_path = os.path.join(output_dir, f"experiment_comparison_detailed_{timestamp}.csv")
    summary_df.to_csv(detailed_path, index=False)
    
    # Generate RMSE comparison table
    rmse_columns = ['source_domain', 'target_domain']
    for exp_type in ['regular_darec_regular_autorec', 'regular_darec_ot_autorec',
                    'ot_darec_regular_autorec', 'ot_darec_ot_autorec']:
        rmse_columns.extend([f'{exp_type}_source_rmse', f'{exp_type}_target_rmse'])
    
    rmse_df = summary_df[rmse_columns] if all(col in summary_df.columns for col in rmse_columns) else summary_df
    rmse_path = os.path.join(output_dir, f"rmse_comparison_{timestamp}.csv")
    rmse_df.to_csv(rmse_path, index=False)
    
    # Calculate average improvements
    improvements = {}
    if 'regular_darec_regular_autorec_target_rmse' in summary_df.columns:
        baseline = 'regular_darec_regular_autorec_target_rmse'
        
        for comparison in ['regular_darec_ot_autorec_target_rmse', 
                          'ot_darec_regular_autorec_target_rmse',
                          'ot_darec_ot_autorec_target_rmse']:
            if comparison in summary_df.columns:
                baseline_vals = summary_df[baseline].dropna()
                comparison_vals = summary_df[comparison].dropna()
                
                if len(baseline_vals) > 0 and len(comparison_vals) > 0:
                    # Calculate relative improvement
                    common_indices = baseline_vals.index.intersection(comparison_vals.index)
                    if len(common_indices) > 0:
                        baseline_subset = baseline_vals[common_indices]
                        comparison_subset = comparison_vals[common_indices]
                        relative_improvement = ((baseline_subset - comparison_subset) / baseline_subset * 100).mean()
                        improvements[comparison] = relative_improvement
    
    # Save improvements summary
    if improvements:
        improvements_df = pd.DataFrame([improvements])
        improvements_path = os.path.join(output_dir, f"improvements_summary_{timestamp}.csv")
        improvements_df.to_csv(improvements_path, index=False)
    
    logger.info(f"Comparison tables saved:")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"  Detailed: {detailed_path}")
    logger.info(f"  RMSE comparison: {rmse_path}")
    if improvements:
        logger.info(f"  Improvements: {improvements_path}")
    
    return {
        'summary_path': summary_path,
        'detailed_path': detailed_path,
        'rmse_path': rmse_path,
        'improvements_path': improvements_path if improvements else None,
        'summary_df': summary_df
    }

def run_comprehensive_experiments(domain_pairs, autorec_config, darec_config, 
                                models_base_dir, logs_base_dir, 
                                skip_autorec=False, skip_darec=False):
    """
    Run all four combinations of experiments.
    
    Args:
        domain_pairs: List of (source_path, target_path) tuples
        autorec_config: Configuration for AutoRec training
        darec_config: Configuration for DARec training
        models_base_dir: Base directory for saving models
        logs_base_dir: Base directory for saving logs
        skip_autorec: Skip AutoRec training (use existing models)
        skip_darec: Skip DARec training (use existing models)
    """
    logger = logging.getLogger(__name__)
    
    all_results = {}
    all_logs = {}
    
    # Step 1: Train AutoRec models (if not skipped)
    if not skip_autorec:
        logger.info("="*60)
        logger.info("STEP 1: Training AutoRec models")
        logger.info("="*60)
        
        # Regular AutoRec
        regular_autorec_models_dir = os.path.join(models_base_dir, "regular_autorec")
        regular_autorec_logs_dir = os.path.join(logs_base_dir, "regular_autorec")
        os.makedirs(regular_autorec_models_dir, exist_ok=True)
        os.makedirs(regular_autorec_logs_dir, exist_ok=True)
        
        regular_autorec_results = run_autorec_training(
            domain_pairs, autorec_config, 
            regular_autorec_models_dir, regular_autorec_logs_dir, 
            enable_ot=False
        )
        all_results['regular_autorec'] = regular_autorec_results
        
        # OT AutoRec
        ot_autorec_models_dir = os.path.join(models_base_dir, "ot_autorec") 
        ot_autorec_logs_dir = os.path.join(logs_base_dir, "ot_autorec")
        os.makedirs(ot_autorec_models_dir, exist_ok=True)
        os.makedirs(ot_autorec_logs_dir, exist_ok=True)
        
        ot_autorec_results = run_autorec_training(
            domain_pairs, autorec_config,
            ot_autorec_models_dir, ot_autorec_logs_dir,
            enable_ot=True
        )
        all_results['ot_autorec'] = ot_autorec_results
    else:
        logger.info("Skipping AutoRec training - using existing models")
    
    # Step 2: Train DARec models (if not skipped)
    if not skip_darec:
        logger.info("="*60)
        logger.info("STEP 2: Training DARec models")
        logger.info("="*60)
        
        # Define model directories for AutoRec models
        regular_autorec_models_dir = os.path.join(models_base_dir, "regular_autorec")
        ot_autorec_models_dir = os.path.join(models_base_dir, "ot_autorec")
        
        # 1. Regular DARec with Regular AutoRec
        logger.info("Training Regular DARec with Regular AutoRec...")
        regular_darec_regular_autorec_results = run_darec_training(
            domain_pairs, darec_config,
            regular_autorec_models_dir, logs_base_dir,
            enable_ot=False, autorec_ot=False
        )
        all_results['regular_darec_regular_autorec'] = regular_darec_regular_autorec_results
        
        # 2. Regular DARec with OT AutoRec
        logger.info("Training Regular DARec with OT AutoRec...")
        regular_darec_ot_autorec_results = run_darec_training(
            domain_pairs, darec_config,
            ot_autorec_models_dir, logs_base_dir,
            enable_ot=False, autorec_ot=True
        )
        all_results['regular_darec_ot_autorec'] = regular_darec_ot_autorec_results
        
        # 3. OT DARec with Regular AutoRec
        logger.info("Training OT DARec with Regular AutoRec...")
        ot_darec_regular_autorec_results = run_darec_training(
            domain_pairs, darec_config,
            regular_autorec_models_dir, logs_base_dir,
            enable_ot=True, autorec_ot=False
        )
        all_results['ot_darec_regular_autorec'] = ot_darec_regular_autorec_results
        
        # 4. OT DARec with OT AutoRec
        logger.info("Training OT DARec with OT AutoRec...")
        ot_darec_ot_autorec_results = run_darec_training(
            domain_pairs, darec_config,
            ot_autorec_models_dir, logs_base_dir,
            enable_ot=True, autorec_ot=True
        )
        all_results['ot_darec_ot_autorec'] = ot_darec_ot_autorec_results
    else:
        logger.info("Skipping DARec training - using existing models")
    
    # Step 3: Parse logs and generate comparison tables
    logger.info("="*60)
    logger.info("STEP 3: Parsing logs and generating comparison tables")
    logger.info("="*60)
    
    # Parse logs from each experiment type
    log_dirs = {
        'regular_autorec': os.path.join(logs_base_dir, "regular_autorec"),
        'ot_autorec': os.path.join(logs_base_dir, "ot_autorec"),
        'regular_darec_regular_autorec': os.path.join(logs_base_dir, "regular_darec_regular_autorec"),
        'regular_darec_ot_autorec': os.path.join(logs_base_dir, "regular_darec_ot_autorec"),
        'ot_darec_regular_autorec': os.path.join(logs_base_dir, "ot_darec_regular_autorec"),
        'ot_darec_ot_autorec': os.path.join(logs_base_dir, "ot_darec_ot_autorec")
    }
    
    for exp_type, log_dir in log_dirs.items():
        if os.path.exists(log_dir):
            all_logs[exp_type] = parse_training_logs(log_dir, exp_type)
        else:
            logger.warning(f"Log directory not found: {log_dir}")
            all_logs[exp_type] = []
    
    # Generate comparison tables
    comparison_output_dir = os.path.join(logs_base_dir, "comparison_tables")
    comparison_results = generate_comparison_tables(all_logs, comparison_output_dir)
    
    return {
        'experiment_results': all_results,
        'parsed_logs': all_logs,
        'comparison_results': comparison_results
    }

def create_configs_from_args(args):
    """Create configuration dictionaries from command line arguments."""
    # AutoRec config
    autorec_config = create_autorec_config()
    if args.epochs:
        autorec_config['epochs'] = args.epochs
    if args.batch_size:
        autorec_config['batch_size'] = args.batch_size
    if args.learning_rate:
        autorec_config['lr'] = args.learning_rate
    if args.weight_decay:
        autorec_config['wd'] = args.weight_decay
    if args.n_factors:
        autorec_config['n_factors'] = args.n_factors
    
    # Add GW weight if not present (both AutoRec and DARec use the same parameter)
    if 'gw_weight' not in autorec_config:
        autorec_config['gw_weight'] = 0.1
    
    # DARec config
    darec_config = create_default_darec_config()
    if args.epochs:
        darec_config['epochs'] = args.epochs
    if args.batch_size:
        darec_config['batch_size'] = args.batch_size
    if args.learning_rate:
        darec_config['lr'] = args.learning_rate
    if args.weight_decay:
        darec_config['wd'] = args.weight_decay
    if args.n_factors:
        darec_config['n_factors'] = args.n_factors
    
    # Add GW weight if not present (needed for OT DARec)
    if 'gw_weight' not in darec_config:
        darec_config['gw_weight'] = 0.1
    
    return autorec_config, darec_config

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive cross-domain training experiments',
        epilog='''
Examples:
  # Single domain pair:
  python run_comprehensive_experiments.py --source-domain Amazon_Instant_Video --target-domain Apps_for_Android
  
  # Multiple domains using subset:
  python run_comprehensive_experiments.py --subset "0,1,2"
  
  # All domain pairs:
  python run_comprehensive_experiments.py
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Experiment parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, help='Weight decay')
    parser.add_argument('--n-factors', type=int, help='Number of latent factors')
    
    # Domain specification
    parser.add_argument('--source-domain', type=str,
                       help='Source domain name (e.g., Amazon_Instant_Video)')
    parser.add_argument('--target-domain', type=str, 
                       help='Target domain name (e.g., Apps_for_Android)')
    
    # Data and output paths
    parser.add_argument('--data-dir', default='../../data', 
                       help='Directory containing CSV files')
    parser.add_argument('--models-dir', default='./comprehensive_models', 
                       help='Base directory to save model checkpoints')
    parser.add_argument('--logs-dir', default='./comprehensive_logs', 
                       help='Base directory to save training logs')
    
    # Domain selection
    parser.add_argument('--subset', type=str, 
                       help='Comma-separated indices of domains to use (e.g., "0,1,2,3")')
    parser.add_argument('--max-pairs', type=int,
                       help='Maximum number of domain pairs to process')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                       help='Print domain pairs without running experiments')
    parser.add_argument('--list-domains', action='store_true',
                       help='List available domains in data directory and exit')
    parser.add_argument('--skip-autorec', action='store_true',
                       help='Skip AutoRec training (use existing models)')
    parser.add_argument('--skip-darec', action='store_true', 
                       help='Skip DARec training (use existing models)')
    parser.add_argument('--resume-from', type=int, default=0,
                       help='Resume from specific domain pair index')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.logs_dir)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting comprehensive cross-domain training experiment runner")
    logger.info(f"Arguments: {vars(args)}")
    
    # Handle list-domains command
    if args.list_domains:
        try:
            available_domains = get_available_domains(args.data_dir)
            logger.info(f"Available domains in {args.data_dir}:")
            for i, domain in enumerate(available_domains):
                logger.info(f"  {i:2d}: {domain}")
            logger.info(f"\nTotal: {len(available_domains)} domains")
            logger.info("\nExample usage:")
            if len(available_domains) >= 2:
                logger.info(f"  python run_comprehensive_experiments.py --source-domain {available_domains[0]} --target-domain {available_domains[1]}")
            return 0
        except Exception as e:
            logger.error(f"Error listing domains: {e}")
            return 1
    
    # Parse subset if provided
    subset = None
    if args.subset:
        try:
            subset = [int(x.strip()) for x in args.subset.split(',')]
            logger.info(f"Using domain subset: {subset}")
        except ValueError:
            logger.error("Invalid subset format. Use comma-separated integers (e.g., '0,1,2')")
            return 1
    
    # Create domain pairs
    try:
        if args.source_domain and args.target_domain:
            # Validate domain names
            available_domains = get_available_domains(args.data_dir)
            
            if not validate_domain_name(args.source_domain, args.data_dir):
                logger.error(f"Source domain '{args.source_domain}' not found in {args.data_dir}")
                logger.error(f"Available domains: {', '.join(available_domains)}")
                return 1
            
            if not validate_domain_name(args.target_domain, args.data_dir):
                logger.error(f"Target domain '{args.target_domain}' not found in {args.data_dir}")
                logger.error(f"Available domains: {', '.join(available_domains)}")
                return 1
            
            if args.source_domain == args.target_domain:
                logger.error("Source and target domains cannot be the same")
                return 1
            
            # Single domain pair specified
            source_path = f"{args.data_dir}/ratings_{args.source_domain}.csv"
            target_path = f"{args.data_dir}/ratings_{args.target_domain}.csv"
            domain_pairs = [(source_path, target_path)]
            domains = [args.source_domain, args.target_domain]
            logger.info(f"Using specified domain pair: {args.source_domain} -> {args.target_domain}")
        elif args.source_domain or args.target_domain:
            logger.error("Both --source-domain and --target-domain must be specified together")
            return 1
        else:
            # Use subset or all domains
            domain_pairs, domains = create_domain_pairs(args.data_dir, subset)
            logger.info(f"Created {len(domain_pairs)} domain pairs from {len(domains)} domains")
            
            # Apply max pairs limit if specified
            if args.max_pairs and args.max_pairs < len(domain_pairs):
                domain_pairs = domain_pairs[:args.max_pairs]
                logger.info(f"Limited to first {args.max_pairs} domain pairs")
            
            # Apply resume offset
            if args.resume_from > 0:
                if args.resume_from >= len(domain_pairs):
                    logger.error(f"Resume index {args.resume_from} >= total pairs {len(domain_pairs)}")
                    return 1
                domain_pairs = domain_pairs[args.resume_from:]
                logger.info(f"Resuming from pair {args.resume_from}, {len(domain_pairs)} pairs remaining")
        
    except Exception as e:
        logger.error(f"Error creating domain pairs: {e}")
        return 1
    
    # Dry run - just print pairs
    if args.dry_run:
        logger.info("DRY RUN - Domain pairs to be processed:")
        for i, (source, target) in enumerate(domain_pairs):
            source_name = os.path.basename(source).replace('ratings_', '').replace('.csv', '')
            target_name = os.path.basename(target).replace('ratings_', '').replace('.csv', '')
            logger.info(f"{i+1:3d}: {source_name} -> {target_name}")
        return 0
    
    # Create directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Create configurations
    autorec_config, darec_config = create_configs_from_args(args)
    logger.info(f"AutoRec configuration: {autorec_config}")
    logger.info(f"DARec configuration: {darec_config}")
    
    # Run comprehensive experiments
    try:
        results = run_comprehensive_experiments(
            domain_pairs=domain_pairs,
            autorec_config=autorec_config,
            darec_config=darec_config,
            models_base_dir=args.models_dir,
            logs_base_dir=args.logs_dir,
            skip_autorec=args.skip_autorec,
            skip_darec=args.skip_darec
        )
        
        logger.info("All comprehensive experiments completed successfully!")
        logger.info(f"Comparison tables saved to: {results['comparison_results']['summary_path']}")
        
        # Print summary of improvements if available
        if 'summary_df' in results['comparison_results']:
            summary_df = results['comparison_results']['summary_df']
            logger.info(f"Processed {len(summary_df)} domain pairs")
            
            # Print average RMSE for each method if available
            for exp_type in ['regular_darec_regular_autorec', 'regular_darec_ot_autorec',
                            'ot_darec_regular_autorec', 'ot_darec_ot_autorec']:
                target_rmse_col = f'{exp_type}_target_rmse'
                if target_rmse_col in summary_df.columns:
                    avg_rmse = summary_df[target_rmse_col].mean()
                    logger.info(f"Average target RMSE for {exp_type}: {avg_rmse:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Experiments interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)