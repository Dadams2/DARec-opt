"""
DARec Experiment Framework

A flexible framework for running experiments with I-DARec, U-DARec, and their OT variants.
Supports hyperparameter grid search, domain pair testing, and extensible evaluation.
"""

__version__ = "1.0.0"
__author__ = "DARec Research Team"

from .config import (
    ExperimentConfig, MethodConfig, DataConfig, EvaluationConfig,
    get_i_darec_config, get_u_darec_config, get_autorec_config,
    get_quick_test_config, get_full_grid_search_config
)
from .methods import MethodRegistry, BaseMethod
from .runner import ExperimentRunner
from .evaluation import EvaluationFramework
from .utils import DatasetDiscovery, ResultManager

__all__ = [
    'ExperimentConfig', 'MethodConfig', 'DataConfig', 'EvaluationConfig',
    'get_i_darec_config', 'get_u_darec_config', 'get_autorec_config',
    'get_quick_test_config', 'get_full_grid_search_config',
    'MethodRegistry', 'BaseMethod',
    'ExperimentRunner',
    'EvaluationFramework',
    'DatasetDiscovery', 'ResultManager'
]
