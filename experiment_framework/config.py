"""
Configuration system for DARec experiments.
"""
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import itertools


@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    data_dir: str = "/home2/dadams/DARec-opt/data"
    train_ratio: float = 0.9
    test_ratio: float = 0.1
    preprocessed: bool = True
    batch_size: int = 64
    source_domains: Optional[List[str]] = None  # If None, auto-discover
    target_domains: Optional[List[str]] = None  # If None, auto-discover
    max_domain_pairs: Optional[int] = None  # Limit pairs for testing
    
    def __post_init__(self):
        """Auto-discover domains if not specified."""
        if self.source_domains is None or self.target_domains is None:
            from .utils import DatasetDiscovery
            discovery = DatasetDiscovery(self.data_dir)
            if self.source_domains is None:
                self.source_domains = discovery.get_available_domains()
            if self.target_domains is None:
                self.target_domains = discovery.get_available_domains()


@dataclass
class MethodConfig:
    """Configuration for a specific method."""
    name: str
    module_path: str  # e.g., "DArec_opt.I_DArec.Train_DArec"
    class_name: str   # e.g., "I_DArec"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    grid_search_params: Dict[str, List[Any]] = field(default_factory=dict)
    pretrained_required: bool = False
    pretrained_paths: Dict[str, str] = field(default_factory=dict)
    
    def get_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all hyperparameter combinations for grid search."""
        if not self.grid_search_params:
            return [self.hyperparameters.copy()]
        
        # Get parameter names and value lists
        param_names = list(self.grid_search_params.keys())
        param_values = list(self.grid_search_params.values())
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*param_values):
            params = self.hyperparameters.copy()
            params.update(dict(zip(param_names, combo)))
            combinations.append(params)
        
        return combinations


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and analysis."""
    metrics: List[str] = field(default_factory=lambda: ["rmse", "mae"])
    save_predictions: bool = False
    save_embeddings: bool = False
    plot_training_curves: bool = True
    additional_analysis: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Main configuration for experiments."""
    name: str
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    methods: List[MethodConfig] = field(default_factory=list)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "experiment_results"
    random_seed: int = 42
    device: str = "cuda"
    parallel: bool = False
    max_workers: int = 4
    resume: bool = False  # Resume from previous run
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Handle nested configs
        if 'data' in data:
            data['data'] = DataConfig(**data['data'])
        if 'evaluation' in data:
            data['evaluation'] = EvaluationConfig(**data['evaluation'])
        if 'methods' in data:
            data['methods'] = [MethodConfig(**m) for m in data['methods']]
        
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def to_json(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def get_domain_pairs(self) -> List[tuple]:
        """Get all domain pairs to test."""
        pairs = []
        for source in self.data.source_domains:
            for target in self.data.target_domains:
                if source != target:  # Don't test same domain pairs
                    pairs.append((source, target))
        
        if self.data.max_domain_pairs:
            pairs = pairs[:self.data.max_domain_pairs]
        
        return pairs
    
    def get_total_experiments(self) -> int:
        """Calculate total number of experiments to run."""
        total = 0
        domain_pairs = len(self.get_domain_pairs())
        
        for method in self.methods:
            param_combinations = len(method.get_param_combinations())
            total += domain_pairs * param_combinations
        
        return total


# Predefined method configurations
def get_i_darec_config() -> MethodConfig:
    """Get configuration for I-DARec method."""
    return MethodConfig(
        name="I_DARec",
        module_path="DArec_opt.I_DArec.Train_DArec",
        class_name="I_DArec",
        hyperparameters={
            "epochs": 70,
            "batch_size": 64,
            "lr": 1e-3,
            "wd": 1e-4,
            "n_factors": 200,
            "RPE_hidden_size": 200,
        },
        grid_search_params={
            "lr": [1e-4, 1e-3, 1e-2],
            "wd": [1e-5, 1e-4, 1e-3],
            "n_factors": [100, 200, 400],
        },
        pretrained_required=True,
        pretrained_paths={
            "S_pretrained_weights": "Pretrained_ParametersS_AutoRec_50.pkl",
            "T_pretrained_weights": "Pretrained_ParametersT_AutoRec_50.pkl"
        }
    )


def get_u_darec_config() -> MethodConfig:
    """Get configuration for U-DARec method."""
    return MethodConfig(
        name="U_DARec",
        module_path="DArec_opt.U_DArec.Train_DArec",
        class_name="U_DArec",
        hyperparameters={
            "epochs": 20,
            "batch_size": 64,
            "lr": 1e-3,
            "wd": 1e-5,
            "n_factors": 200,
            "RPE_hidden_size": 200,
        },
        grid_search_params={
            "lr": [1e-4, 1e-3, 1e-2],
            "wd": [1e-6, 1e-5, 1e-4],
            "n_factors": [100, 200, 400],
        },
        pretrained_required=True,
        pretrained_paths={
            "S_pretrained_weights": "Pretrained_ParametersS_AutoRec_20.pkl",
            "T_pretrained_weights": "Pretrained_ParametersT_AutoRec_20.pkl"
        }
    )


def get_autorec_config(variant="I") -> MethodConfig:
    """Get configuration for AutoRec baseline."""
    return MethodConfig(
        name=f"{variant}_AutoRec",
        module_path=f"DArec_opt.{variant}_DArec.Train_AutoRec",
        class_name=f"{variant}_AutoRec",
        hyperparameters={
            "epochs": 50 if variant == "I" else 20,
            "batch_size": 64,
            "lr": 1e-3,
            "wd": 1e-4,
            "n_factors": 200,
            "train_S": False,  # Train target by default
        },
        grid_search_params={
            "lr": [1e-4, 1e-3, 1e-2],
            "wd": [1e-5, 1e-4, 1e-3],
            "n_factors": [100, 200, 400],
        }
    )


# Default experiment configurations
def get_quick_test_config() -> ExperimentConfig:
    """Get a quick test configuration with limited scope."""
    data_config = DataConfig(
        max_domain_pairs=2,  # Only test 2 domain pairs
        batch_size=32
    )
    
    methods = [
        MethodConfig(
            name="I_DARec_quick",
            module_path="DArec_opt.I_DArec.Train_DArec",
            class_name="I_DARec",
            hyperparameters={"epochs": 5, "batch_size": 32, "lr": 1e-3},
            grid_search_params={"lr": [1e-3, 1e-2]},  # Only 2 values
        )
    ]
    
    return ExperimentConfig(
        name="quick_test",
        description="Quick test with limited scope",
        data=data_config,
        methods=methods,
        output_dir="quick_test_results"
    )


def get_full_grid_search_config() -> ExperimentConfig:
    """Get full grid search configuration."""
    return ExperimentConfig(
        name="full_grid_search",
        description="Complete grid search across all methods and domains",
        methods=[
            get_i_darec_config(),
            get_u_darec_config(),
            get_autorec_config("I"),
            get_autorec_config("U"),
        ],
        output_dir="full_grid_search_results"
    )
