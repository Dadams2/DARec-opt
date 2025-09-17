"""
Method registry system for DARec experiments.
"""
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Type
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add project paths to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BaseMethod(ABC):
    """Base class for all DARec methods."""
    
    def __init__(self, config: Dict[str, Any], source_domain: str, target_domain: str):
        self.config = config
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.device = config.get('device', 'cuda')
        self.model = None
        self.training_history = {}
        
    @abstractmethod
    def prepare_data(self, data_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare training and test data loaders."""
        pass
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model."""
        pass
    
    @abstractmethod
    def train(self, train_loader: Any, test_loader: Any) -> Dict[str, List[float]]:
        """Train the model and return training history."""
        pass
    
    @abstractmethod
    def evaluate(self, test_loader: Any) -> Dict[str, float]:
        """Evaluate the model and return metrics."""
        pass
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        if self.model is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))


class IDarecMethod(BaseMethod):
    """I-DARec method implementation."""
    
    def prepare_data(self, data_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare data for I-DARec."""
        from DArec_opt.I_DArec.Data_Preprocessing import Mydata
        from torch.utils.data import DataLoader
        
        # Construct file paths
        data_dir = data_config['data_dir']
        source_path = os.path.join(data_dir, f"ratings_{self.source_domain}.csv")
        target_path = os.path.join(data_dir, f"ratings_{self.target_domain}.csv")
        
        # Create datasets
        train_dataset = Mydata(
            source_path, target_path, 
            train=True, 
            preprocessed=data_config.get('preprocessed', True)
        )
        test_dataset = Mydata(
            source_path, target_path, 
            train=False, 
            preprocessed=data_config.get('preprocessed', True)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=data_config.get('batch_size', 64), 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=data_config.get('batch_size', 64), 
            shuffle=False
        )
        
        # Update config with data dimensions
        self.config['n_users'] = train_dataset.S_data.shape[1]
        self.config['S_n_items'] = train_dataset.S_data.shape[0]
        self.config['T_n_items'] = train_dataset.T_data.shape[0]
        
        return train_loader, test_loader
    
    def build_model(self) -> nn.Module:
        """Build I-DARec model."""
        from DArec_opt.I_DArec.I_DArec import I_DArec
        
        # Create args object
        class Args:
            pass
        
        args = Args()
        for key, value in self.config.items():
            setattr(args, key, value)
        
        model = I_DArec(args)
        
        # Load pretrained weights if specified
        if self.config.get('pretrained_required', False):
            s_weights = self.config.get('S_pretrained_weights')
            t_weights = self.config.get('T_pretrained_weights')
            
            if s_weights and os.path.exists(s_weights):
                model.S_autorec.load_state_dict(torch.load(s_weights, map_location=self.device))
            if t_weights and os.path.exists(t_weights):
                model.T_autorec.load_state_dict(torch.load(t_weights, map_location=self.device))
        
        model = model.to(self.device)
        self.model = model
        return model
    
    def train(self, train_loader: Any, test_loader: Any) -> Dict[str, List[float]]:
        """Train I-DARec model."""
        import torch.optim as optim
        from DArec_opt.I_DArec.function import MRMSELoss, DArec_Loss
        import math
        from tqdm import tqdm
        
        model = self.model
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            weight_decay=self.config.get('wd', 1e-4),
            lr=self.config.get('lr', 1e-3)
        )
        
        RMSE = MRMSELoss().to(self.device)
        criterion = DArec_Loss().to(self.device)
        
        train_rmse = []
        test_rmse = []
        
        def train_epoch():
            model.train()
            total_rmse = 0
            total_mask = 0
            
            for idx, d in enumerate(train_loader):
                source_rating, target_rating, source_labels, target_labels = d
                source_rating = source_rating.to(self.device)
                target_rating = target_rating.to(self.device)
                
                # Create combined labels
                labels = torch.cat([
                    torch.zeros(source_rating.size(0), dtype=torch.long),
                    torch.ones(target_rating.size(0), dtype=torch.long)
                ]).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass for source
                source_class_output, source_pred, _ = model(source_rating, True)
                # Forward pass for target  
                target_class_output, _, target_pred = model(target_rating, False)
                
                # Combine outputs
                class_output = torch.cat([source_class_output, target_class_output])
                
                loss, source_mask, target_mask = criterion(
                    class_output, source_pred, target_pred,
                    source_rating, target_rating, labels
                )
                
                total_rmse += loss.item()
                total_mask += torch.sum(source_mask).item() + torch.sum(target_mask).item()
                
                loss.backward()
                optimizer.step()
            
            return math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        
        def test_epoch():
            model.eval()
            total_rmse = 0
            total_mask = 0
            
            with torch.no_grad():
                for idx, d in enumerate(test_loader):
                    source_rating, target_rating, _, _ = d
                    target_rating = target_rating.to(self.device)
                    
                    # Evaluate on target domain
                    _, _, target_pred = model(target_rating, False)
                    loss, mask = RMSE(target_pred, target_rating)
                    
                    total_rmse += loss.item()
                    total_mask += torch.sum(mask).item()
            
            return math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        
        epochs = self.config.get('epochs', 70)
        for epoch in tqdm(range(epochs), desc=f"Training {self.source_domain}->{self.target_domain}"):
            train_rmse.append(train_epoch())
            test_rmse.append(test_epoch())
        
        self.training_history = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        return self.training_history
    
    def evaluate(self, test_loader: Any) -> Dict[str, float]:
        """Evaluate I-DARec model."""
        from DArec_opt.I_DArec.function import MRMSELoss
        import math
        
        model = self.model
        model.eval()
        
        RMSE = MRMSELoss().to(self.device)
        total_rmse = 0
        total_mask = 0
        total_mae = 0
        
        with torch.no_grad():
            for idx, d in enumerate(test_loader):
                source_rating, target_rating, _, _ = d
                target_rating = target_rating.to(self.device)
                
                # Evaluate on target domain
                _, _, target_pred = model(target_rating, False)
                loss, mask = RMSE(target_pred, target_rating)
                
                total_rmse += loss.item()
                total_mask += torch.sum(mask).item()
                
                # Calculate MAE
                mae = torch.sum(torch.abs(target_pred - target_rating) * mask.float()).item()
                total_mae += mae
        
        rmse = math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        mae = total_mae / total_mask if total_mask > 0 else float('inf')
        
        return {
            'rmse': rmse,
            'mae': mae,
            'best_rmse': min(self.training_history.get('test_rmse', [rmse]))
        }


class UDarecMethod(BaseMethod):
    """U-DARec method implementation."""
    
    def prepare_data(self, data_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare data for U-DARec."""
        from DArec_opt.U_DArec.Data_Preprocessing import Mydata
        from torch.utils.data import DataLoader
        
        # Construct file paths
        data_dir = data_config['data_dir']
        source_path = os.path.join(data_dir, f"ratings_{self.source_domain}.csv")
        target_path = os.path.join(data_dir, f"ratings_{self.target_domain}.csv")
        
        # Create datasets
        train_dataset = Mydata(
            source_path, target_path, 
            train=True, 
            preprocessed=data_config.get('preprocessed', True)
        )
        test_dataset = Mydata(
            source_path, target_path, 
            train=False, 
            preprocessed=data_config.get('preprocessed', True)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=data_config.get('batch_size', 64), 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=data_config.get('batch_size', 64), 
            shuffle=False
        )
        
        # Update config with data dimensions
        self.config['n_users'] = train_dataset.S_data.shape[0]
        self.config['S_n_items'] = train_dataset.S_data.shape[1]
        self.config['T_n_items'] = train_dataset.T_data.shape[1]
        
        return train_loader, test_loader
    
    def build_model(self) -> nn.Module:
        """Build U-DARec model."""
        from DArec_opt.U_DArec.U_DArec import U_DArec
        
        # Create args object
        class Args:
            pass
        
        args = Args()
        for key, value in self.config.items():
            setattr(args, key, value)
        
        model = U_DArec(args)
        
        # Load pretrained weights if specified
        if self.config.get('pretrained_required', False):
            s_weights = self.config.get('S_pretrained_weights')
            t_weights = self.config.get('T_pretrained_weights')
            
            if s_weights and os.path.exists(s_weights):
                model.S_autorec.load_state_dict(torch.load(s_weights, map_location=self.device))
            if t_weights and os.path.exists(t_weights):
                model.T_autorec.load_state_dict(torch.load(t_weights, map_location=self.device))
        
        model = model.to(self.device)
        self.model = model
        return model
    
    def train(self, train_loader: Any, test_loader: Any) -> Dict[str, List[float]]:
        """Train U-DARec model."""
        import torch.optim as optim
        from DArec_opt.U_DArec.function import MRMSELoss, DArec_Loss
        import math
        import numpy as np
        from tqdm import tqdm
        
        model = self.model
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            weight_decay=self.config.get('wd', 1e-5),
            lr=self.config.get('lr', 1e-3)
        )
        
        RMSE = MRMSELoss().to(self.device)
        criterion = DArec_Loss().to(self.device)
        
        train_rmse = []
        test_rmse = []
        
        def train_epoch(epoch):
            model.train()
            total_rmse = 0
            total_mask = 0
            
            for idx, d in enumerate(train_loader):
                source_rating, target_rating, source_labels, target_labels = d
                source_rating = source_rating.to(self.device)
                target_rating = target_rating.to(self.device)
                
                # Calculate alpha for domain adaptation
                p = float(idx + epoch * len(train_loader)) / self.config.get('epochs', 20) / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                optimizer.zero_grad()
                
                # Forward pass
                source_class_output, source_pred, _ = model(source_rating, alpha, True)
                target_class_output, _, target_pred = model(target_rating, alpha, False)
                
                # Create labels
                source_labels_tensor = torch.zeros(source_rating.size(0), dtype=torch.long).to(self.device)
                target_labels_tensor = torch.ones(target_rating.size(0), dtype=torch.long).to(self.device)
                labels = torch.cat([source_labels_tensor, target_labels_tensor])
                
                # Combine outputs
                class_output = torch.cat([source_class_output, target_class_output])
                
                loss, source_mask, target_mask = criterion(
                    class_output, source_pred, target_pred,
                    source_rating, target_rating, labels
                )
                
                total_rmse += loss.item()
                total_mask += torch.sum(source_mask).item() + torch.sum(target_mask).item()
                
                loss.backward()
                optimizer.step()
            
            return math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        
        def test_epoch():
            model.eval()
            total_rmse = 0
            total_mask = 0
            
            with torch.no_grad():
                for idx, d in enumerate(test_loader):
                    source_rating, target_rating, _, _ = d
                    target_rating = target_rating.to(self.device)
                    
                    # Evaluate on target domain
                    _, _, target_pred = model(target_rating, 1.0, False)
                    loss, mask = RMSE(target_pred, target_rating)
                    
                    total_rmse += loss.item()
                    total_mask += torch.sum(mask).item()
            
            return math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        
        epochs = self.config.get('epochs', 20)
        for epoch in tqdm(range(epochs), desc=f"Training {self.source_domain}->{self.target_domain}"):
            train_rmse.append(train_epoch(epoch))
            test_rmse.append(test_epoch())
        
        self.training_history = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        return self.training_history
    
    def evaluate(self, test_loader: Any) -> Dict[str, float]:
        """Evaluate U-DARec model."""
        from DArec_opt.U_DArec.function import MRMSELoss
        import math
        
        model = self.model
        model.eval()
        
        RMSE = MRMSELoss().to(self.device)
        total_rmse = 0
        total_mask = 0
        total_mae = 0
        
        with torch.no_grad():
            for idx, d in enumerate(test_loader):
                source_rating, target_rating, _, _ = d
                target_rating = target_rating.to(self.device)
                
                # Evaluate on target domain
                _, _, target_pred = model(target_rating, 1.0, False)
                loss, mask = RMSE(target_pred, target_rating)
                
                total_rmse += loss.item()
                total_mask += torch.sum(mask).item()
                
                # Calculate MAE
                mae = torch.sum(torch.abs(target_pred - target_rating) * mask.float()).item()
                total_mae += mae
        
        rmse = math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        mae = total_mae / total_mask if total_mask > 0 else float('inf')
        
        return {
            'rmse': rmse,
            'mae': mae,
            'best_rmse': min(self.training_history.get('test_rmse', [rmse]))
        }


class AutoRecMethod(BaseMethod):
    """AutoRec baseline method."""
    
    def __init__(self, config: Dict[str, Any], source_domain: str, target_domain: str, variant: str = "I"):
        super().__init__(config, source_domain, target_domain)
        self.variant = variant  # "I" or "U"
        
    def prepare_data(self, data_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare data for AutoRec."""
        if self.variant == "I":
            from DArec_opt.I_DArec.Data_Preprocessing import Mydata
        else:
            from DArec_opt.U_DArec.Data_Preprocessing import Mydata
        
        from torch.utils.data import DataLoader
        
        # Construct file paths
        data_dir = data_config['data_dir']
        source_path = os.path.join(data_dir, f"ratings_{self.source_domain}.csv")
        target_path = os.path.join(data_dir, f"ratings_{self.target_domain}.csv")
        
        # Create datasets
        train_dataset = Mydata(
            source_path, target_path, 
            train=True, 
            preprocessed=data_config.get('preprocessed', True)
        )
        test_dataset = Mydata(
            source_path, target_path, 
            train=False, 
            preprocessed=data_config.get('preprocessed', True)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=data_config.get('batch_size', 64), 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=data_config.get('batch_size', 64), 
            shuffle=False
        )
        
        # Update config with data dimensions
        if self.variant == "I":
            self.config['n_users'] = train_dataset.S_data.shape[1] if self.config.get('train_S', False) else train_dataset.T_data.shape[1]
            self.config['n_items'] = train_dataset.S_data.shape[0] if self.config.get('train_S', False) else train_dataset.T_data.shape[0]
        else:
            self.config['n_users'] = train_dataset.S_data.shape[0] if self.config.get('train_S', False) else train_dataset.T_data.shape[0]
            self.config['n_items'] = train_dataset.S_data.shape[1] if self.config.get('train_S', False) else train_dataset.T_data.shape[1]
        
        return train_loader, test_loader
    
    def build_model(self) -> nn.Module:
        """Build AutoRec model."""
        if self.variant == "I":
            from DArec_opt.IAutoRec import I_AutoRec
            model = I_AutoRec(
                n_users=self.config['n_users'],
                n_items=self.config['n_items'],
                n_factors=self.config.get('n_factors', 200)
            )
        else:
            from DArec_opt.U_DArec.AutoRec import U_AutoRec
            model = U_AutoRec(
                n_users=self.config['n_users'],
                n_items=self.config['n_items'],
                n_factors=self.config.get('n_factors', 200)
            )
        
        model = model.to(self.device)
        self.model = model
        return model
    
    def train(self, train_loader: Any, test_loader: Any) -> Dict[str, List[float]]:
        """Train AutoRec model."""
        import torch.optim as optim
        from DArec_opt.I_DArec.function import MRMSELoss
        import math
        from tqdm import tqdm
        
        model = self.model
        optimizer = optim.Adam(
            model.parameters(),
            weight_decay=self.config.get('wd', 1e-4),
            lr=self.config.get('lr', 1e-3)
        )
        
        criterion = MRMSELoss().to(self.device)
        
        train_rmse = []
        test_rmse = []
        
        def train_epoch():
            model.train()
            total_rmse = 0
            total_mask = 0
            loc = 0 if self.config.get('train_S', False) else 1
            
            for idx, d in enumerate(train_loader):
                data = d[loc].to(self.device)
                optimizer.zero_grad()
                _, pred = model(data)
                
                loss, mask = criterion(pred, data)
                total_rmse += loss.item()
                total_mask += torch.sum(mask).item()
                
                loss.backward()
                optimizer.step()
            
            return math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        
        def test_epoch():
            model.eval()
            total_rmse = 0
            total_mask = 0
            loc = 0 if self.config.get('train_S', False) else 1
            
            with torch.no_grad():
                for idx, d in enumerate(test_loader):
                    data = d[loc].to(self.device)
                    _, pred = model(data)
                    
                    loss, mask = criterion(pred, data)
                    total_rmse += loss.item()
                    total_mask += torch.sum(mask).item()
            
            return math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        
        epochs = self.config.get('epochs', 50)
        for epoch in tqdm(range(epochs), desc=f"Training AutoRec {self.source_domain}->{self.target_domain}"):
            train_rmse.append(train_epoch())
            test_rmse.append(test_epoch())
        
        self.training_history = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        return self.training_history
    
    def evaluate(self, test_loader: Any) -> Dict[str, float]:
        """Evaluate AutoRec model."""
        from DArec_opt.I_DArec.function import MRMSELoss
        import math
        
        model = self.model
        model.eval()
        
        criterion = MRMSELoss().to(self.device)
        total_rmse = 0
        total_mask = 0
        total_mae = 0
        loc = 0 if self.config.get('train_S', False) else 1
        
        with torch.no_grad():
            for idx, d in enumerate(test_loader):
                data = d[loc].to(self.device)
                _, pred = model(data)
                
                loss, mask = criterion(pred, data)
                total_rmse += loss.item()
                total_mask += torch.sum(mask).item()
                
                # Calculate MAE
                mae = torch.sum(torch.abs(pred - data) * mask.float()).item()
                total_mae += mae
        
        rmse = math.sqrt(total_rmse / total_mask) if total_mask > 0 else float('inf')
        mae = total_mae / total_mask if total_mask > 0 else float('inf')
        
        return {
            'rmse': rmse,
            'mae': mae,
            'best_rmse': min(self.training_history.get('test_rmse', [rmse]))
        }


class MethodRegistry:
    """Registry for managing different DARec methods."""
    
    def __init__(self):
        self._methods: Dict[str, Type[BaseMethod]] = {}
        self._register_default_methods()
    
    def _register_default_methods(self):
        """Register default methods."""
        self.register("I_DARec", IDarecMethod)
        self.register("U_DARec", UDarecMethod)
        self.register("I_AutoRec", lambda config, source, target: AutoRecMethod(config, source, target, "I"))
        self.register("U_AutoRec", lambda config, source, target: AutoRecMethod(config, source, target, "U"))
    
    def register(self, name: str, method_class: Type[BaseMethod]):
        """Register a new method."""
        self._methods[name] = method_class
    
    def get_method(self, name: str, config: Dict[str, Any], source_domain: str, target_domain: str) -> BaseMethod:
        """Get a method instance."""
        if name not in self._methods:
            raise ValueError(f"Method {name} not registered. Available methods: {list(self._methods.keys())}")
        
        return self._methods[name](config, source_domain, target_domain)
    
    def list_methods(self) -> List[str]:
        """List all registered methods."""
        return list(self._methods.keys())
    
    def register_from_config(self, method_config):
        """Register a method from configuration."""
        # This allows for dynamic method registration
        # Implementation would depend on specific requirements
        pass


# Global registry instance
method_registry = MethodRegistry()
