import torch
import torch.nn as nn
from function import *
import argparse
from AutoRec import I_AutoRec
# Import OT functions (optional for GW and W loss)
try:
    from OT_torch_ import GW_distance_uniform, cost_matrix_batch_torch, IPOT_distance_torch_batch_uniform
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    print("Warning: Optimal transport functions not available. OT loss will be disabled.")
class I_DArec(nn.Module):
    def __init__(self, args):
        """
        args:
          n_users: int
          S_n_items: int
          T_n_items: int
          n_factors: int
          RPE_hidden_size: int
          is_source: bool     if input is sourse data
        """

        super(I_DArec, self).__init__()
        self.args = args
        self.n_factors = args.n_factors
        self.n_users = args.n_users
        self.S_n_items = args.S_n_items
        self.T_n_items = args.T_n_items
        self.S_autorec = I_AutoRec(self.n_users, self.S_n_items, self.n_factors)
        self.T_autorec = I_AutoRec(self.n_users, self.T_n_items, self.n_factors)
        # 加载预训练
        # self.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
        # self.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
        # 冻结预训练过的AutoRec参数
        for para in self.S_autorec.parameters():
            para.requires_grad = False
        for para in self.T_autorec.parameters():
            para.requires_grad = False
        self.RPE_hidden_size = args.RPE_hidden_size

        self.RPE = nn.Sequential(
            nn.Linear(self.n_factors, self.RPE_hidden_size),
            nn.ReLU()
            # nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size),
            # nn.Sigmoid()
        )
        self.DC = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.Sigmoid(),
            nn.Linear(self.RPE_hidden_size // 2, 2),
            nn.Sigmoid()
            # nn.Linear(self.RPE_hidden_size // 4, 2),
            # nn.Sigmoid()
        )
        self.S_RP = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.n_users)
        )
        self.T_RP = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.n_users)
        )

    def forward(self, rating_matrix, is_source, source_rating_matrix=None, target_rating_matrix=None, 
                enable_ot=False, gw_weight=0.1, w_weight=0.1):
        """
        rating_matrix: input matrix
        is_source: bool
        source_rating_matrix, target_rating_matrix: needed for OT loss (batch, n_users)
        enable_ot: whether to compute full OT loss (GW + W distance) (requires both source and target matrices)
        gw_weight: weight for GW distance component
        w_weight: weight for Wasserstein distance component
        """
        # Standard forward for prediction/classification
        if is_source == True:
            embedding, _, _ = self.S_autorec(rating_matrix)
        else:
            embedding, _, _ = self.T_autorec(rating_matrix)
        feature = self.RPE(embedding)
        source_prediction = self.S_RP(feature)
        target_prediction = self.T_RP(feature)
        class_output = self.DC(feature)

        # Compute full OT loss if enabled and both source and target rating matrices are provided
        ot_loss = None
        if (enable_ot and OT_AVAILABLE and 
            source_rating_matrix is not None and target_rating_matrix is not None):
            # Get embeddings for both domains
            source_emb, _, _ = self.S_autorec(source_rating_matrix)
            target_emb, _, _ = self.T_autorec(target_rating_matrix)
            
            # Prepare embeddings for OT computation
            # Add batch dimension if missing
            if source_emb.dim() == 2:
                source_emb_ot = source_emb.unsqueeze(0)
            else:
                source_emb_ot = source_emb
            if target_emb.dim() == 2:
                target_emb_ot = target_emb.unsqueeze(0)
            else:
                target_emb_ot = target_emb
            
            # Compute GW distance
            gw_loss = None
            if gw_weight > 0:
                # Transpose to (batch, d, n) for GW computation
                source_emb_gw = source_emb_ot.transpose(1, 2)
                target_emb_gw = target_emb_ot.transpose(1, 2)
                gw_loss = GW_distance_uniform(source_emb_gw, target_emb_gw)
                if hasattr(gw_loss, 'mean'):
                    gw_loss = gw_loss.mean()
            
            # Compute Wasserstein distance
            w_loss = None
            if w_weight > 0:
                # Compute cosine distance matrix between embeddings
                cos_distance = cost_matrix_batch_torch(source_emb_ot.transpose(2, 1), target_emb_ot.transpose(2, 1))
                cos_distance = cos_distance.transpose(1, 2)
                
                # Apply threshold similar to the original usage
                beta = 0.1
                min_score = cos_distance.min()
                max_score = cos_distance.max()
                threshold = min_score + beta * (max_score - min_score)
                cos_dist = torch.nn.functional.relu(cos_distance - threshold)
                
                # Compute Wasserstein distance
                bs = source_emb_ot.size(0)
                n_items_source = source_emb_ot.size(1)
                n_items_target = target_emb_ot.size(1)
                w_loss = -IPOT_distance_torch_batch_uniform(cos_dist, bs, n_items_source, n_items_target, 30)
                if hasattr(w_loss, 'mean'):
                    w_loss = w_loss.mean()
            
            # Combine losses
            ot_loss = 0
            if gw_loss is not None:
                ot_loss += gw_weight * gw_loss
            if w_loss is not None:
                ot_loss += w_weight * w_loss
            
            # Return None if both weights are 0
            if gw_weight == 0 and w_weight == 0:
                ot_loss = None

        return class_output, source_prediction, target_prediction, ot_loss