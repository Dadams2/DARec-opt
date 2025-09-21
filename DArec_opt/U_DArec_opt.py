import torch
import torch.nn as nn
from function import *
import argparse
from U_DArec.AutoRec import U_AutoRec
# Import GW distance from OT_torch_
from OT_torch_ import GW_distance_uniform

class U_DArec(nn.Module):
    def __init__(self, args):
        """
        args:
          n_users: int
          S_n_items: int
          T_n_items: int
          n_factors: int
          RPE_hidden_size: int
          is_source: bool     if input is source data
        """

        super(U_DArec, self).__init__()
        self.args = args
        self.n_factors = args.n_factors
        self.n_users = args.n_users
        self.S_n_items = args.S_n_items
        self.T_n_items = args.T_n_items
        self.S_autorec = U_AutoRec(self.n_users, self.S_n_items, self.n_factors)
        self.T_autorec = U_AutoRec(self.n_users, self.T_n_items, self.n_factors)
        # Load pretrained weights if available
        # self.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
        # self.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
        # Freeze pretrained AutoRec parameters
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
            nn.Linear(self.RPE_hidden_size // 2, self.S_n_items)
        )
        self.T_RP = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.T_n_items)
        )

    def forward(self, rating_matrix, is_source, source_rating_matrix=None, target_rating_matrix=None, gw_alpha=0.1):
        """
        rating_matrix: input matrix
        is_source: bool
        source_rating_matrix, target_rating_matrix: needed for GW loss (batch, n_items)
        gw_alpha: weight for GW loss
        """
        # Standard forward for prediction/classification
        if is_source == True:
            embedding, _ = self.S_autorec(rating_matrix)
        else:
            embedding, _ = self.T_autorec(rating_matrix)
        feature = self.RPE(embedding)
        source_prediction = self.S_RP(feature)
        target_prediction = self.T_RP(feature)
        class_output = self.DC(feature)

        # Compute GW loss if both source and target rating matrices are provided
        gw_loss = None
        if source_rating_matrix is not None and target_rating_matrix is not None:
            # Get embeddings for both domains (detach to avoid double backward if needed)
            source_emb, _ = self.S_autorec(source_rating_matrix)
            target_emb, _ = self.T_autorec(target_rating_matrix)
            # GW expects shape (batch, d, n), so transpose if needed
            # Here, batch = 1, d = n_factors, n = n_items (for U_DArec)
            # source_emb: (batch, n_factors)
            # Add batch dim if missing
            if source_emb.dim() == 2:
                source_emb = source_emb.unsqueeze(0)
            if target_emb.dim() == 2:
                target_emb = target_emb.unsqueeze(0)
            # Transpose to (batch, d, n)
            source_emb = source_emb.transpose(1, 2)
            target_emb = target_emb.transpose(1, 2)
            # Compute GW loss
            gw_loss = GW_distance_uniform(source_emb, target_emb)
            # If GW returns a tensor, take mean scalar
            if hasattr(gw_loss, 'mean'):
                gw_loss = gw_loss.mean()

        return class_output, source_prediction, target_prediction, gw_loss


def create_args(n_users, S_n_items, T_n_items, n_factors=400, RPE_hidden_size=200, 
                S_pretrained_weights=None, T_pretrained_weights=None):
    """Helper function to create args object for U_DArec"""
    class Args:
        def __init__(self):
            self.n_users = n_users
            self.S_n_items = S_n_items
            self.T_n_items = T_n_items
            self.n_factors = n_factors
            self.RPE_hidden_size = RPE_hidden_size
            self.S_pretrained_weights = S_pretrained_weights
            self.T_pretrained_weights = T_pretrained_weights
    
    return Args()


if __name__ == "__main__":
    # Test the U_DArec model
    n_users = 1000
    S_n_items = 500
    T_n_items = 600
    batch_size = 32
    
    # Create args
    args = create_args(n_users, S_n_items, T_n_items)
    
    # Create model
    model = U_DArec(args).cuda()
    
    # Create sample data
    source_data = torch.randn(batch_size, S_n_items).cuda()
    target_data = torch.randn(batch_size, T_n_items).cuda()
    
    # Test forward pass
    class_output, source_pred, target_pred, gw_loss = model(
        source_data, True, 
        source_rating_matrix=source_data, 
        target_rating_matrix=target_data
    )
    
    print(f"Class output shape: {class_output.shape}")
    print(f"Source prediction shape: {source_pred.shape}")
    print(f"Target prediction shape: {target_pred.shape}")
    print(f"GW loss: {gw_loss}")