import torch
import torch.nn as nn
from function import *
import matplotlib.pyplot as plt
# Import GW distance from OT_torch_ (optional for GW loss)
try:
    from OT_torch_ import GW_distance_uniform
    GW_AVAILABLE = True
except ImportError:
    GW_AVAILABLE = False
    print("Warning: GW_distance_uniform not available. GW loss will be disabled.")
class U_AutoRec(nn.Module):
    """
    input -> hidden -> output(output.shape == input.shape)
    encoder: input -> hidden
    decoder: hidden -> output
    """

    def __init__(self, n_users, n_items, n_factors=800, p_drop=0.1):
        super(U_AutoRec, self).__init__()
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.encoder = nn.Sequential(
            nn.Linear(self.n_items, self.n_factors),
            # nn.Dropout(p_drop),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_factors, self.n_items),
            # nn.Dropout(p_drop)
        )

    def forward(self, x, other_domain_x=None, enable_gw=False):
        """
        Args:
            x: input tensor
            other_domain_x: input from other domain for GW loss computation
            enable_gw: whether to compute GW loss between domains
        
        Returns:
            tuple: (embedding, reconstruction, gw_loss)
                   gw_loss is None if not enabled or other_domain_x is None
        """
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        
        # Compute GW loss if enabled and other domain data is provided
        gw_loss = None
        if enable_gw and GW_AVAILABLE and other_domain_x is not None:
            # Get embedding for other domain
            other_embedding = self.encoder(other_domain_x)
            
            # Prepare embeddings for GW computation
            # Add batch dimension if missing
            if embedding.dim() == 2:
                embedding_gw = embedding.unsqueeze(0)
            else:
                embedding_gw = embedding
            if other_embedding.dim() == 2:
                other_embedding_gw = other_embedding.unsqueeze(0)
            else:
                other_embedding_gw = other_embedding
            
            # Transpose to (batch, d, n) for GW computation
            embedding_gw = embedding_gw.transpose(1, 2)
            other_embedding_gw = other_embedding_gw.transpose(1, 2)
            
            # Compute GW loss
            gw_loss = GW_distance_uniform(embedding_gw, other_embedding_gw)
            # If GW returns a tensor, take mean scalar
            if hasattr(gw_loss, 'mean'):
                gw_loss = gw_loss.mean()
        
        return embedding, reconstruction, gw_loss