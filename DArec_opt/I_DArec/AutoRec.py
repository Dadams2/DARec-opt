import torch
import torch.nn as nn
from function import *
# Import GW distance from OT_torch_ (optional for GW loss)
try:
    from OT_torch_ import GW_distance_uniform
    GW_AVAILABLE = True
except ImportError:
    GW_AVAILABLE = False
    print("Warning: GW_distance_uniform not available. GW loss will be disabled.")
class I_AutoRec(nn.Module):
    """
    input -> hidden -> output(output.shape == input.shape)
    encoder: input -> hidden
    decoder: hidden -> output
    """

    def __init__(self, n_users, n_items, n_factors=800):
        super(I_AutoRec, self).__init__()
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.encoder = nn.Sequential(
            nn.Linear(self.n_users, self.n_factors),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_factors, self.n_users),
            nn.Identity(),
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


if __name__ == "__main__":
    # 行是user，列是Item
    x = torch.randn(5000, 1500)
    net = I_AutoRec(1500, 5000)
    loss = MRMSELoss()
    
    # Test original functionality (backward compatible)
    result = net(x)
    if len(result) == 3:
        embedding, y, gw_loss = result
        print(f"GW loss: {gw_loss}")
    else:
        embedding, y = result
        print("No GW loss (backward compatible mode)")
    
    print(y.shape)
    print(embedding.shape)
    print(loss(x, y).data)
    
    # Test GW functionality with dummy second domain data
    x2 = torch.randn(5000, 1500)  # Different domain data
    embedding_gw, y_gw, gw_loss = net(x, other_domain_x=x2, enable_gw=True)
    print(f"With GW enabled - GW loss: {gw_loss}")
    print(f"GW embedding shape: {embedding_gw.shape}")
    print(f"GW reconstruction shape: {y_gw.shape}")
