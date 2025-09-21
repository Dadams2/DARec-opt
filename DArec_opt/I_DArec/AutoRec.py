import torch
import torch.nn as nn
from function import *
# Import OT functions (optional for GW and W loss)
try:
    from OT_torch_ import GW_distance_uniform, cost_matrix_batch_torch, IPOT_distance_torch_batch_uniform
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    print("Warning: Optimal transport functions not available. OT loss will be disabled.")
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

    def forward(self, x, other_domain_x=None, enable_ot=False, gw_weight=0.1, w_weight=0.1):
        """
        Args:
            x: input tensor
            other_domain_x: input from other domain for OT loss computation
            enable_ot: whether to compute full OT loss (GW + W distance) between domains
            gw_weight: weight for GW distance component
            w_weight: weight for Wasserstein distance component
        
        Returns:
            tuple: (embedding, reconstruction, ot_loss)
                   ot_loss is None if not enabled or other_domain_x is None
        """
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        
        # Compute full OT loss if enabled and other domain data is provided
        ot_loss = None
        if enable_ot and OT_AVAILABLE and other_domain_x is not None:
            # Get embedding for other domain
            other_embedding = self.encoder(other_domain_x)
            
            # Prepare embeddings for OT computation
            # Add batch dimension if missing
            if embedding.dim() == 2:
                embedding_ot = embedding.unsqueeze(0)
            else:
                embedding_ot = embedding
            if other_embedding.dim() == 2:
                other_embedding_ot = other_embedding.unsqueeze(0)
            else:
                other_embedding_ot = other_embedding
            
            # Transpose to (batch, d, n) for GW computation
            embedding_gw = embedding_ot.transpose(1, 2)
            other_embedding_gw = other_embedding_ot.transpose(1, 2)
            
            # Compute GW distance
            gw_loss = None
            if gw_weight > 0:
                gw_loss = GW_distance_uniform(embedding_gw, other_embedding_gw)
                if hasattr(gw_loss, 'mean'):
                    gw_loss = gw_loss.mean()
            
            # Compute Wasserstein distance
            w_loss = None
            if w_weight > 0:
                # Compute cosine distance matrix between embeddings
                cos_distance = cost_matrix_batch_torch(embedding_ot.transpose(2, 1), other_embedding_ot.transpose(2, 1))
                cos_distance = cos_distance.transpose(1, 2)
                
                # Apply threshold similar to the original usage
                beta = 0.1
                min_score = cos_distance.min()
                max_score = cos_distance.max()
                threshold = min_score + beta * (max_score - min_score)
                cos_dist = torch.nn.functional.relu(cos_distance - threshold)
                
                # Compute Wasserstein distance
                bs = embedding_ot.size(0)
                n_items_x = embedding_ot.size(1)
                n_items_other = other_embedding_ot.size(1)
                w_loss = -IPOT_distance_torch_batch_uniform(cos_dist, bs, n_items_x, n_items_other, 30)
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
        
        return embedding, reconstruction, ot_loss


if __name__ == "__main__":
    # 行是user，列是Item
    x = torch.randn(5000, 1500)
    net = I_AutoRec(1500, 5000)
    loss = MRMSELoss()
    
    # Test original functionality (backward compatible)
    result = net(x)
    if len(result) == 3:
        embedding, y, ot_loss = result
        print(f"OT loss: {ot_loss}")
    else:
        embedding, y = result
        print("No OT loss (backward compatible mode)")
    
    print(y.shape)
    print(embedding.shape)
    print(loss(x, y).data)
    
    # Test full OT functionality with dummy second domain data
    x2 = torch.randn(5000, 1500)  # Different domain data
    embedding_ot, y_ot, ot_loss = net(x, other_domain_x=x2, enable_ot=True, gw_weight=0.1, w_weight=0.1)
    print(f"With OT enabled - Full OT loss: {ot_loss}")
    print(f"OT embedding shape: {embedding_ot.shape}")
    print(f"OT reconstruction shape: {y_ot.shape}")
    
    # Test only GW distance
    embedding_gw, y_gw, gw_only_loss = net(x, other_domain_x=x2, enable_ot=True, gw_weight=0.1, w_weight=0.0)
    print(f"With only GW enabled - GW loss: {gw_only_loss}")
    
    # Test only W distance
    embedding_w, y_w, w_only_loss = net(x, other_domain_x=x2, enable_ot=True, gw_weight=0.0, w_weight=0.1)
    print(f"With only W enabled - W loss: {w_only_loss}")
