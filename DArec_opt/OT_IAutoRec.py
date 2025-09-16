import torch
import torch.nn as nn
from function import *
from OT_torch_ import cost_matrix_batch_samples_torch, IPOT_distance_torch

class OT_AutoRec(nn.Module):
    """
    AutoRec with Optimal Transport regularization between two domains.
    """
    def __init__(self, n_users, n_items, n_factors=800, ot_weight=0.1):
        super(OT_AutoRec, self).__init__()
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.ot_weight = ot_weight
        self.encoder = nn.Sequential(
            nn.Linear(self.n_users, self.n_factors),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_factors, self.n_users),
            nn.Identity(),
        )

    def forward(self, x, x_other=None):
        """
        x: input data for this domain (batch, n_users)
        x_other: input data for the other domain (batch, n_users), for OT regularization
        Returns: embedding, prediction, ot_loss (if x_other is not None)
        """
        embedding = self.encoder(x)
        pred = self.decoder(embedding)
        ot_loss = None
        if x_other is not None:
            with torch.no_grad():
                embedding_other = self.encoder(x_other)
            # Ensure both embeddings have the same batch size for OT
            min_batch = min(embedding.size(0), embedding_other.size(0))
            if min_batch == 0:
                ot_loss = None
            else:
                emb1 = embedding[:min_batch]
                emb2 = embedding_other[:min_batch]
                # Compute cost matrix (cosine distance)
                cost = cost_matrix_batch_samples_torch(emb1, emb2)
                n = m = min_batch
                miu = torch.ones(n, 1, device=embedding.device) / n
                nu = torch.ones(m, 1, device=embedding.device) / m
                ot_loss = IPOT_distance_torch(cost, n, m, miu, nu)
        return embedding, pred, ot_loss