import torch
import torch.nn as nn
from function import *
import matplotlib.pyplot as plt
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

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding, self.decoder(embedding)