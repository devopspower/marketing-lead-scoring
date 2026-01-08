import torch
import torch.nn as nn

class LeadScoringNet(nn.Module):
    """
    Hybrid Deep Learning model for Tabular Lead Scoring.
    Combines Entity Embeddings for categorical features with a 
    Deep MLP for numerical features.
    """
    def __init__(self, emb_dims, n_num):
        """
        Args:
            emb_dims (list of tuples): List of (num_classes, embedding_dim) for each cat feature.
            n_num (int): Number of numerical features.
        """
        super(LeadScoringNet, self).__init__()
        
        # 1. Categorical Stream: Parallel Embedding Layers
        # Each categorical feature gets its own dedicated embedding space
        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        
        # Calculate the total size of concatenated embeddings
        n_emb = sum([y for x, y in emb_dims])
        
        # 2. Dense MLP Head
        # Input size = concatenated embeddings + numerical features
        self.mlp = nn.Sequential(
            nn.Linear(n_emb + n_num, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Normalizes gradients for stable tabular training
            nn.Dropout(0.3),       # Regularization to prevent overfitting on specific lead sources
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Final output is a single logit. 
            # We use BCEWithLogitsLoss during training for numerical stability.
            nn.Linear(32, 1) 
        )

    def forward(self, x_cat, x_num):
        """
        Forward pass.
        x_cat: Tensor of categorical indices [Batch, N_Categorical]
        x_num: Tensor of scaled numerical values [Batch, N_Numerical]
        """
        # Pass each categorical feature through its respective embedding layer
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_outputs.append(emb_layer(x_cat[:, i]))
        
        # Concatenate all embedding vectors
        x_emb = torch.cat(emb_outputs, dim=1)
        
        # Concatenate embeddings with numerical features
        combined = torch.cat([x_emb, x_num], dim=1)
        
        # Pass through the MLP stack
        logits = self.mlp(combined)
        
        # Squeeze to return a 1D tensor of scores
        return logits.squeeze()