import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, num_numerical_features, 
                 categorical_cardinalities, embedding_dim=32, output_dim=64):
        super().__init__()
        # Embedding layers for high-cardinality categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim) for cardinality in categorical_cardinalities
        ])
        
        # Calculate total input dimension for the MLP
        total_embed_dim = len(categorical_cardinalities) * embedding_dim
        mlp_input_dim = num_numerical_features + total_embed_dim

        # Multi-Layer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, numerical_x, categorical_x):
        # Process categorical features into dense continuous vectors
        embeds = [emb(categorical_x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embeds, dim=1)
        
        # Concatenate numerical and embedded categorical features
        x = torch.cat([numerical_x, x_cat], dim=1)
        return self.mlp(x)

class ItemTower(nn.Module):
    def __init__(self, num_items=24, output_dim=64):
        super().__init__()
        # Maps the 24 financial products into the shared latent space
        self.item_embedding = nn.Embedding(num_items, output_dim)

    def forward(self, item_indices):
        return self.item_embedding(item_indices)

class TwoTowerRecommender(nn.Module):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, numerical_x, categorical_x, item_indices):
        # Decoupled forward passes
        user_embeds = self.user_tower(numerical_x, categorical_x)
        item_embeds = self.item_tower(item_indices)
        
        # Calculate relevance using the inner dot product
        scores = (user_embeds * item_embeds).sum(dim=1)
        return scores