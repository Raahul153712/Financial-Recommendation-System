import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import os

# Import the model classes you just created
from model import UserTower, ItemTower, TwoTowerRecommender

class SantanderDataset(Dataset):
    def __init__(self, data_path, num_items=24):
        # Load the optimized parquet file into memory
        self.df = pl.read_parquet(data_path)
        
        # Select numerical and categorical columns based on the casted datatypes
        self.cat_cols = [col for col in self.df.columns if self.df[col].dtype == pl.Int32][:10]
        self.num_cols = [col for col in self.df.columns if self.df[col].dtype == pl.Float32][:10]
        
        self.numerical_data = self.df.select(self.num_cols).to_numpy()
        self.categorical_data = self.df.select(self.cat_cols).to_numpy()

        # Normalize the numerical data to prevent FP16 overflow
        means = np.nanmean(self.numerical_data, axis=0)
        stds = np.nanstd(self.numerical_data, axis=0)
        
        # Prevent division by zero for any constant columns
        stds[stds == 0] = 1.0 
        
        self.numerical_data = (self.numerical_data - means) / stds
        
        # Impute remaining NaN values with the mean (0.0 after normalization)
        self.numerical_data = np.nan_to_num(self.numerical_data, nan=0.0)
        
        # Extract target item indices (Simulated for pipeline execution)
        self.purchased_items = np.random.randint(0, num_items, size=(len(self.df),))
        self.num_items = num_items

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.numerical_data[idx], dtype=torch.float32)
        # Ensure categorical inputs are strictly positive for the PyTorch Embedding layer
        cat_x = torch.tensor(np.abs(self.categorical_data[idx]) % 1000, dtype=torch.long)
        
        pos_item = torch.tensor([self.purchased_items[idx]], dtype=torch.long)
        
        # Negative Sampling: Randomly select an item the user did not purchase
        available_negatives = [i for i in range(self.num_items) if i!= pos_item.item()]
        neg_item = torch.tensor([np.random.choice(available_negatives)], dtype=torch.long)
        
        return num_x, cat_x, pos_item, neg_item

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on: {device}")

    # Strict hyperparameters to prevent VRAM overflow
    batch_size = 4096
    epochs = 3
    num_items = 24
    
    dataset = SantanderDataset("data/processed/train_optimized.parquet", num_items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_numerical = len(dataset.num_cols)
    # Define categorical cardinalities (Must exceed the maximum integer value in your categorical columns)
    cat_cardinalities =  [1000] * len(dataset.cat_cols)  # Adjust as needed based on your data

    user_tower = UserTower(num_numerical_features=num_numerical, categorical_cardinalities=cat_cardinalities).to(device)
    item_tower = ItemTower(num_items=num_items).to(device)
    model = TwoTowerRecommender(user_tower, item_tower).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() 

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (num_x, cat_x, pos_item, neg_item) in enumerate(dataloader):
            num_x, cat_x = num_x.to(device), cat_x.to(device)
            pos_item, neg_item = pos_item.to(device), neg_item.to(device)

            optimizer.zero_grad()

            # Execute forward pass in 16-bit Mixed Precision
            with autocast(device_type="cuda"):
                pos_scores = model(num_x, cat_x, pos_item.squeeze())
                pos_labels = torch.ones_like(pos_scores)
                
                neg_scores = model(num_x, cat_x, neg_item.squeeze())
                neg_labels = torch.zeros_like(neg_scores)
                
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([pos_labels, neg_labels])
                loss = criterion(scores, labels)

            # Scale gradients and optimize
            scaler.scale(loss).backward()
            
            # Unscale the gradients before clipping
            scaler.unscale_(optimizer)
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

    # Serialize and save model weights
    torch.save(model.state_dict(), "data/processed/two_tower_model.pt")
    print("Training complete. Model weights saved successfully.")

if __name__ == "__main__":
    train()