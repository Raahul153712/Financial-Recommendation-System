import torch
import numpy as np
import polars as pl
from model import UserTower, ItemTower, TwoTowerRecommender

def average_precision_at_k(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_items = 24
    
    # Load dataset and extract feature columns
    df = pl.read_parquet("data/processed/train_optimized.parquet").head(5000) # Small validation sample
    cat_cols = [col for col in df.columns if df[col].dtype == pl.Int32][:10]
    num_cols = [col for col in df.columns if df[col].dtype == pl.Float32][:10]
    
    numerical_data = df.select(num_cols).to_numpy()
    categorical_data = df.select(cat_cols).to_numpy()
    
    # Apply identical preprocessing from training
    means = np.nanmean(numerical_data, axis=0)
    stds = np.nanstd(numerical_data, axis=0)
    stds[stds == 0] = 1.0 
    numerical_data = np.nan_to_num((numerical_data - means) / stds, nan=0.0)
    categorical_data = np.abs(categorical_data) % 1000

    # Initialize model
    user_tower = UserTower(num_numerical_features=len(num_cols), 
                           categorical_cardinalities= [1000] *len(cat_cols)).to(device)
    item_tower = ItemTower(num_items=num_items).to(device)
    model = TwoTowerRecommender(user_tower, item_tower).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load("data/processed/two_tower_model.pt", map_location=device, weights_only=True))
    model.eval()

    # Generate the 24 item embeddings offline
    with torch.no_grad():
        all_items = torch.arange(num_items, device=device)
        item_embeddings = model.item_tower(all_items)

    map_score = 0.0
    
    # Simulate Evaluation
    np.random.seed(42)
    with torch.no_grad():
        for i in range(len(df)):
            num_x = torch.tensor(numerical_data[i:i+1], dtype=torch.float32).to(device)
            cat_x = torch.tensor(categorical_data[i:i+1], dtype=torch.long).to(device)
            
            # Single forward pass for the user
            user_embed = model.user_tower(num_x, cat_x)
            
            # Rapid dot product retrieval against all items
            scores = torch.matmul(user_embed, item_embeddings.T).squeeze()
            top_7_preds = torch.argsort(scores, descending=True)[:7].cpu().numpy().tolist()
            
            # Synthetic ground truth for demonstration
            actual_purchase = [np.random.randint(0, num_items)]
            map_score += average_precision_at_k(actual_purchase, top_7_preds, k=7)
            
    print(f"Validation MAP@7: {map_score / len(df):.4f}")

if __name__ == "__main__":
    evaluate()