import streamlit as st
import torch
import numpy as np
import polars as pl
import sys
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Ensure the src directory is in the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import UserTower, ItemTower, TwoTowerRecommender
from visualize_graphs import plot_recommendation_bar_chart, plot_historical_context_line, plot_feature_importance

st.set_page_config(page_title="Next Best Action Engine", layout="wide")
st.title("Financial Next Best Action (NBA) Recommender")

@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_items = 24
    
    df = pl.read_parquet("data/processed/train_optimized.parquet").head(100)
    cat_cols = [col for col in df.columns if df[col].dtype == pl.Int32][:10]
    num_cols = [col for col in df.columns if df[col].dtype == pl.Float32][:10]
    
    user_tower = UserTower(num_numerical_features=len(num_cols), categorical_cardinalities= [1000] *len(cat_cols)).to(device)
    item_tower = ItemTower(num_items=num_items).to(device)
    model = TwoTowerRecommender(user_tower, item_tower).to(device)
    
    model.load_state_dict(torch.load("data/processed/two_tower_model.pt", map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        all_items = torch.arange(num_items, device=device)
        item_embeddings = model.item_tower(all_items)
        
    # Generate synthetic profit margins for the 24 products
    np.random.seed(42)
    product_margins = np.random.uniform(10.0, 2000.0, size=num_items)

    return df, num_cols, cat_cols, model, item_embeddings, product_margins, device

df, num_cols, cat_cols, model, item_embeddings, product_margins, device = load_pipeline()

product_dict = {
    0: "Saving Account", 1: "Guarantees", 2: "Current Account", 3: "Derivative Account",
    4: "Payroll Account", 5: "Junior Account", 6: "MAS Particular Account", 7: "Particular Account",
    8: "Particular Plus Account", 9: "Short-term Deposits", 10: "Medium-term Deposits", 11: "Long-term Deposits",
    12: "e-account", 13: "Funds", 14: "Mortgages", 15: "Pensions",
    16: "Loans", 17: "Taxes", 18: "Credit Card", 19: "Securities",
    20: "Home Account", 21: "Payroll", 22: "Pension Account", 23: "Direct Debit"
}

user_idx = st.sidebar.number_input("Select Customer ID (0-99):", min_value=0, max_value=99, value=0)

if st.sidebar.button("Generate Recommendation"):
    user_row = df[user_idx]
    
    num_data = user_row.select(num_cols).to_numpy()
    cat_data = user_row.select(cat_cols).to_numpy()
    
    # Extract population statistics from the dataframe for Z-score normalization
    all_num_data = df.select(num_cols).to_numpy()
    means = np.nanmean(all_num_data, axis=0)
    stds = np.nanstd(all_num_data, axis=0)
    stds[stds == 0] = 1.0 
    
    # Apply identical Z-score normalization and mean imputation as used in training
    num_data = (num_data - means) / stds
    num_data = np.nan_to_num(num_data, nan=0.0)
    
    # Preprocess into tensors
    num_x = torch.tensor(num_data, dtype=torch.float32).to(device)
    cat_x = torch.tensor(np.abs(cat_data) % 1000, dtype=torch.long).to(device)
    
    with torch.no_grad():
        user_embed = model.user_tower(num_x, cat_x)
        logits = torch.matmul(user_embed, item_embeddings.T).squeeze()
        probabilities = torch.sigmoid(logits).cpu().numpy()
        
    expected_margins = probabilities * product_margins
    top_3_indices = np.argsort(expected_margins)[::-1][:3]
    
    st.subheader(f"Next Best Action for Customer #{user_idx}")
    
    # Extract the English product names and their corresponding margins
    top_3_names = [product_dict[idx] for idx in top_3_indices]
    top_3_margins = [expected_margins[idx] for idx in top_3_indices]
    
    # Generate the Matplotlib figure and render it in Streamlit
    fig_bar = plot_recommendation_bar_chart(top_3_names, top_3_margins)
    st.pyplot(fig_bar)
    st.subheader("Customer Financial Trajectory (6-Month History)")
    
    # Simulate 6 months of historical dates
    today = datetime.date.today()
    dates = pd.date_range(end=today, periods=6, freq='ME')
    
    # Generate synthetic cumulative balances and transaction volumes for the user
    np.random.seed(user_idx)
    base_balance = np.abs(float(num_data) * 1000) + 5000 
    balances = base_balance + np.random.normal(0, 1000, 6).cumsum()
    transaction_volumes = np.random.randint(10, 50, 6)
    
    # Generate the Matplotlib figure and render it in Streamlit
    fig_line = plot_historical_context_line(dates, balances, transaction_volumes)
    st.pyplot(fig_line)
    st.subheader("Model Diagnostics: Local Feature Importance")
    
    # Extract absolute weight magnitudes from the first linear layer
    first_layer_weights = model.user_tower.mlp[0].weight.detach().cpu().numpy()
    
    # Multiply the static weights by the customer's specific normalized input data to get local importance
    local_importance = np.abs(first_layer_weights[:, :len(num_cols)] * num_data).mean(axis=0)
    
    # Update the feature names for English display
    display_cols = ["Gross Income" if col == "renta" else col for col in num_cols]
    
    # Generate the Matplotlib figure and render it in Streamlit
    fig_importance = plot_feature_importance(display_cols, local_importance)
    st.pyplot(fig_importance)