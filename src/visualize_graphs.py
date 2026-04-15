import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Step 2: Define global Matplotlib rcParams
# This standardizes the financial aesthetic by removing 
# heavy borders and enabling clean gridlines.
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#EBEBEB',
    'grid.linewidth': 1.2,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.family': 'sans-serif'
})

# Step 3: Define explicit hex color palette
# These colors are adapted from professional financial publications 
# like the Financial Times to ensure high contrast and readability.
FINANCIAL_COLORS = {
    'trendy_teal': '#0D7680',
    'winter_cream': '#FFF1E0',
    'ivory_linen': '#F2DFCE',
    'bridal_heath': '#FFF9F5',
    'deep_navy': '#001F3F',
    'alert_red': '#D0222A'
}

def plot_recommendation_bar_chart(products, expected_margins):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Reverse lists so the highest margin product appears at the top of the horizontal chart
    products_rev = products[::-1]
    margins_rev = expected_margins[::-1]

    # Plot horizontal bars
    bars = ax.barh(products_rev, margins_rev, color=FINANCIAL_COLORS['deep_navy'])

    # Add exact data labels to the terminal end of each bar
    ax.bar_label(bars, fmt='$%.2f', padding=5, color=FINANCIAL_COLORS['deep_navy'], fontweight='bold')

    ax.set_xlabel("Expected Margin ($)")
    ax.set_title("Top 3 Next Best Actions by Expected Margin", loc='left', fontweight='bold')

    plt.tight_layout()
    return fig



# This function creates a dual-axis line and bar chart to visualize a customer's historical account balance and transaction volume over time.
# - The primary y-axis (line chart) shows the account balance, with a 3-month rolling average to highlight trends.
# - The secondary y-axis (bar chart) shows transaction volume, with a subtle overlay to avoid overpowering the balance data.
# - The x-axis is properly formatted with datetime objects to ensure accurate chronological representation.

def plot_historical_context_line(dates, balances, transaction_volumes):
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Convert string dates to datetime objects for proper Matplotlib chronological scaling
    dates = pd.to_datetime(dates)

    # Plot primary axis (Continuous Account Balance)
    ax1.plot(dates, balances, color=FINANCIAL_COLORS['trendy_teal'], linewidth=2.5, label="Account Balance")
    ax1.set_xlabel("Date", fontweight='bold')
    ax1.set_ylabel("Balance ($)", color=FINANCIAL_COLORS['trendy_teal'], fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=FINANCIAL_COLORS['trendy_teal'])

    # Calculate and plot a 3-month rolling average to smooth short-term volatility
    rolling_avg = pd.Series(balances).rolling(window=3, min_periods=1).mean()
    ax1.plot(dates, rolling_avg, color=FINANCIAL_COLORS['alert_red'], linestyle='--', linewidth=1.5, label="3-Month Trend")

    # Create a secondary axis for Transaction Volume (Bar Chart overlay)
    ax2 = ax1.twinx()
    ax2.bar(dates, transaction_volumes, alpha=0.2, color=FINANCIAL_COLORS['deep_navy'], width=20, label="Transaction Volume")
    ax2.set_ylabel("Tx Volume", color=FINANCIAL_COLORS['deep_navy'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=FINANCIAL_COLORS['deep_navy'])
    
    # Strip the top spine from the secondary axis to maintain the clean aesthetic
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(right=False)
    
    plt.title("Historical Account Balance & Activity", loc='left', fontweight='bold')
    fig.tight_layout()
    
    return fig


def plot_feature_importance(feature_names, importance_values):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort the features by importance magnitude in ascending order 
    # so the largest is at the top of the horizontal bar chart
    sorted_indices = np.argsort(importance_values)
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = [importance_values[i] for i in sorted_indices]
    
    # Plot horizontal bars
    ax.barh(sorted_names, sorted_importances, color=FINANCIAL_COLORS['trendy_teal'])
    
    ax.set_xlabel("Relative Importance Magnitude", fontweight='bold')
    ax.set_title("Neural Network Feature Drivers", loc='left', fontweight='bold')
    
    plt.tight_layout()
    return fig