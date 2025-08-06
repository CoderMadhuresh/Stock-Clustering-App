# Import Libraries
# Load necessary libraries for data fetching, analysis, visualization, and web UI
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Configuration
st.set_page_config(page_title="Stock Clustering App", layout="wide")
st.title("📊 Stock Clustering using PCA and KMeans")

# Sidebar Configuration for Clustering
# Allow user to select number of clusters and stock tickers
st.sidebar.header("Clustering Settings")
n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=15, value=9)

# Define Stock Universe
# Nifty 50 stocks (some are not available) as default options
all_tickers = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "SBIN.NS", "INFY.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "LT.NS", "HCLTECH.NS", "KOTAKBANK.NS", "SUNPHARMA.NS", "MARUTI.NS",
    "M&M.NS", "AXISBANK.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "NTPC.NS",
    "TITAN.NS", "ONGC.NS", "BEL.NS", "ADANIPORTS.NS", "ADANIENT.NS",
    "WIPRO.NS", "POWERGRID.NS", "TATAMOTORS.NS", "JSWSTEEL.NS",
    "COALINDIA.NS", "NESTLEIND.NS", "ASIANPAINT.NS", "TATASTEEL.NS",
    "JIOFIN.NS", "GRASIM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "TECHM.NS",
    "CIPLA.NS", "DRREDDY.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS",
    "INDUSINDBK.NS"
]

# Multiselect input for stocks
default_subset = all_tickers[:10]
selected_tickers = st.sidebar.multiselect("Select Stocks", options=all_tickers, default=default_subset)


# Step 1: Load Historical Data
# Download closing prices from Yahoo Finance
@st.cache_data(ttl=3600)
def load_data(tickers):
    df = yf.download(tickers, start='2000-01-01', end='2025-06-26')['Close']
    return df


data = load_data(selected_tickers)

# Step 2: Compute Log Returns
# Daily log returns are more stable for statistical modeling
returns = np.log(data / data.shift(1)).dropna()

# Step 3: Apply PCA (Dimensionality Reduction)
# Reduce high-dimensional return data into top 3 principal components
returns_clean = returns.dropna(axis=1)
returns_clean = returns_clean.apply(pd.to_numeric, errors='coerce')
returns_clean = returns_clean.dropna(axis=1)
X = returns_clean.T
if X.empty:
    st.error("❌ PCA input is empty. No valid stock data. Try selecting fewer or different tickers.")
    st.stop()

if X.isnull().values.any():
    st.error("❌ PCA input still contains NaNs.")
    st.stop()

pca = PCA(n_components=3)
pca_scores = pca.fit_transform(X)


# Step 4: KMeans Clustering
# Group similar stocks based on their PCA representation
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(pca_scores)

# Step 5: Organize Clustered Data
# Combine PCA scores with tickers and cluster labels
pca_df = pd.DataFrame(pca_scores, columns=['PC1', 'PC2', 'PC3'])
pca_df['Ticker'] = returns.columns
pca_df['Cluster'] = labels

# Step 6: Visualize PCA Clusters in 2D
st.subheader("📌 PCA Clustering (2D View)")
fig2d, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=100, ax=ax)
# Annotate each point with ticker name
for i, row in pca_df.iterrows():
    ax.text(row['PC1'] + 0.01, row['PC2'], row['Ticker'], fontsize=8)
plt.title("KMeans Clustering of Stocks in 2D PCA Space")
plt.grid(True)
st.pyplot(fig2d)

# Step 7: Show Cluster Members
st.subheader("🧾 Cluster Members")
for cluster_id in range(n_clusters):
    members = pca_df[pca_df['Cluster'] == cluster_id]['Ticker'].tolist()
    st.markdown(f"**Cluster {cluster_id + 1}:** " + ", ".join(members))

# Step 8: Visualize Clusters in 3D PCA Space
st.subheader("📍 PCA Clustering (3D View)")
fig3d = plt.figure(figsize=(10, 7))
ax3d = fig3d.add_subplot(111, projection='3d')
sc = ax3d.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['Cluster'], cmap='tab10', s=80)
ax3d.set_xlabel('PC1')
ax3d.set_ylabel('PC2')
ax3d.set_zlabel('PC3')
plt.colorbar(sc, label='Cluster')
st.pyplot(fig3d)

# Step 9: Time-Series Analysis of Cluster Mean Returns
# Compute mean return across all stocks in each cluster
st.subheader("📈 Cluster Time-Series Mean Return")
cluster_time_series = []
for cluster in range(n_clusters):
    members = pca_df[pca_df['Cluster'] == cluster]['Ticker']
    cluster_returns = returns[members].mean(axis=1)
    cluster_time_series.append(cluster_returns)

cluster_df = pd.concat(cluster_time_series, axis=1)
cluster_df.columns = [f'Cluster {i + 1}' for i in range(n_clusters)]

# Display line chart of cluster-wise average returns over time
st.line_chart(cluster_df)

# Step 10: Correlation Between Cluster Mean Returns
st.subheader("🔗 Correlation Between Cluster Mean Returns")
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(cluster_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)

# Step 11: Volatility Distribution by Cluster
# Calculate and visualize volatility for each stock grouped by cluster
st.subheader("📊 Stock Volatility by Cluster")
volatility_per_stock = returns.std()
volatility_df = pd.DataFrame({'Ticker': volatility_per_stock.index, 'Volatility': volatility_per_stock.values})
volatility_df = volatility_df.merge(pca_df[['Ticker', 'Cluster']], on='Ticker')

fig_vol, ax_vol = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Volatility', data=volatility_df, palette='tab10', ax=ax_vol)
plt.grid(True)
st.pyplot(fig_vol)

# Step 12: Cluster Summary Table
# Show each cluster's mean return and volatility (across stocks & time)
st.subheader("📋 Cluster Summary: Mean Return & Volatility")
cluster_summary = {}
for cluster in range(n_clusters):
    members = pca_df[pca_df['Cluster'] == cluster]['Ticker']
    cluster_returns = returns[members]
    mean_return = cluster_returns.mean(axis=1).mean()
    volatility = cluster_returns.std(axis=1).mean()
    cluster_summary[cluster + 1] = {'Mean Return': mean_return, 'Volatility': volatility}

summary_df = pd.DataFrame(cluster_summary).T
st.dataframe(summary_df.style.format({"Mean Return": "{:.4f}", "Volatility": "{:.4f}"}))








