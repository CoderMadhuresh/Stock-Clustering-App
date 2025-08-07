# 📊 Stock Clustering using PCA and KMeans

_This project performs quantitative analysis of Indian equities by applying Principal Component Analysis (PCA) and KMeans Clustering on historical return data of Nifty 50 stocks. It helps in grouping similar stocks based on their return patterns and volatilities, providing a solid foundation for portfolio diversification, risk management, and factor analysis._

## Objective 
To cluster Nifty 50 stocks based on their return structure and volatility characteristics using:

1. Log Return Calculation

2. Dimensionality Reduction (PCA)

3. KMeans Clustering

4. Volatility and Correlation Analysis

## Tech Stack
* Python

* Streamlit for interactive UI

* Yahoo Finance API (yfinance) for data fetching

* scikit-learn for PCA and KMeans

* matplotlib & seaborn for visualizations

* pandas & numpy for data manipulation

## Functionalities
1. Interactive selection of stock universe and cluster count

2. Visualize 2D and 3D PCA-based clusters

3. View cluster members and their statistical summary

4. Volatility distribution analysis per cluster

5. Time-series and correlation analysis between clusters

## Visual Outputs and Interpretations
### 1. 2D PCA Clustering
Visualizes how stocks group together based on return patterns. Clearly distinguishable clusters (e.g., tech stocks, PSU banks, Adani group).

### 2. 3D PCA Clustering

Offers deeper insight into spatial separation of stock clusters. Shows outlier behavior (e.g., INDUSINDBK.NS as a separate cluster).

### 3. Cluster Time-Series Returns

Tracks average return of each cluster over time. Helps identify consistent outperformers or laggards.

### 4. Volatility Distribution by Cluster

Boxplots reveal how volatility varies across clusters. Clusters like Adani stocks have higher volatility, whereas FMCG stocks have lower.

## Observations
### Cluster Members (Sample)  
Cluster 1: HCLTECH.NS, INFY.NS, TCS.NS, TECHM.NS, WIPRO.NS 

Cluster 2: COALINDIA.NS, JIOFIN.NS, NTPC.NS, POWERGRID.NS, SBIN.NS

Cluster 3: APOLLOHOSP.NS, BHARTIARTL.NS, CIPLA.NS, DRREDDY.NS, HDFCLIFE.NS, HEROMOTOCO.NS, SBILIFE.NS, SUNPHARMA.NS, TITAN.NS

Cluster 4: INDUSINDBK.NS

Cluster 5: ADANIENT.NS, ADANIPORTS.NS

Cluster 6: AXISBANK.NS, BAJAJFINSV.NS, BAJFINANCE.NS, GRASIM.NS, HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS, M&M.NS, MARUTI.NS, RELIANCE.NS, ULTRACEMCO.NS
 
Cluster 7: JSWSTEEL.NS, LT.NS, TATAMOTORS.NS, TATASTEEL.NS

Cluster 8: BEL.NS, ONGC.NS

Cluster 9: ASIANPAINT.NS, HINDUNILVR.NS, ITC.NS, NESTLEIND.NS, TATACONSUM.NS  

### Cluster Return & Volatility Summary:
|Cluster| Mean Return | Volatility |  
|:------|-------------|------------|  
| 0     | 0.000534    | 0.008012   |  
| 1     | 0.000969    | 0.012181   |  
| 2     | 0.000707    | 0.011313   |  
| 3     | -0.001145   | NaN        |  
| 4     | 0.000451    | 0.007574   |  
| 5     | 0.000719    | 0.010802   |  
| 6     | 0.000442    | 0.010282   |  
| 7     | 0.001612    | 0.011192   |  
| 8     | 0.000041    | 0.008627   |  


## Key Takeaways:

1. Cluster 7 (Auto/Metals) shows the highest average return.

2. Cluster 5 (Adani Group) has moderate returns but higher volatility.

3. Cluster 9 (FMCG) offers low volatility, stable returns — ideal for defensive positioning.

4. Cluster 4 (INDUSINDBK.NS) behaves as an outlier and forms a singleton cluster.

5. Low correlation between defensive (FMCG) and volatile clusters (Adani/Auto) supports diversification strategies.

6. High correlation within banking and infrastructure clusters (e.g., PSU stocks) is expected.

## Step-by-step workflow:
1. Fetch historical prices using yfinance
2. Calculate log returns
3. Apply PCA to reduce dimensions to top 3 components
4. Perform KMeans clustering on PCA results
5. Visualize clusters in 2D and 3D
6. Analyze cluster volatility, correlation, and time-series returns

## Use Cases
1. Quantitative Portfolio Construction

2. Risk-Parity Strategies

3. Market Regime Segmentation

4. Sector Rotation Analysis

5. Factor Clustering & Feature Engineering

6. Portfolio Diversification: Construct cluster-neutral baskets

## Getting Started
```bash
pip install streamlit yfinance pandas numpy scikit-learn matplotlib seaborn
streamlit run main.py
```
_You can also try this program on my Streamlit App -> [stock Clustering using PCA and KMeans](https://stock-clustering-app.streamlit.app)_
## Contributing
* Fork the repository
* Create a feature branch
* Commit your changes 
* Push and submit a PR

## Screenshots

[2D Graph
](https://github.com/CoderMadhuresh/Stock-Clustering-App/blob/main/Assets/2D%20Graph.png) </br>

[3D Graph
](https://github.com/CoderMadhuresh/Stock-Clustering-App/blob/main/Assets/3D%20Graph.png) </br>

[Matrix Correlation
](https://github.com/CoderMadhuresh/Stock-Clustering-App/blob/main/Assets/Correlation%20Matrix.png) </br>

[Volatility
](https://github.com/CoderMadhuresh/Stock-Clustering-App/blob/main/Assets/Volatility.png) </br>
