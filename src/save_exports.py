import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure exports directory exists
os.makedirs('exports', exist_ok=True)

# Load Data
df = pd.read_csv('data/processed/featured_sales_data.csv')

# 1. Feature Correlation Matrix (Proof of Feature Engineering)
plt.figure(figsize=(10, 8))
corr = df[['estimated_sales', 'price', 'competitor_price', 'promotion_intensity', 'ad_spend']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('exports/correlation_matrix.png')
print("Saved exports/correlation_matrix.png")

# 2. Price vs Demand Scatter (Proof of Elasticity)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='estimated_sales', data=df, alpha=0.3)
plt.title('Price vs Estimated Sales Volume')
plt.xlabel('Price ($)')
plt.ylabel('Sales Volume')
plt.savefig('exports/price_elasticity_scatter.png')
print("Saved exports/price_elasticity_scatter.png")

# 3. Competitor Price Gap Distribution (Proof of Market Context)
plt.figure(figsize=(10, 6))
sns.histplot(df['competitor_diff'], kde=True, bins=30)
plt.title('Distribution of Price Gap (Us vs Competitor)')
plt.xlabel('Price Difference ($)')
plt.savefig('exports/competitor_gap_dist.png')
print("Saved exports/competitor_gap_dist.png")