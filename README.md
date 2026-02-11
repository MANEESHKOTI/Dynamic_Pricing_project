# Dynamic Pricing & Demand Forecasting Engine

## 📌 Project Overview
This project implements an end-to-end dynamic pricing solution for e-commerce. It leverages machine learning (Random Forest) and econometric modeling (Log-Log Regression) to optimize prices, forecast demand, and maximize revenue.

## 📂 Repository Structure
* **`data/`**: Contains raw and processed datasets.
* **`notebooks/`**: Jupyter notebooks for EDA, Feature Engineering, and Modeling.
* **`src/`**: Source code for data loading and model training.
* **`dashboard/`**: Source code for the interactive Streamlit dashboard.
* **`reports/`**: Executive Summary and findings.

## 🚀 Key Features
* **Demand Forecasting:** Predicts sales volume based on price, competitor data, and seasonality.
* **Price Elasticity Analysis:** Quantifies consumer sensitivity to price changes using statistical modeling.
* **Revenue Optimization:** Algorithmic recommendation of optimal price points.
* **Interactive Dashboard:** Scenario simulation tool for business stakeholders.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Dynamic_Pricing_Project
Install dependencies:

Bash
pip install -r requirements.txt
Run the Data Pipeline:

Run src/data_loader.py to clean data.

Run notebooks 02 and 03 to train models.

Launch the Dashboard:

Bash
streamlit run dashboard/app.py
📊 Key Insights
Inelastic Demand: The analysis revealed a price elasticity of +0.11, suggesting that for this specific product segment, demand is driven more by Ad Spend and Promotion Intensity than by price cuts.

Strategic Implication: The business should focus on maintaining premium pricing while optimizing marketing spend, rather than engaging in price wars.