import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ---------------------------------------------------------
# CONFIG & SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")

# Load Assets
@st.cache_data
def load_data():
    # Load cleaned data for historical context
    df = pd.read_csv('data/processed/cleaned_sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model():
    # Load the trained Machine Learning model
    model = joblib.load('src/demand_model.pkl')
    return model

try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("Error: Could not find data/model. Make sure you ran the notebooks first!")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR: SCENARIO CONTROLS [Bible Ref: Scenario Analysis]
# ---------------------------------------------------------
st.sidebar.header("🕹️ Pricing Scenario Controls")

# Extract defaults from the latest data point
latest_data = df.iloc[-1]
default_price = float(latest_data['price'])
default_comp = float(latest_data['competitor_price'])

# Sliders
price_input = st.sidebar.slider("My Price ($)", min_value=10.0, max_value=200.0, value=default_price)
competitor_price_input = st.sidebar.slider("Competitor Price ($)", min_value=10.0, max_value=200.0, value=default_comp)
promo_input = st.sidebar.slider("Promotion Intensity (0-10)", min_value=0.0, max_value=10.0, value=float(latest_data['promotion_intensity']))
ad_spend_input = st.sidebar.slider("Ad Spend ($)", min_value=0.0, max_value=5000.0, value=float(latest_data['ad_spend']))

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
st.title("💰 Dynamic Pricing Optimization Engine")
st.markdown("Use the sidebar to simulate pricing scenarios and forecast revenue impact.")

# 1. PREDICTION ENGINE
# Prepare input vector matching the training features
# Features: ['price', 'competitor_price', 'promotion_intensity', 'ad_spend', 'competitor_diff', 'price_ratio', 'price_x_promo']
competitor_diff = price_input - competitor_price_input
price_ratio = price_input / (competitor_price_input + 0.01)
price_x_promo = price_input * promo_input

input_data = pd.DataFrame([{
    'price': price_input,
    'competitor_price': competitor_price_input,
    'promotion_intensity': promo_input,
    'ad_spend': ad_spend_input,
    'competitor_diff': competitor_diff,
    'price_ratio': price_ratio,
    'price_x_promo': price_x_promo
}])

# Predict
predicted_demand = model.predict(input_data)[0]
projected_revenue = predicted_demand * price_input

# Display KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Sales Volume", f"{int(predicted_demand)} units")
col2.metric("Projected Revenue", f"${projected_revenue:,.2f}")
col3.metric("Competitor Gap", f"${competitor_diff:.2f}", delta_color="inverse")

st.divider()

# 2. ELASTICITY SIMULATION (The "Optimization" Visual)
# Generate a demand curve for the current context
st.subheader("📉 Price Elasticity & Revenue Optimization Curve")

prices_to_test = np.linspace(10, 200, 50)
simulations = []

for p in prices_to_test:
    # Recalculate dependent features
    c_diff = p - competitor_price_input
    p_ratio = p / (competitor_price_input + 0.01)
    p_x_promo = p * promo_input
    
    sim_row = pd.DataFrame([{
        'price': p,
        'competitor_price': competitor_price_input,
        'promotion_intensity': promo_input,
        'ad_spend': ad_spend_input,
        'competitor_diff': c_diff,
        'price_ratio': p_ratio,
        'price_x_promo': p_x_promo
    }])
    
    pred_d = model.predict(sim_row)[0]
    pred_r = pred_d * p
    simulations.append({'Price': p, 'Demand': pred_d, 'Revenue': pred_r})

sim_df = pd.DataFrame(simulations)

# Plotting with Plotly
fig = go.Figure()

# Revenue Line
fig.add_trace(go.Scatter(x=sim_df['Price'], y=sim_df['Revenue'],
                         mode='lines', name='Revenue ($)', line=dict(color='green', width=3)))

# Demand Line (Secondary Axis)
fig.add_trace(go.Scatter(x=sim_df['Price'], y=sim_df['Demand'],
                         mode='lines', name='Demand (Units)', line=dict(color='blue', dash='dot'), yaxis='y2'))

# Mark selected price
fig.add_vline(x=price_input, line_dash="dash", line_color="red", annotation_text="Current Selection")

# Layout details
fig.update_layout(
    title="Projected Revenue vs. Price",
    xaxis_title="Price ($)",
    yaxis_title="Revenue ($)",
    yaxis2=dict(title="Demand (Units)", overlaying='y', side='right'),
    legend=dict(x=0, y=1.1, orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# 3. HISTORICAL CONTEXT
st.subheader("📊 Historical Trends")
tab1, tab2 = st.tabs(["Sales Volume", "Competitor Pricing"])

with tab1:
    fig_hist = px.line(df, x='date', y='footfall', title='Historical Footfall (Demand Proxy)')
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    fig_comp = px.line(df, x='date', y=['price', 'competitor_price'], title='Our Price vs Competitor')
    st.plotly_chart(fig_comp, use_container_width=True)