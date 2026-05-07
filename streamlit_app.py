import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="QuickHyreAI Forecaster", layout="wide", initial_sidebar_state="expanded")

API_URL = "http://127.0.0.1:8000"

st.title("🎯 QuickHyreAI - Sales Forecasting System")
st.markdown("**Multi-Model Time Series Forecasting with ARIMA, Prophet, XGBoost & LSTM**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    horizon = st.slider("Forecast Horizon (days)", 1, 56, 8, help="Number of days to forecast")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh API Status"):
            st.rerun()
    with col2:
        if st.button("📊 Retrain Models"):
            with st.spinner("Training all models..."):
                try:
                    response = requests.post(f"{API_URL}/train", timeout=600)
                    if response.status_code == 200:
                        st.success("✅ Training complete!")
                    else:
                        st.error(f"Training failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ API Error: {e}")

# Fetch available states
try:
    response = requests.get(f"{API_URL}/", timeout=10)
    api_data = response.json()
    available_states = api_data.get("available_states", [])
    trained_states = api_data.get("trained_states", [])
except requests.exceptions.RequestException:
    st.error("❌ Cannot connect to API. Make sure the server is running at http://127.0.0.1:8000")
    st.stop()

# Main UI
tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📈 Analytics", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_state = st.selectbox(
            "Select State",
            available_states,
            index=0 if available_states else None,
            help="Choose a state to forecast"
        )
    
    with col2:
        is_trained = selected_state.lower() in [s.lower() for s in trained_states]
        status_color = "🟢" if is_trained else "🟡"
        st.metric("Model Status", f"{status_color} {'Trained' if is_trained else 'Demo'}")
    
    if selected_state:
        with st.spinner(f"Fetching forecast for {selected_state}..."):
            try:
                response = requests.get(
                    f"{API_URL}/predict/{selected_state}",
                    params={"horizon": horizon},
                    timeout=30
                )
                
                if response.status_code == 200:
                    forecast_data = response.json()
                    
                    # Display key info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("State", forecast_data["state"])
                    with col2:
                        st.metric("Model", forecast_data["model_type"].upper())
                    with col3:
                        st.metric("Horizon", f"{forecast_data['horizon']} days")
                    
                    # Create DataFrame
                    forecast_list = forecast_data["forecast"]
                    df = pd.DataFrame(forecast_list)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date")
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["date"],
                        y=df["prediction"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#0066cc", width=3),
                        marker=dict(size=8)
                    ))
                    fig.update_layout(
                        title=f"Sales Forecast - {selected_state}",
                        xaxis_title="Date",
                        yaxis_title="Sales ($)",
                        hovermode="x unified",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.subheader("📋 Forecast Details")
                    df_display = df.copy()
                    df_display["date"] = df_display["date"].dt.strftime("%Y-%m-%d")
                    df_display["prediction"] = df_display["prediction"].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv,
                        file_name=f"{selected_state.lower()}_forecast.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. Server might be busy.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Error: {e}")

with tab2:
    st.subheader("📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total States", len(available_states))
    with col2:
        st.metric("Trained Models", len(trained_states))
    with col3:
        st.metric("Coverage", f"{len(trained_states)}/{len(available_states)}")
    
    st.subheader("✅ Trained States")
    if trained_states:
        # Display in grid
        cols = st.columns(4)
        for idx, state in enumerate(sorted(trained_states)):
            with cols[idx % 4]:
                st.info(f"🟢 {state.title()}")
    else:
        st.warning("No states trained yet. Click 'Retrain Models' to train.")
    
    st.subheader("📈 Model Architecture")
    st.markdown("""
    **4 Competing Models:**
    1. **ARIMA/SARIMA** - Classical time series model with seasonal decomposition
    2. **Facebook Prophet** - Handles trend, seasonality, and holidays
    3. **XGBoost** - Gradient boosting with engineered lag/rolling features
    4. **LSTM** - Deep learning neural network for temporal patterns
    
    **Best Model Selection:** Trained on validation set, ranked by MAE → RMSE
    """)

with tab3:
    st.subheader("ℹ️ About This System")
    st.markdown("""
    ### QuickHyreAI Forecasting System
    
    A production-ready time series forecasting platform built with:
    - **Backend:** FastAPI + Python
    - **Models:** ARIMA, Prophet, XGBoost, LSTM
    - **Features:** Lag (1,7,30), Rolling Mean/Std, Holiday Flags, Time Features
    - **Validation:** Time-series split with no leakage
    - **Data:** 50 US states, beverage sales (2019-2023)
    
    ### API Endpoints
    - `GET /` - System status & available states
    - `GET /predict/{state}?horizon=8` - Get forecast
    - `POST /train` - Retrain all models
    - `GET /health` - Health check
    
    ### How to Use
    1. Select a state from the dropdown
    2. Choose forecast horizon (1-56 days)
    3. View predictions and download CSV
    4. Click "Retrain Models" to update with latest data
    """)
    
    st.divider()
    st.caption("✨ Built with FastAPI + Streamlit | Model Selection: Best MAE on Validation Split")
