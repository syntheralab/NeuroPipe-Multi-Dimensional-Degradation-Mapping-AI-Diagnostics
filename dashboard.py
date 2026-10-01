import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Pipeline Integrity & Risk Assessment",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme Professional Aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #00D4B5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 14px;
        color: #A0AAB5;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & MODEL TRAINING (CACHED)
# ==========================================
@st.cache_data
def load_and_prep_data():
    # Load dataset
    df = pd.read_csv("pipeline_degradation_data.csv")
    
    # Define features and target
    features = [
        'pipe_age_years', 'operating_pressure_psi', 'flow_rate_m3h',
        'soil_corrosivity_index', 'wall_thickness_mm', 'temperature_c', 'h2s_content_ppm'
    ]
    target = 'degradation_rate_mmyear'
    
    X = df[features]
    y = df[target]
    
    # Train Random Forest Regressor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return df, model, features

df, model, features = load_and_prep_data()

# Dictionary readable labels
feature_labels = {
    'pipe_age_years': 'Pipe Age (Years)',
    'operating_pressure_psi': 'Operating Pressure (PSI)',
    'flow_rate_m3h': 'Flow Rate (m³/h)',
    'soil_corrosivity_index': 'Soil Corrosivity Index',
    'wall_thickness_mm': 'Wall Thickness (mm)',
    'temperature_c': 'Temperature (°C)',
    'h2s_content_ppm': 'H₂S Content (ppm)'
}

# ==========================================
# 3. SIDEBAR - CONTROL & INPUTS
# ==========================================
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("📍 Pipeline Parameter Simulation")
age = st.sidebar.slider("Pipe Age (Years)", 0, 50, 20)
pressure = st.sidebar.slider("Operating Pressure (PSI)", 100, 1500, 800)
flow_rate = st.sidebar.slider("Flow Rate (m³/h)", 50, 1000, 450)
soil_corr = st.sidebar.slider("Soil Corrosivity Index", 1.0, 10.0, 5.5)
wall_thick = st.sidebar.slider("Wall Thickness (mm)", 5.0, 30.0, 15.0)
temp = st.sidebar.slider("Temperature (°C)", 10, 100, 45)
h2s = st.sidebar.slider("H₂S Content (ppm)", 0, 100, 25)

# Build input DataFrame for prediction
user_input = pd.DataFrame([[age, pressure, flow_rate, soil_corr, wall_thick, temp, h2s]], columns=features)
predicted_rate = model.predict(user_input)[0]

# Calculate Estimated Remaining Life
remaining_wall = max(0.0, wall_thick - 3.0) # Assume 3mm minimum safe thickness
estimated_life = remaining_wall / predicted_rate if predicted_rate > 0 else 999.0

# ==========================================
# 4. MAIN DASHBOARD HEADER
# ==========================================
st.title("⚡ Pipeline Integrity & Risk Assessment Dashboard")
st.markdown("Real-time monitoring, machine learning degradation forecasting, and parameter sensitivity analysis.")
st.markdown("---")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Degradation Rate</div>
        <div class="metric-value">{predicted_rate:.3f} <span style="font-size:14px;">mm/year</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    risk_color = "#FF4B4B" if estimated_life < 10 else "#FFAA00" if estimated_life < 20 else "#00D4B5"
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {risk_color};">
        <div class="metric-label">Estimated Remaining Life</div>
        <div class="metric-value">{estimated_life:.1f} <span style="font-size:14px;">Years</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Operating Pressure</div>
        <div class="metric-value">{pressure} <span style="font-size:14px;">PSI</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Wall Thickness</div>
        <div class="metric-value">{wall_thick:.1f} <span style="font-size:14px;">mm</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. VISUALIZATIONS SECTION
# ==========================================
tab1, tab2 = st.columns(2)

with tab1:
    st.subheader("📊 Feature Importance Analysis")
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    importances.index = [feature_labels.get(col, col) for col in importances.index]
    
    # PERBAIKAN PADA BAGIAN INI:
    fig_imp = px.bar(
        x=importances.values,
        y=importances.index,
        orientation='h',
        title="Primary Drivers of Pipeline Degradation",
        labels={'x': 'Relative Importance', 'y': 'Pipeline Parameter'},
        color=importances.values,
        color_continuous_scale='Viridis',
        template="plotly_dark"
    )
    fig_imp.update_layout(showlegend=False, height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.subheader("📈 Operating Pressure vs Degradation Rate")
    fig_scatter = px.scatter(
        df,
        x='operating_pressure_psi',
        y='degradation_rate_mmyear',
        color='pipe_age_years',
        size='h2s_content_ppm',
        hover_data=['wall_thickness_mm'],
        title="Historical Pressure vs Degradation (Color: Age, Size: H₂S)",
        labels={
            'operating_pressure_psi': 'Operating Pressure (PSI)',
            'degradation_rate_mmyear': 'Degradation Rate (mm/yr)',
            'pipe_age_years': 'Age (Yrs)'
        },
        template="plotly_dark"
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# 6. HISTORICAL DATA TABLE
# ==========================================
st.markdown("---")
st.subheader("📋 Historical Pipeline Dataset")
st.dataframe(
    df.rename(columns=feature_labels).style.highlight_max(axis=0, color='#3B2323'),
    use_container_width=True,
    height=300
)