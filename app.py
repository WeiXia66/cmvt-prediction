# ==============================================================================
# CatBoost CMVT Risk Prediction Model - Streamlit Version
# Version: 2.0 - Migrated from R Shiny with identical styling
# Author: Thomas Webster (何思吉)
# Date: 2025-11-28
# Features: Pie Chart + Feature Importance Plot + Enhanced UI
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from catboost import CatBoostClassifier
import os
import base64

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="CMVT Risk Prediction - CatBoost Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Custom CSS - Matching Shiny Style Exactly
# ==============================================================================
st.markdown("""
<style>
    /* ==================== Import Fonts ==================== */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap');
    
    /* ==================== Global Styles ==================== */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    
    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* ==================== Custom Header ==================== */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 20px;
        margin: -80px -80px 30px -80px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        border-bottom-left-radius: 20px;
        border-bottom-right-radius: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .custom-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
    }
    
    .custom-header::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -5%;
        width: 150px;
        height: 150px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    
    .custom-header h1 {
        margin: 0 0 15px 0;
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .custom-header p {
        margin: 8px 0 0 0;
        font-size: 1.15em;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    .badge-ml {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85em;
        margin-top: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* ==================== Sidebar Styles ==================== */
    [data-testid="stSidebar"] {
        background: white !important;
        padding: 20px;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 20px;
    }
    
    .sidebar-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 25px;
    }
    
    /* ==================== Input Group Styles ==================== */
    .input-group-custom {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px 18px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .input-group-custom:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        border-left-color: #764ba2;
    }
    
    .input-label {
        color: #1d3557;
        font-weight: 600;
        font-size: 1.05em;
        margin-bottom: 8px;
    }
    
    .unit-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        margin-left: 8px;
        font-weight: 600;
    }
    
    /* ==================== Button Styles ==================== */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Predict Button */
    div[data-testid="stButton"]:first-of-type > button {
        background: linear-gradient(135deg, #06a77d 0%, #05896a 100%) !important;
        color: white !important;
        font-size: 1.2em;
        padding: 16px 30px;
        border: none !important;
        box-shadow: 0 4px 15px rgba(6, 167, 125, 0.4);
    }
    
    div[data-testid="stButton"]:first-of-type > button:hover {
        background: linear-gradient(135deg, #05896a 0%, #047256 100%) !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(6, 167, 125, 0.5);
    }
    
    /* Reset Button */
    div[data-testid="stButton"]:last-of-type > button {
        background: linear-gradient(135deg, #f77f00 0%, #d67000 100%) !important;
        color: white !important;
        font-size: 1em;
        padding: 12px 25px;
        border: none !important;
        box-shadow: 0 3px 10px rgba(247, 127, 0, 0.3);
        margin-top: 10px;
    }
    
    div[data-testid="stButton"]:last-of-type > button:hover {
        background: linear-gradient(135deg, #d67000 0%, #bf6300 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(247, 127, 0, 0.4);
    }
    
    /* ==================== Main Panel ==================== */
    .main-panel {
        background: white;
        padding: 35px 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .panel-title {
        color: #1d3557;
        font-weight: 700;
        font-size: 1.8em;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
    }
    
    /* ==================== Result Card ==================== */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
    }
    
    .result-card h4 {
        color: #1d3557;
        font-weight: 700;
        margin-bottom: 15px;
        font-size: 1.4em;
    }
    
    /* ==================== Info Card ==================== */
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .info-card h5 {
        color: #1565c0;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .info-card p {
        color: #0d47a1;
        margin: 0;
        line-height: 1.6;
    }
    
    /* ==================== Risk Display ==================== */
    .risk-high {
        color: #e63946;
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-low {
        color: #06a77d;
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
    }
    
    .probability-display {
        color: #1d3557;
        font-size: 2em;
        font-weight: 700;
        text-align: center;
    }
    
    .probability-label {
        color: #6c757d;
        font-size: 1em;
        text-align: center;
    }
    
    /* ==================== Plot Container ==================== */
    .plot-container {
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        background: white;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.03);
    }
    
    /* ==================== Footer ==================== */
    .app-footer {
        background: white;
        padding: 25px;
        border-radius: 12px;
        margin-top: 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        color: #6c757d;
        line-height: 1.8;
    }
    
    .disclaimer {
        font-weight: 600;
        color: #e63946;
        font-size: 1.05em;
    }
    
    /* ==================== Welcome Message ==================== */
    .welcome-container {
        text-align: center;
        padding: 60px 20px;
    }
    
    .welcome-title {
        color: #4e73df;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    .welcome-subtitle {
        color: #6c757d;
        font-size: 1.3em;
        margin-bottom: 10px;
    }
    
    .welcome-powered {
        color: #adb5bd;
        font-size: 1em;
    }
    
    /* ==================== Metric Display ==================== */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* ==================== Section Divider ==================== */
    .section-divider {
        height: 3px;
        background: linear-gradient(to right, #667eea, #764ba2);
        border-radius: 3px;
        margin: 25px 0;
    }
    
    /* ==================== Streamlit Element Overrides ==================== */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #dee2e6;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #06a77d;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #e63946;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_image_as_base64(image_path):
    """Load image and convert to base64 for display"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

@st.cache_resource
def load_model():
    """Load CatBoost model with caching"""
    model_path = "CatBoost_model.cbm"
    if os.path.exists(model_path):
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    return None

@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    importance_path = "CatBoost_important.csv"
    if os.path.exists(importance_path):
        return pd.read_csv(importance_path)
    return None

@st.cache_data
def load_threshold():
    """Load optimal threshold from metrics file"""
    metrics_path = "Test_Evaluation_metrics_final.csv"
    if os.path.exists(metrics_path):
        try:
            metrics = pd.read_csv(metrics_path)
            catboost_metrics = metrics[metrics['Model'] == 'CatBoost']
            if len(catboost_metrics) > 0 and 'Threshold' in catboost_metrics.columns:
                threshold = float(catboost_metrics['Threshold'].iloc[0])
                if 0 < threshold < 1:
                    return threshold
        except Exception as e:
            pass
    return 0.5  # Default threshold

def create_pie_chart(prob_yes, pred_class, threshold):
    """Create risk pie chart using Plotly"""
    risk_color = "#e63946" if pred_class == "High Risk" else "#f77f00"
    non_risk_color = "#06a77d"
    
    fig = go.Figure(data=[go.Pie(
        labels=['Risk', 'non-Risk'],
        values=[prob_yes * 100, (1 - prob_yes) * 100],
        hole=0.5,
        marker_colors=[risk_color, non_risk_color],
        textinfo='percent',
        textfont_size=16,
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
    )])
    
    # Add center annotation
    fig.add_annotation(
        text=f"<b>{pred_class}</b><br>{prob_yes*100:.2f}%<br><span style='font-size:12px;color:#6c757d'>CMVT Probability</span>",
        x=0.5, y=0.5,
        font_size=20,
        font_color=risk_color if pred_class == "High Risk" else "#06a77d",
        showarrow=False
    )
    
    fig.update_layout(
        title=dict(
            text=f"Classification Threshold: {threshold*100:.1f}%",
            font=dict(size=14, color="#1d3557"),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=80, b=20, l=20, r=20),
        height=450
    )
    
    return fig

def create_feature_importance_plot(feature_importance):
    """Create feature importance bar chart"""
    if feature_importance is None:
        return None
    
    # Sort by importance
    df = feature_importance.sort_values('Overall', ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        x=df['Overall'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=df['Overall'],
            colorscale=[[0, '#a8dadc'], [1, '#457b9d']],
            line=dict(width=0)
        ),
        text=df['Overall'].round(2),
        textposition='outside',
        textfont=dict(size=12, color='#1d3557', family='Arial Black')
    )])
    
    fig.update_layout(
        title=dict(
            text="CatBoost Model - Feature Importance",
            font=dict(size=18, color="#1d3557", family="Arial Black"),
            x=0.5
        ),
        xaxis_title=dict(text="Importance Scores", font=dict(size=14, color="#1d3557")),
        yaxis_title=dict(text="Features", font=dict(size=14, color="#1d3557")),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e9ecef',
            gridwidth=0.5
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color='black', family='Arial Black')
        ),
        margin=dict(t=60, b=40, l=100, r=60),
        height=500
    )
    
    return fig

def create_welcome_figure():
    """Create welcome placeholder figure"""
    fig = go.Figure()
    
    fig.add_annotation(
        text="<b>CMVT Risk Prediction</b>",
        x=0.5, y=0.6,
        font=dict(size=36, color="#4e73df"),
        showarrow=False,
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        text="Enter patient data and click 'Predict'",
        x=0.5, y=0.45,
        font=dict(size=18, color="#6c757d"),
        showarrow=False,
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        text="Powered by CatBoost Machine Learning",
        x=0.5, y=0.32,
        font=dict(size=14, color="#adb5bd"),
        showarrow=False,
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=20, b=20, l=20, r=20),
        height=400
    )
    
    return fig

# ==============================================================================
# Initialize Session State
# ==============================================================================
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'prob_yes' not in st.session_state:
    st.session_state.prob_yes = None
if 'pred_class' not in st.session_state:
    st.session_state.pred_class = None

# Default values
DEFAULT_VALUES = {
    'DD': 400.0,
    'Age': 65,
    'K': 4.4,
    'BAS': 0.02,
    'SBP': 140,
    'CRP': 10.0,
    'CYSC': 2.6
}

# Initialize input values in session state
for key, default in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================================
# Header
# ==============================================================================
# Try to load hospital logo
logo_base64 = load_image_as_base64("hospital_logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height:100px; margin-right:20px; filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));">' if logo_base64 else '🏥'

st.markdown(f"""
<div class="custom-header">
    <h1 style="display: flex; align-items: center; justify-content: center; gap: 20px;">
        {logo_html}
        Calculation Tool for Predicting CMVT Risk in Hospitalized CKD Patients (CatBoost Model)
    </h1>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Load Model and Data
# ==============================================================================
model = load_model()
feature_importance = load_feature_importance()
optimal_threshold = load_threshold()

# Model status notification
if model is not None:
    st.sidebar.success(f"✓ Model loaded! Threshold: {optimal_threshold:.3f}")
else:
    st.sidebar.error("✗ Model not found. Please upload CatBoost_model.cbm")

# ==============================================================================
# Sidebar - Input Parameters
# ==============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-title">📋 Patient Parameters</p>', unsafe_allow_html=True)
    
    # Info card
    st.markdown("""
    <div class="info-card">
        <h5>📌 Data Entry</h5>
        <p>Enter patient clinical parameters below. All fields are required for prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # D-Dimer
    st.markdown("""
    <div class="input-group-custom">
        <div class="input-label">D-Dimer <span class="unit-badge">ng/ml</span></div>
    </div>
    """, unsafe_allow_html=True)
    DD = st.number_input("D-Dimer", min_value=0.0, max_value=25000.0, 
                         value=float(st.session_state.DD), step=10.0, 
                         label_visibility="collapsed", key="input_DD")
    
    # Age
    st.markdown("""
    <div class="input-group-custom">
        <div class="input-label">Age <span class="unit-badge">years</span></div>
    </div>
    """, unsafe_allow_html=True)
    Age = st.slider("Age", min_value=18, max_value=100, 
                    value=int(st.session_state.Age), step=1,
                    label_visibility="collapsed", key="input_Age")
    
    # Serum potassium
    st.markdown("""
    <div class="input-group-custom">
        <div class="input-label">Serum potassium <span class="unit-badge">mmol/L</span></div>
    </div>
    """, unsafe_allow_html=True)
    K = st.number_input("K", min_value=2.0, max_value=8.0, 
                        value=float(st.session_state.K), step=0.1,
                        label_visibility="collapsed", key="input_K")
    
    # Basophils (BAS)
    st.markdown("""
    <div class="input-group-custom">
        <div class="input-label">Basophils (BAS) <span class="unit-badge">×10⁹/L</span></div>
    </div>
    """, unsafe_allow_html=True)
    BAS = st.number_input("BAS", min_value=0.0, max_value=0.20, 
                          value=float(st.session_state.BAS), step=0.01,
                          format="%.2f", label_visibility="collapsed", key="input_BAS")
    
    # Systolic BP
    st.markdown("""
    <div class="input-group-custom">
        <div class="input-label">Systolic BP (SBP) <span class="unit-badge">mmHg</span></div>
    </div>
    """, unsafe_allow_html=True)
    SBP = st.number_input("SBP", min_value=80, max_value=250, 
                          value=int(st.session_state.SBP), step=1,
                          label_visibility="collapsed", key="input_SBP")
    
    # C-Reactive Protein
    st.markdown("""
    <div class="input-group-custom">
        <div class="input-label">C-Reactive Protein (CRP) <span class="unit-badge">mg/L</span></div>
    </div>
    """, unsafe_allow_html=True)
    CRP = st.number_input("CRP", min_value=0.0, max_value=300.0, 
                          value=float(st.session_state.CRP), step=0.5,
                          label_visibility="collapsed", key="input_CRP")
    
    # Cystatin C
    st.markdown("""
    <div class="input-group-custom">
        <div class="input-label">Cystatin C (CYSC) <span class="unit-badge">mg/L</span></div>
    </div>
    """, unsafe_allow_html=True)
    CYSC = st.number_input("CYSC", min_value=0.5, max_value=10.0, 
                           value=float(st.session_state.CYSC), step=0.1,
                           label_visibility="collapsed", key="input_CYSC")
    
    # Divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        predict_btn = st.button("🔮 PREDICT RISK", use_container_width=True, type="primary")
    
    with col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)

# ==============================================================================
# Handle Reset Button
# ==============================================================================
if reset_btn:
    for key, default in DEFAULT_VALUES.items():
        st.session_state[key] = default
    st.session_state.predicted = False
    st.session_state.prob_yes = None
    st.session_state.pred_class = None
    st.rerun()

# ==============================================================================
# Handle Predict Button
# ==============================================================================
if predict_btn:
    if model is None:
        st.error("⚠ Model not loaded! Please ensure CatBoost_model.cbm exists.")
    else:
        # Validation
        validation_rules = [
            {"value": DD, "min": 0, "max": 25000, "name": "D-Dimer", "unit": "ng/ml"},
            {"value": Age, "min": 18, "max": 100, "name": "Age", "unit": "years"},
            {"value": K, "min": 2.0, "max": 8.0, "name": "Serum potassium", "unit": "mmol/L"},
            {"value": BAS, "min": 0, "max": 0.20, "name": "BAS", "unit": "×10⁹/L"},
            {"value": SBP, "min": 80, "max": 250, "name": "SBP", "unit": "mmHg"},
            {"value": CRP, "min": 0, "max": 300, "name": "CRP", "unit": "mg/L"},
            {"value": CYSC, "min": 0.5, "max": 10.0, "name": "CYSC", "unit": "mg/L"}
        ]
        
        validation_passed = True
        for rule in validation_rules:
            if rule["value"] < rule["min"] or rule["value"] > rule["max"]:
                st.error(f"⚠ {rule['name']}: Valid range is {rule['min']}-{rule['max']} {rule['unit']}")
                validation_passed = False
                break
        
        if validation_passed:
            try:
                # Prepare input data (same order as training)
                new_data = pd.DataFrame({
                    'DD': [float(DD)],
                    'Age': [float(Age)],
                    'K': [float(K)],
                    'BAS': [float(BAS)],
                    'SBP': [float(SBP)],
                    'CRP': [float(CRP)],
                    'CYSC': [float(CYSC)]
                })
                
                # Make prediction
                prob_yes = model.predict_proba(new_data)[0][1]
                pred_class = "High Risk" if prob_yes >= optimal_threshold else "Low Risk"
                
                # Store results
                st.session_state.predicted = True
                st.session_state.prob_yes = prob_yes
                st.session_state.pred_class = pred_class
                
                st.success(f"✓ Prediction Complete: {pred_class} ({prob_yes*100:.2f}%)")
                
            except Exception as e:
                st.error(f"✗ Prediction error: {str(e)}")

# ==============================================================================
# Main Panel - Results Display
# ==============================================================================
st.markdown('<div class="main-panel">', unsafe_allow_html=True)
st.markdown('<h2 class="panel-title">📊 Prediction Result</h2>', unsafe_allow_html=True)

# Risk Classification Card
st.markdown('<div class="result-card"><h4>Risk Classification</h4>', unsafe_allow_html=True)

if st.session_state.predicted and st.session_state.prob_yes is not None:
    # Show prediction result
    fig = create_pie_chart(st.session_state.prob_yes, st.session_state.pred_class, optimal_threshold)
    st.plotly_chart(fig, use_container_width=True)
else:
    # Show welcome message
    fig = create_welcome_figure()
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Feature Importance Card
st.markdown("""
<div class="result-card">
    <h4>Feature Importance Analysis</h4>
    <div class="info-card">
        <h5>🧠 About Feature Importance</h5>
        <p>This chart shows the relative importance of each clinical parameter in the CatBoost model's decision-making process.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if feature_importance is not None:
    importance_fig = create_feature_importance_plot(feature_importance)
    st.plotly_chart(importance_fig, use_container_width=True)
else:
    st.warning("Feature importance data not available. Please ensure CatBoost_important.csv exists.")

st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("""
<div class="app-footer">
    <p>
        <span class="disclaimer">⚠ DISCLAIMER:</span>
        This CMVT Risk Prediction System is based on a CatBoost model trained on clinical data. 
        It is intended for research and clinical reference purposes only. 
        All clinical decisions should be made by qualified healthcare professionals after comprehensive 
        evaluation of the patient's complete medical history, physical examination, and other relevant 
        diagnostic information. This tool does not provide medical advice and should not be used as 
        a substitute for professional medical judgment.
    </p>
    <hr style="border-top: 1px solid #dee2e6; margin: 20px 0;">
    <p style="font-size: 0.95em;">
        Developed by WeiXia, Center for Scientific Research and Medical Transformation, 
        Jingzhou Hospital Affiliated to Yangtze University, Hubei, China | Website design by Mingye Lei
    </p>
</div>
""", unsafe_allow_html=True)
