import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Elite Estates | House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 15px rgba(0,123,255,0.3);
        transform: translateY(-2px);
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
        color: #212529; /* Explicit dark text for contrast */
    }

    .metric-card h2 {
        color: #212529 !important;
        margin: 0;
    }

    .metric-card p {
        color: #6c757d !important;
        margin-bottom: 2px;
    }

    .prediction-container {
        background: linear-gradient(135deg, #007bff 0%, #00d4ff 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(0,123,255,0.2);
    }

    .sidebar .sidebar-content {
        background-color: #ffffff;
    }

    h1, h2, h3 {
        color: #212529;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model: {e}. Please run the training script first.")
    st.stop()

# Load dataset for visualization
@st.cache_data
def load_data():
    return pd.read_csv('house_price_regression_dataset.csv').dropna()

df = load_data()

# Sidebar for inputs
st.sidebar.image("https://img.icons8.com/clouds/200/000000/home.png", width=100)
st.sidebar.title("Property Features")
st.sidebar.markdown("Adjust the features below to predict the market value.")

sq_ft_options = list(range(500, 5001, 100))
sq_ft = st.sidebar.selectbox("Square Footage", options=sq_ft_options, index=sq_ft_options.index(2500))

bedrooms = st.sidebar.selectbox("Number of Bedrooms", options=[1, 2, 3, 4, 5], index=2)
bathrooms = st.sidebar.selectbox("Number of Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1)

year_built_options = list(range(1950, 2025))
year_built = st.sidebar.selectbox("Year Built", options=year_built_options, index=year_built_options.index(2010))

lot_size_options = [round(x, 1) for x in np.arange(0.5, 5.1, 0.1)]
lot_size = st.sidebar.selectbox("Lot Size (Acres)", options=lot_size_options, index=lot_size_options.index(2.5))

garage_size = st.sidebar.radio("Garage Size (Cars)", options=[0, 1, 2], index=1, horizontal=True)

neighborhood_quality_options = list(range(1, 11))
neighborhood_quality = st.sidebar.selectbox("Neighborhood Quality Score", options=neighborhood_quality_options, index=6)

predict_btn = st.sidebar.button("Predict Output")

# Main Page Layout with Tabs
tab1, tab2 = st.tabs(["🏠 Home Valuator", "📊 Market Analytics"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🏠 Elite Estates Predictor")
        st.markdown("### Precision Market Valuation for Modern Homes")
        st.markdown("---")
        
        if predict_btn:
            features = np.array([[sq_ft, bedrooms, bathrooms, year_built, lot_size, garage_size, neighborhood_quality]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            
            st.markdown(f"""
                <div class="prediction-container">
                    <p style="font-size: 1.2rem; opacity: 0.9; margin-bottom: 5px;">Estimated Market Value</p>
                    <h1 style="font-size: 3.5rem; color: white; margin: 0;">${prediction:,.2f}</h1>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### Market Insight")
            # Square Footage vs Price Scatter with current prediction point
            fig = px.scatter(df, x='Square_Footage', y='House_Price', opacity=0.6,
                             labels={'Square_Footage': 'Living Area (sq ft)', 'House_Price': 'Price ($)'},
                             title="Trend Analysis: Size vs Price")
            
            fig.add_trace(go.Scatter(x=[sq_ft], y=[prediction], mode='markers',
                                     marker=dict(color='#ff4b4b', size=18, symbol='star', line=dict(color='white', width=2)),
                                     name='Your Estimate'))
            
            fig.update_traces(marker=dict(size=8))
            fig.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Set your property features in the sidebar and click **Predict Output** to see the valuation.")
            
            # Show a generic market trend if no prediction yet
            fig = px.scatter(df, x='Square_Footage', y='House_Price', opacity=0.4,
                             trendline="ols", trendline_color_override="#ff4b4b",
                             labels={'Square_Footage': 'Living Area (sq ft)', 'House_Price': 'Price ($)'},
                             title="Overall Market Trend (Size vs Price)")
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Property Specs")
        st.markdown(f"""
        <div class="metric-card">
            <p>Living Space</p>
            <h2>{sq_ft:,} <span style="font-size: 1rem; font-weight: 300;">sq ft</span></h2>
        </div>
        <br>
        <div class="metric-card">
            <p>Configuration</p>
            <h2>{bedrooms} <span style="font-size: 1rem; font-weight: 300;">Bed | </span>{bathrooms} <span style="font-size: 1rem; font-weight: 300;">Bath</span></h2>
        </div>
        <br>
        <div class="metric-card">
            <p>Built Year</p>
            <h2>{year_built}</h2>
        </div>
        <br>
        <div class="metric-card">
            <p>Land Area</p>
            <h2>{lot_size} <span style="font-size: 1rem; font-weight: 300;">Acres</span></h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Pro Tip: Neighborhood quality and square footage are the strongest predictors in this model.")

with tab2:
    st.title("📊 Market Analytics")
    st.markdown("### Strategic Data Insights for Property Valuation")
    
    with st.expander("🔍 View Raw Training Data"):
        st.write("Below is a sample of the historical property data used to train the Elite Estates model.")
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # Correlation Heatmap
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale='RdBu_r', origin='lower',
                             title="Feature Correlation Matrix")
        fig_corr.update_layout(template='plotly_white')
        st.write("#### 1. Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("A heatmap showing how different property features relate to each other and the final price.")

    with row1_col2:
        # House Price Distribution
        fig_dist = px.histogram(df, x="House_Price", nbins=30, marginal="box",
                                color_discrete_sequence=['#007bff'],
                                title="Market Price Distribution")
        fig_dist.update_layout(template='plotly_white')
        st.write("#### 2. Price Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("The spread of property prices across the entire market sample.")

    st.markdown("---")
    
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Neighborhood Quality Impact
        fig_neighborhood = px.box(df, x="Neighborhood_Quality", y="House_Price",
                                 color="Neighborhood_Quality",
                                 title="Neighborhood Quality vs Price Impact")
        fig_neighborhood.update_layout(template='plotly_white')
        st.write("#### 3. Neighborhood Appraisal")
        st.plotly_chart(fig_neighborhood, use_container_width=True)
        st.caption("Understanding how local quality ratings influence property valuation.")

    with row2_col2:
        # Feature Importance (Proxy via Coefficients)
        # Scale values roughly to show relative importance
        coef = model.coef_
        features_names = ['Size', 'Beds', 'Baths', 'Age', 'Lot', 'Garage', 'Quality']
        importance_df = pd.DataFrame({'Feature': features_names, 'Importance': np.abs(coef)})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        fig_import = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Blues',
                            title="Model Sensitivity Analysis (Feature Importance)")
        fig_import.update_layout(template='plotly_white')
        st.write("#### 4. Feature Importance")
        st.plotly_chart(fig_import, use_container_width=True)
        st.caption("Shows which factors have the most significant weight in the prediction formula.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Built with 💙 using Streamlit and Sklearn | Elite Estates © 2026</p>", unsafe_allow_html=True)
