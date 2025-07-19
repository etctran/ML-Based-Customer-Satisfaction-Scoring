#!/usr/bin/env python3
"""
Customer Satisfaction Prediction - Streamlit Web Application
Professional MLOps interface for predicting customer satisfaction scores.
"""

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from model.data_cleaning import DataCleaning, DataPreprocessStrategy
except ImportError:
    st.error("Could not import data cleaning modules. Make sure you're running from the project root.")
    st.stop()

def load_model():
    """Try to load a trained model from various locations"""
    # Try different possible model locations for pickle files first
    possible_paths = [
        Path("saved_model"),
        Path("mlruns"),
        Path("model"),
        Path("."),
    ]
    
    # First try to load pickle files
    for path in possible_paths:
        if path.exists():
            # Look for pickle files
            for model_file in path.rglob("*.pkl"):
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    return model
                except Exception as e:
                    continue
    
    # Try MLflow if available
    try:
        import mlflow
        import mlflow.lightgbm
        
        # Set tracking URI to local mlruns using file:// format
        tracking_uri = f"file:///{os.path.abspath('./mlruns').replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Try to find runs with models
        runs = mlflow.search_runs(experiment_ids=["0"], max_results=5)
        
        if not runs.empty:
            # Look for runs with models
            for _, run in runs.iterrows():
                run_id = run['run_id']
                try:
                    # Try to load the model
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.lightgbm.load_model(model_uri)
                    return model
                except Exception as e:
                    continue
        
        st.warning("⚠️ No MLflow models found. Train a model first.")
        return None
        
    except ImportError:
        st.error("❌ MLflow not available. Train a model first with: `python run_pipeline.py`")
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not access MLflow: {e}")
        st.info("💡 Run: `python run_pipeline.py` to train a model")
        return None

def preprocess_input(data_dict):
    """Preprocess input data for prediction (simplified version)"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Simple preprocessing - just ensure numeric columns and handle missing values
        # Fill any missing values with median (though there shouldn't be any from UI)
        numeric_columns = [
            "payment_sequential", "payment_installments", "payment_value",
            "price", "freight_value", "product_name_lenght", 
            "product_description_lenght", "product_photos_qty",
            "product_weight_g", "product_length_cm", "product_height_cm", 
            "product_width_cm"
        ]
        
        # Ensure all columns are present and numeric
        for col in numeric_columns:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Select only the columns the model expects (in the right order)
        df = df[numeric_columns]
        
        return df
        
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Customer Satisfaction Predictor", 
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Customer Satisfaction Predictor")
    st.markdown("### Predict customer satisfaction score based on order features")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ No model available for predictions")
        st.info("🔧 **To get started:**")
        st.code("python run_pipeline.py", language="bash")
        st.stop()
    
    # Create sidebar for inputs
    st.sidebar.header("📊 Input Features")
    
    # Input fields
    payment_sequential = st.sidebar.slider(
        "Payment Sequential", 
        min_value=1, max_value=10, value=1,
        help="Number of payment methods used"
    )
    
    payment_installments = st.sidebar.slider(
        "Payment Installments", 
        min_value=1, max_value=24, value=1,
        help="Number of installments"
    )
    
    payment_value = st.sidebar.number_input(
        "Payment Value ($)", 
        min_value=0.0, value=29.99, step=0.01,
        help="Total amount paid"
    )
    
    price = st.sidebar.number_input(
        "Product Price ($)", 
        min_value=0.0, value=29.99, step=0.01,
        help="Price of the product"
    )
    
    freight_value = st.sidebar.number_input(
        "Freight Value ($)", 
        min_value=0.0, value=8.72, step=0.01,
        help="Shipping cost"
    )
    
    product_name_length = st.sidebar.slider(
        "Product Name Length", 
        min_value=1, max_value=100, value=40,
        help="Length of product name"
    )
    
    product_description_length = st.sidebar.slider(
        "Product Description Length", 
        min_value=1, max_value=1000, value=268,
        help="Length of product description"
    )
    
    product_photos_qty = st.sidebar.slider(
        "Product Photos Quantity", 
        min_value=1, max_value=20, value=4,
        help="Number of product photos"
    )
    
    product_weight_g = st.sidebar.number_input(
        "Product Weight (grams)", 
        min_value=0.0, value=500.0, step=1.0,
        help="Weight in grams"
    )
    
    product_length_cm = st.sidebar.number_input(
        "Product Length (cm)", 
        min_value=0.0, value=19.0, step=0.1,
        help="Length in centimeters"
    )
    
    product_height_cm = st.sidebar.number_input(
        "Product Height (cm)", 
        min_value=0.0, value=8.0, step=0.1,
        help="Height in centimeters"
    )
    
    product_width_cm = st.sidebar.number_input(
        "Product Width (cm)", 
        min_value=0.0, value=13.0, step=0.1,
        help="Width in centimeters"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Current Input Summary")
        
        input_data = {
            "payment_sequential": payment_sequential,
            "payment_installments": payment_installments,
            "payment_value": payment_value,
            "price": price,
            "freight_value": freight_value,
            "product_name_lenght": product_name_length,  # Note: keeping original typo for compatibility
            "product_description_lenght": product_description_length,  # Note: keeping original typo
            "product_photos_qty": product_photos_qty,
            "product_weight_g": product_weight_g,
            "product_length_cm": product_length_cm,
            "product_height_cm": product_height_cm,
            "product_width_cm": product_width_cm,
        }
        
        # Display input data in a nice format
        input_df = pd.DataFrame([input_data]).T
        input_df.columns = ["Value"]
        st.dataframe(input_df, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if st.button("🔮 Predict Customer Satisfaction", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    # Preprocess the input
                    processed_data = preprocess_input(input_data)
                    
                    if processed_data is not None:
                        # Make prediction
                        prediction = model.predict(processed_data)[0]
                        
                        # Display result
                        st.metric(
                            label="Predicted Satisfaction Score", 
                            value=f"{prediction:.2f}",
                            help="Scale: 1 (Very Dissatisfied) to 5 (Very Satisfied)"
                        )
                        
                        # Add interpretation
                        if prediction >= 4.5:
                            st.success("😄 Excellent! Customer likely to be very satisfied")
                        elif prediction >= 4.0:
                            st.success("😊 Good! Customer likely to be satisfied")
                        elif prediction >= 3.0:
                            st.warning("😐 Neutral. Customer satisfaction may vary")
                        elif prediction >= 2.0:
                            st.warning("😕 Poor. Customer likely to be dissatisfied")
                        else:
                            st.error("😤 Very Poor. Customer likely to be very dissatisfied")
                            
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    
    # Model information
    st.markdown("---")
    st.subheader("ℹ️ About This Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Algorithm**: Random Forest Regressor
        
        **Features**: 12 order and product characteristics
        
        **Target**: Customer satisfaction score (1-5)
        """)
    
    with col2:
        st.info("""
        **Performance**:
        - RMSE: ~1.26
        - R² Score: ~0.164
        - MSE: ~1.59
        """)
    
    with col3:
        st.info("""
        **Data Source**: Olist Brazilian E-commerce
        
        **Training**: ZenML + MLflow pipeline
        
        **Framework**: Scikit-learn compatible
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io/) and [ZenML](https://zenml.io/)")

if __name__ == "__main__":
    main()
