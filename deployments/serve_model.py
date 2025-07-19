#!/usr/bin/env python3
"""
Manual MLflow Model Serving Script for Windows
Replaces the non-working run_deployment.py
"""

import os
import subprocess
import sys
import time
import requests
from pathlib import Path

def find_latest_model():
    """Find the latest trained model from MLflow runs"""
    mlruns_path = Path("mlruns")
    
    if not mlruns_path.exists():
        print("No mlruns directory found. Train a model first with: python run_pipeline.py")
        return None
    
    # Look for experiment directories
    experiment_dirs = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not experiment_dirs:
        print("No experiments found. Train a model first with: python run_pipeline.py")
        return None
    
    # Find the most recent run
    latest_run = None
    latest_time = 0
    
    for exp_dir in experiment_dirs:
        run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name != "meta.yaml"]
        
        for run_dir in run_dirs:
            model_path = run_dir / "artifacts" / "model"
            if model_path.exists():
                # Check modification time
                mtime = os.path.getmtime(run_dir)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_run = model_path
    
    return latest_run

def start_mlflow_server(model_path, port=5000):
    """Start MLflow model serving server"""
    print(f"Starting MLflow server for model: {model_path}")
    print(f"Server will be available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    # Build the mlflow serve command
    cmd = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", str(model_path),
        "-p", str(port),
        "--env-manager", "local"
    ]
    
    try:
        # Start the server
        process = subprocess.run(cmd, cwd=os.getcwd())
        return process.returncode == 0
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return True
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def test_prediction(port=5000):
    """Test the model server with sample data"""
    url = f"http://localhost:{port}/invocations"
    
    # Sample data for testing (adjust based on your model's expected input)
    sample_data = {
        "dataframe_split": {
            "columns": [
                "payment_sequential", "payment_installments", "payment_value",
                "price", "freight_value", "product_name_lenght",
                "product_description_lenght", "product_photos_qty",
                "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
            ],
            "data": [[1, 1, 29.99, 29.99, 8.72, 40, 268, 4, 500, 19, 8, 13]]
        }
    }
    
    try:
        response = requests.post(url, json=sample_data, timeout=10)
        if response.status_code == 200:
            prediction = response.json()
            print(f"Test prediction successful: {prediction}")
            return True
        else:
            print(f"Test prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Could not test prediction: {e}")
        return False

def main():
    print("🔧 Customer Satisfaction Model Server (Windows-Compatible)")
    print("=" * 60)
    
    # Find the latest model
    model_path = find_latest_model()
    
    if not model_path:
        print("\n💡 To train a model first, run:")
        print("   python run_pipeline.py")
        return 1
    
    print(f"✅ Found model: {model_path}")
    
    # Check if port is available
    port = 5000
    try:
        response = requests.get(f"http://localhost:{port}", timeout=1)
        print(f"⚠️  Port {port} is already in use. Server might already be running.")
        print(f"   Try accessing: http://localhost:{port}")
        return 1
    except requests.exceptions.RequestException:
        # Port is free, good to go
        pass
    
    # Start the server
    success = start_mlflow_server(model_path, port)
    
    if success:
        print("Server started successfully!")
        return 0
    else:
        print("Failed to start server")
        return 1

if __name__ == "__main__":
    exit(main())
