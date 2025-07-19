# Customer Satisfaction Prediction

A machine learning pipeline that predicts customer satisfaction scores (1-5) using Brazilian e-commerce data. Built with ZenML and MLflow.

## What it does

Predicts customer satisfaction based on payment details, product characteristics, and shipping costs. **Best model: Random Forest with RMSE of 1.26**

## Quick Start

```bash
# Setup
git clone <your-repo>
cd customer-satisfaction
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
zenml init

# Train model
python run_pipeline.py

# Run web app
streamlit run streamlit_app.py
```

## Models Available

- **Random Forest** (default) - Best performance
- LightGBM - Fast gradient boosting
- XGBoost - Alternative boosting
- Linear Regression - Baseline

Change model in `steps/config.py`:

```python
model_name = "randomforest"  # or lightgbm, xgboost, linear_regression
```

## Results

| Model             | RMSE     | R² Score  |
| ----------------- | -------- | --------- |
| **Random Forest** | **1.26** | **0.164** |
| XGBoost           | 1.30     | 0.109     |
| LightGBM          | 1.32     | 0.083     |
| Linear Regression | 1.37     | 0.018     |

## Project Structure

```
├── data/           # Dataset
├── steps/          # Pipeline steps
├── model/          # Model training code
├── pipelines/      # ZenML pipelines
├── deployments/    # API and serving
├── streamlit_app.py # Web interface
└── requirements.txt
```

## Tech Stack

- **ML**: Scikit-learn, LightGBM, XGBoost
- **MLOps**: ZenML, MLflow
- **Web**: Streamlit, FastAPI
- **Data**: Pandas, NumPy
