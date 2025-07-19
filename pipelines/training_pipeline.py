from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation
from steps.config import ModelNameConfig


@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    
    # Create model configuration
    config = ModelNameConfig()
    
    model = train_model(x_train, x_test, y_train, y_test, config)
    evaluation(model, x_test, y_test)
