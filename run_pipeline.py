from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    # The pipeline function expects a data_path parameter, but our ingest_data step 
    # has the path hardcoded, so we can pass any dummy value
    training_run = train_pipeline("C:\Projects\customer-satisfaction\data\olist_customers_dataset.csv")

    print("Pipeline execution completed!")
    print("Training and evaluation finished successfully.")
    print(f"Run ID: {training_run.id}")
    print("Check the ZenML dashboard at http://127.0.0.1:8080 for more details.")
