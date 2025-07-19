import logging

import mlflow
import numpy as np
import pandas as pd
from model.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step

from typing import Tuple


@step
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        
        # Log metrics using MLflow (with proper experiment context)
        try:
            # Set experiment to Default (ID: 0) if not already set
            mlflow.set_experiment("Default")
            
            # Only log metrics if we're in an active run, otherwise just skip logging
            if mlflow.active_run():
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("rmse", rmse)
            else:
                print(f"Metrics calculated - MSE: {mse}, R2: {r2_score}, RMSE: {rmse}")
        except Exception as mlflow_error:
            print(f"MLflow logging failed: {mlflow_error}")
            print(f"Metrics calculated - MSE: {mse}, R2: {r2_score}, RMSE: {rmse}")
        
        return r2_score, rmse
    except Exception as e:
        logging.error(e)
        raise e
