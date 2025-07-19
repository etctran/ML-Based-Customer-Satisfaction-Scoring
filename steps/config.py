from pydantic import BaseModel


class ModelNameConfig(BaseModel):
    """Model Configurations"""

    model_name: str = "randomforest"  # Best performing model - RMSE: 1.2599
    fine_tuning: bool = False
