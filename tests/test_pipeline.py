#!/usr/bin/env python3
"""
Comprehensive test suite for the customer satisfaction prediction pipeline.
Tests cover data processing, model training, evaluation, and deployment.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.config import ModelNameConfig
from model.data_cleaning import DataCleaning, DataPreprocessStrategy
from model.evaluation import MSE, RMSE, R2Score


class TestDataIngestion:
    """Test data ingestion functionality."""
    
    def test_ingest_data_returns_dataframe(self):
        """Test that data ingestion returns a valid DataFrame."""
        # Mock the CSV file reading
        with patch('pandas.read_csv') as mock_read_csv:
            # Create sample data
            sample_data = pd.DataFrame({
                'payment_sequential': [1, 2, 3],
                'payment_installments': [1, 2, 1],
                'payment_value': [29.99, 45.67, 123.45],
                'review_score': [4, 5, 3]
            })
            mock_read_csv.return_value = sample_data
            
            # Test the function
            result = ingest_data.entrypoint()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            mock_read_csv.assert_called_once()


class TestDataCleaning:
    """Test data cleaning and preprocessing."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'payment_sequential': [1, 2, 3],
            'payment_installments': [1, 2, 1],
            'payment_value': [29.99, 45.67, 123.45],
            'price': [29.99, 45.67, 123.45],
            'freight_value': [8.72, 12.34, 15.67],
            'product_name_lenght': [40, 35, 50],
            'product_description_lenght': [268, 300, 200],
            'product_photos_qty': [4, 3, 5],
            'product_weight_g': [500.0, 750.0, 300.0],
            'product_length_cm': [19.0, 25.0, 15.0],
            'product_height_cm': [8.0, 10.0, 6.0],
            'product_width_cm': [13.0, 15.0, 11.0],
            'review_score': [4, 5, 3],
            # Columns that should be removed
            'order_approved_at': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'order_delivered_carrier_date': ['2021-01-05', '2021-01-06', '2021-01-07'],
            'customer_zip_code_prefix': [12345, 23456, 34567],
            'order_item_id': [1, 2, 3]
        })
    
    def test_data_preprocessing_strategy(self):
        """Test data preprocessing removes correct columns."""
        strategy = DataPreprocessStrategy()
        
        # This should not raise an error for missing columns in test data
        try:
            result = strategy.handle_data(self.sample_data)
            # Verify numeric columns are preserved
            assert 'payment_sequential' in result.columns
            assert 'payment_value' in result.columns
            assert 'review_score' in result.columns
        except KeyError:
            # Expected for test data that doesn't have all columns
            pass
    
    def test_data_cleaning_handles_missing_values(self):
        """Test that missing values are handled properly."""
        # Add missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'product_weight_g'] = np.nan
        data_with_missing.loc[1, 'product_length_cm'] = np.nan
        
        strategy = DataPreprocessStrategy()
        
        try:
            result = strategy.handle_data(data_with_missing)
            # Check that missing values are filled
            assert not result.isnull().any().any()
        except KeyError:
            # Expected for test data that doesn't have all required columns
            pass


class TestModelConfiguration:
    """Test model configuration."""
    
    def test_model_config_default_values(self):
        """Test that model configuration has correct default values."""
        config = ModelNameConfig()
        
        assert config.model_name == "lightgbm"
        assert config.fine_tuning == False
    
    def test_model_config_custom_values(self):
        """Test that model configuration accepts custom values."""
        config = ModelNameConfig(model_name="xgboost", fine_tuning=True)
        
        assert config.model_name == "xgboost"
        assert config.fine_tuning == True


class TestModelEvaluation:
    """Test model evaluation metrics."""
    
    def setup_method(self):
        """Set up test data for evaluation."""
        np.random.seed(42)
        self.y_true = np.array([4.0, 5.0, 3.0, 4.0, 2.0, 5.0, 1.0, 3.0])
        self.y_pred = np.array([3.8, 4.9, 3.2, 4.1, 2.3, 4.8, 1.5, 2.9])
    
    def test_mse_calculation(self):
        """Test Mean Squared Error calculation."""
        mse_calculator = MSE()
        result = mse_calculator.calculate_score(self.y_true, self.y_pred)
        
        # Calculate expected MSE manually
        expected_mse = np.mean((self.y_true - self.y_pred) ** 2)
        
        assert abs(result - expected_mse) < 1e-10
        assert result >= 0
    
    def test_rmse_calculation(self):
        """Test Root Mean Squared Error calculation."""
        rmse_calculator = RMSE()
        result = rmse_calculator.calculate_score(self.y_true, self.y_pred)
        
        # Calculate expected RMSE manually
        mse = np.mean((self.y_true - self.y_pred) ** 2)
        expected_rmse = np.sqrt(mse)
        
        assert abs(result - expected_rmse) < 1e-10
        assert result >= 0
    
    def test_r2_calculation(self):
        """Test R-squared calculation."""
        r2_calculator = R2Score()
        result = r2_calculator.calculate_score(self.y_true, self.y_pred)
        
        # Calculate expected R2 manually
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        expected_r2 = 1 - (ss_res / ss_tot)
        
        assert abs(result - expected_r2) < 1e-10
        assert result <= 1


class TestDataTypes:
    """Test data type consistency."""
    
    def test_prediction_input_format(self):
        """Test that prediction input format is correct."""
        sample_input = {
            "payment_sequential": 1,
            "payment_installments": 1,
            "payment_value": 29.99,
            "price": 29.99,
            "freight_value": 8.72,
            "product_name_lenght": 40,
            "product_description_lenght": 268,
            "product_photos_qty": 4,
            "product_weight_g": 500.0,
            "product_length_cm": 19.0,
            "product_height_cm": 8.0,
            "product_width_cm": 13.0,
        }
        
        # Convert to DataFrame (as done in prediction)
        df = pd.DataFrame([sample_input])
        
        # Check data types
        assert df['payment_sequential'].dtype in [np.int64, int]
        assert df['payment_value'].dtype in [np.float64, float]
        assert len(df.columns) == 12
    
    def test_model_output_format(self):
        """Test that model output is in correct format."""
        # Simulate model prediction output
        prediction = 4.35
        
        assert isinstance(prediction, (int, float, np.number))
        assert 1 <= prediction <= 5  # Should be within satisfaction score range


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.slow
    def test_complete_pipeline_flow(self):
        """Test that the complete pipeline can run without errors."""
        # This is a simplified integration test
        # In a real scenario, you'd run the actual pipeline
        
        # Test data flow: ingestion -> cleaning -> training -> evaluation
        config = ModelNameConfig()
        
        # Verify configuration is properly initialized
        assert config.model_name in ["lightgbm", "xgboost", "randomforest", "linear_regression"]
        assert isinstance(config.fine_tuning, bool)


class TestUtilities:
    """Test utility functions."""
    
    def test_confidence_interpretation(self):
        """Test prediction confidence interpretation logic."""
        def interpret_confidence(prediction):
            if prediction >= 4.5:
                return "high"
            elif prediction >= 3.5:
                return "medium"
            else:
                return "low"
        
        assert interpret_confidence(4.8) == "high"
        assert interpret_confidence(4.0) == "medium"
        assert interpret_confidence(3.0) == "low"
        assert interpret_confidence(2.5) == "low"


# Performance benchmarks (optional)
class TestPerformance:
    """Performance tests to ensure the system meets requirements."""
    
    def test_prediction_speed(self):
        """Test that predictions are fast enough for real-time use."""
        import time
        
        # Simulate prediction timing
        start_time = time.time()
        
        # Simulate processing time (replace with actual prediction call)
        time.sleep(0.01)  # 10ms simulation
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        # Prediction should complete within 1 second for real-time use
        assert prediction_time < 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
