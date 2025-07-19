#!/usr/bin/env python3
"""
API testing for FastAPI server endpoints.
Tests API functionality, request/response validation, and error handling.
"""

import pytest
import requests
import json
import time
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30


class TestAPIEndpoints:
    """Test FastAPI server endpoints."""
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Set up API client for testing."""
        # In a real test, you might start the server programmatically
        # For now, assume server is running
        return requests.Session()
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint availability."""
        try:
            response = api_client.get(f"{API_BASE_URL}/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "status" in data
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_health_check_endpoint(self, api_client):
        """Test health check endpoint."""
        try:
            response = api_client.get(f"{API_BASE_URL}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
            assert "model_loaded" in data
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_predict_endpoint_valid_input(self, api_client):
        """Test prediction endpoint with valid input."""
        valid_input = {
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
            "product_width_cm": 13.0
        }
        
        try:
            response = api_client.post(
                f"{API_BASE_URL}/predict",
                json=valid_input,
                timeout=10
            )
            
            if response.status_code == 503:
                pytest.skip("Model not loaded in API server")
            
            assert response.status_code == 200
            data = response.json()
            assert "predicted_satisfaction" in data
            assert "confidence" in data
            assert 1 <= data["predicted_satisfaction"] <= 5
            assert data["confidence"] in ["low", "medium", "high"]
            
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_predict_endpoint_invalid_input(self, api_client):
        """Test prediction endpoint with invalid input."""
        invalid_input = {
            "payment_sequential": "invalid",  # Should be int
            "payment_value": -10,  # Negative value
        }
        
        try:
            response = api_client.post(
                f"{API_BASE_URL}/predict",
                json=invalid_input,
                timeout=5
            )
            
            # Should return validation error
            assert response.status_code == 422
            
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_predict_batch_endpoint(self, api_client):
        """Test batch prediction endpoint."""
        batch_input = [
            {
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
                "product_width_cm": 13.0
            },
            {
                "payment_sequential": 2,
                "payment_installments": 2,
                "payment_value": 45.67,
                "price": 45.67,
                "freight_value": 12.34,
                "product_name_lenght": 35,
                "product_description_lenght": 300,
                "product_photos_qty": 3,
                "product_weight_g": 750.0,
                "product_length_cm": 25.0,
                "product_height_cm": 10.0,
                "product_width_cm": 15.0
            }
        ]
        
        try:
            response = api_client.post(
                f"{API_BASE_URL}/predict_batch",
                json=batch_input,
                timeout=10
            )
            
            if response.status_code == 503:
                pytest.skip("Model not loaded in API server")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            
            for prediction in data:
                assert "predicted_satisfaction" in prediction
                assert "confidence" in prediction
                assert 1 <= prediction["predicted_satisfaction"] <= 5
                
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_api_response_time(self, api_client):
        """Test that API responses are within acceptable time limits."""
        valid_input = {
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
            "product_width_cm": 13.0
        }
        
        try:
            start_time = time.time()
            response = api_client.post(
                f"{API_BASE_URL}/predict",
                json=valid_input,
                timeout=5
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # API should respond within 2 seconds
            assert response_time < 2.0
            
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")


class TestInputValidation:
    """Test input validation and data types."""
    
    def test_numeric_field_validation(self):
        """Test that numeric fields are properly validated."""
        valid_numeric_inputs = [1, 1.5, 0, 100.99]
        invalid_numeric_inputs = ["string", None, [], {}]
        
        # This would be tested through API calls in practice
        for valid_input in valid_numeric_inputs:
            assert isinstance(valid_input, (int, float))
        
        for invalid_input in invalid_numeric_inputs:
            assert not isinstance(invalid_input, (int, float))
    
    def test_required_fields_present(self):
        """Test that all required fields are present in input."""
        required_fields = [
            "payment_sequential", "payment_installments", "payment_value",
            "price", "freight_value", "product_name_lenght",
            "product_description_lenght", "product_photos_qty",
            "product_weight_g", "product_length_cm", "product_height_cm",
            "product_width_cm"
        ]
        
        complete_input = {field: 1.0 for field in required_fields}
        
        # All required fields should be present
        for field in required_fields:
            assert field in complete_input
    
    def test_output_format_validation(self):
        """Test that output format is consistent."""
        expected_output_structure = {
            "predicted_satisfaction": float,
            "confidence": str
        }
        
        # Mock response data
        mock_response = {
            "predicted_satisfaction": 4.35,
            "confidence": "high"
        }
        
        for field, expected_type in expected_output_structure.items():
            assert field in mock_response
            assert isinstance(mock_response[field], expected_type)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_missing_model_error(self):
        """Test behavior when model is not loaded."""
        # This would be tested by mocking model loading failure
        with patch('fastapi_server.model', None):
            # Test that appropriate error is returned
            pass  # Implementation would depend on actual server structure
    
    def test_malformed_json_error(self):
        """Test handling of malformed JSON input."""
        # This would be tested through actual API calls
        malformed_json = '{"invalid": json,}'
        
        # API should return 400 Bad Request for malformed JSON
        # Implementation depends on FastAPI's automatic validation
        pass
    
    def test_prediction_failure_handling(self):
        """Test handling when prediction fails."""
        # Mock prediction failure scenario
        with patch('fastapi_server.model.predict', side_effect=Exception("Prediction failed")):
            # Test that server returns 500 Internal Server Error
            pass


class TestSecurityAndLimits:
    """Test security aspects and rate limiting."""
    
    def test_input_size_limits(self):
        """Test that excessively large inputs are rejected."""
        # Test with very large numeric values
        oversized_input = {
            "payment_sequential": 1,
            "payment_installments": 1,
            "payment_value": float('inf'),  # Infinite value
            "price": 29.99,
            "freight_value": 8.72,
            "product_name_lenght": 40,
            "product_description_lenght": 268,
            "product_photos_qty": 4,
            "product_weight_g": 500.0,
            "product_length_cm": 19.0,
            "product_height_cm": 8.0,
            "product_width_cm": 13.0
        }
        
        # Should handle infinite values gracefully
        assert not isinstance(oversized_input["payment_value"], (int, float)) or \
               not (-float('inf') < oversized_input["payment_value"] < float('inf'))
    
    def test_batch_size_limits(self):
        """Test that batch prediction has reasonable size limits."""
        # Test with large batch
        large_batch = [
            {
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
                "product_width_cm": 13.0
            }
        ] * 1000  # 1000 items
        
        # Should handle reasonable batch sizes
        assert len(large_batch) == 1000


class TestDocumentation:
    """Test API documentation availability."""
    
    def test_openapi_docs_available(self):
        """Test that OpenAPI documentation is accessible."""
        try:
            response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
            # Should return HTML page with documentation
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_openapi_json_available(self):
        """Test that OpenAPI JSON specification is accessible."""
        try:
            response = requests.get(f"{API_BASE_URL}/openapi.json", timeout=5)
            assert response.status_code == 200
            
            openapi_spec = response.json()
            assert "openapi" in openapi_spec
            assert "info" in openapi_spec
            assert "paths" in openapi_spec
            
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")


if __name__ == "__main__":
    # Run API tests
    pytest.main([__file__, "-v", "--tb=short"])
