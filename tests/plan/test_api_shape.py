"""
Tests for treasury planning API response shape and validation.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from oasis.api import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_treasury_request() -> dict[str, Any]:
    """Valid treasury planning request."""
    return {
        "current_balance": 150000.0,
        "risk_tolerance": "medium",
        "expected_inflows": {
            "multiplier": 1.2,
            "events": [{"day": 3, "amount": 25000}, {"day": 7, "amount": 15000}],
        },
        "expected_outflows": {
            "multiplier": 0.9,
            "events": [{"day": 5, "amount": 10000}],
        },
        "vendor_payment_schedule": {"upcoming_payments": [8000, 5000, 3000]},
    }


def test_treasury_plan_api_success(
    client: TestClient, valid_treasury_request: dict[str, Any]
) -> None:
    """Test successful treasury planning API request."""
    response = client.post("/treasury/plan", json=valid_treasury_request)

    assert response.status_code == 200

    data = response.json()

    # Check required top-level fields
    required_fields = ["forecast", "buckets", "notes", "inputs", "metadata"]
    for field in required_fields:
        assert field in data

    # Validate forecast structure
    forecast = data["forecast"]
    assert isinstance(forecast, list)
    assert len(forecast) == 14  # 14-day forecast

    # Validate each forecast day
    for i, day in enumerate(forecast):
        assert day["day"] == i + 1
        assert "date" in day
        assert "inflow" in day
        assert "outflow" in day
        assert "net" in day
        assert "balance" in day
        assert "risk_level" in day

        # Validate data types
        assert isinstance(day["inflow"], (int, float))
        assert isinstance(day["outflow"], (int, float))
        assert isinstance(day["net"], (int, float))
        assert isinstance(day["balance"], (int, float))
        assert isinstance(day["risk_level"], str)
        assert day["risk_level"] in ["low", "medium", "high", "critical"]

    # Validate buckets structure
    buckets = data["buckets"]
    required_buckets = ["reserve", "operating", "vendor"]
    for bucket_name in required_buckets:
        assert bucket_name in buckets
        bucket = buckets[bucket_name]

        assert "amount" in bucket
        assert "ratio" in bucket
        assert "purpose" in bucket

        assert isinstance(bucket["amount"], (int, float))
        assert isinstance(bucket["ratio"], (int, float))
        assert isinstance(bucket["purpose"], str)
        assert bucket["amount"] >= 0
        assert 0 <= bucket["ratio"] <= 1

    # Validate notes
    assert isinstance(data["notes"], str)
    assert len(data["notes"]) > 0

    # Validate inputs
    inputs = data["inputs"]
    assert "current_balance" in inputs
    assert "risk_tolerance" in inputs
    assert "forecast_days" in inputs
    assert inputs["forecast_days"] == 14

    # Validate metadata
    metadata = data["metadata"]
    assert "generated_at" in metadata
    assert "model_version" in metadata
    assert "seed" in metadata
    assert metadata["seed"] == 42


def test_treasury_plan_api_minimal_request(client: TestClient) -> None:
    """Test API with minimal required fields."""
    minimal_request = {"current_balance": 100000.0}

    response = client.post("/treasury/plan", json=minimal_request)
    assert response.status_code == 200

    data = response.json()

    # Should still have all required fields
    required_fields = ["forecast", "buckets", "notes", "inputs", "metadata"]
    for field in required_fields:
        assert field in data

    # Should use defaults
    assert data["inputs"]["risk_tolerance"] == "medium"
    assert len(data["forecast"]) == 14


def test_treasury_plan_api_validation_current_balance(client: TestClient) -> None:
    """Test API validation for current_balance field."""
    # Test negative balance
    invalid_request = {"current_balance": -1000.0, "risk_tolerance": "medium"}

    response = client.post("/treasury/plan", json=invalid_request)
    assert response.status_code == 422  # Validation error

    # Test zero balance
    zero_request = {"current_balance": 0.0, "risk_tolerance": "medium"}

    response = client.post("/treasury/plan", json=zero_request)
    assert response.status_code == 200  # Should be valid


def test_treasury_plan_api_validation_risk_tolerance(client: TestClient) -> None:
    """Test API validation for risk_tolerance field."""
    valid_risks = ["low", "medium", "high"]

    for risk in valid_risks:
        request = {"current_balance": 100000.0, "risk_tolerance": risk}

        response = client.post("/treasury/plan", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["inputs"]["risk_tolerance"] == risk


def test_treasury_plan_api_forecast_balance_consistency(
    client: TestClient, valid_treasury_request: dict[str, Any]
) -> None:
    """Test that forecast balances are calculated correctly."""
    response = client.post("/treasury/plan", json=valid_treasury_request)
    assert response.status_code == 200

    data = response.json()
    forecast = data["forecast"]
    current_balance = valid_treasury_request["current_balance"]

    # First day should start with current balance + first day's net
    expected_first_balance = current_balance + forecast[0]["net"]
    assert abs(forecast[0]["balance"] - expected_first_balance) < 0.01

    # Each subsequent day should be previous balance + net
    for i in range(1, len(forecast)):
        expected_balance = forecast[i - 1]["balance"] + forecast[i]["net"]
        assert (
            abs(forecast[i]["balance"] - expected_balance) < 0.01
        )  # Allow for rounding


def test_treasury_plan_api_forecast_net_calculation(
    client: TestClient, valid_treasury_request: dict[str, Any]
) -> None:
    """Test that net calculation is correct."""
    response = client.post("/treasury/plan", json=valid_treasury_request)
    assert response.status_code == 200

    data = response.json()
    forecast = data["forecast"]

    # Each day's net should be inflow - outflow
    for day in forecast:
        expected_net = day["inflow"] - day["outflow"]
        assert abs(day["net"] - expected_net) < 0.01  # Allow for rounding


def test_treasury_plan_api_bucket_allocation_consistency(
    client: TestClient, valid_treasury_request: dict[str, Any]
) -> None:
    """Test that bucket allocation is consistent."""
    response = client.post("/treasury/plan", json=valid_treasury_request)
    assert response.status_code == 200

    data = response.json()
    buckets = data["buckets"]
    current_balance = valid_treasury_request["current_balance"]

    # Total allocated should not exceed current balance
    total_allocated = (
        buckets["reserve"]["amount"]
        + buckets["operating"]["amount"]
        + buckets["vendor"]["amount"]
    )
    assert total_allocated <= current_balance

    # Ratios should sum to approximately 1.0
    total_ratio = (
        buckets["reserve"]["ratio"]
        + buckets["operating"]["ratio"]
        + buckets["vendor"]["ratio"]
    )
    assert abs(total_ratio - 1.0) < 0.01  # Allow for rounding


def test_treasury_plan_api_risk_tolerance_impact(client: TestClient) -> None:
    """Test that risk tolerance affects bucket allocation."""
    base_request = {"current_balance": 200000.0}

    risk_results = {}
    for risk in ["low", "medium", "high"]:
        request = {**base_request, "risk_tolerance": risk}
        response = client.post("/treasury/plan", json=request)
        assert response.status_code == 200

        data = response.json()
        risk_results[risk] = data["buckets"]["reserve"]["ratio"]

    # Low risk should have highest reserve ratio
    assert risk_results["low"] > risk_results["medium"]
    assert risk_results["medium"] > risk_results["high"]


def test_treasury_plan_api_response_shape_consistency(client: TestClient) -> None:
    """Test that response shape is consistent across different inputs."""
    test_cases = [
        {"current_balance": 50000.0, "risk_tolerance": "low"},
        {"current_balance": 150000.0, "risk_tolerance": "medium"},
        {"current_balance": 500000.0, "risk_tolerance": "high"},
    ]

    for test_case in test_cases:
        response = client.post("/treasury/plan", json=test_case)
        assert response.status_code == 200

        data = response.json()

        # All responses should have same structure
        assert all(
            field in data
            for field in ["forecast", "buckets", "notes", "inputs", "metadata"]
        )

        # Forecast should always have 14 days
        assert len(data["forecast"]) == 14

        # Buckets should always have same structure
        for bucket_name in ["reserve", "operating", "vendor"]:
            assert bucket_name in data["buckets"]
            bucket = data["buckets"][bucket_name]
            assert all(field in bucket for field in ["amount", "ratio", "purpose"])


def test_treasury_plan_api_error_handling(client: TestClient) -> None:
    """Test API error handling for invalid requests."""
    # Test missing required field
    invalid_request = {
        "risk_tolerance": "medium"
        # Missing current_balance
    }

    response = client.post("/treasury/plan", json=invalid_request)
    assert response.status_code == 422  # Validation error

    # Test invalid JSON
    response = client.post(
        "/treasury/plan",
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_treasury_plan_api_deterministic_response(
    client: TestClient, valid_treasury_request: dict[str, Any]
) -> None:
    """Test that identical requests produce identical responses."""
    # Make multiple requests with same parameters
    responses = []
    for _ in range(3):
        response = client.post("/treasury/plan", json=valid_treasury_request)
        assert response.status_code == 200
        responses.append(response.json())

    # All responses should be identical
    first_response = responses[0]
    for response in responses[1:]:
        assert response["forecast"] == first_response["forecast"]
        assert response["buckets"] == first_response["buckets"]
        assert response["notes"] == first_response["notes"]
        assert response["inputs"] == first_response["inputs"]
        # Metadata timestamps may differ, but other fields should be same
        assert (
            response["metadata"]["model_version"]
            == first_response["metadata"]["model_version"]
        )
        assert response["metadata"]["seed"] == first_response["metadata"]["seed"]


def test_treasury_plan_api_forecast_risk_levels(
    client: TestClient, valid_treasury_request: dict[str, Any]
) -> None:
    """Test that forecast risk levels are properly calculated."""
    response = client.post("/treasury/plan", json=valid_treasury_request)
    assert response.status_code == 200

    data = response.json()
    forecast = data["forecast"]

    # All days should have valid risk levels
    valid_risk_levels = ["low", "medium", "high", "critical"]
    for day in forecast:
        assert day["risk_level"] in valid_risk_levels

        # Risk level should correlate with balance and outflow
        balance = day["balance"]
        outflow = day["outflow"]

        if balance <= 0:
            assert day["risk_level"] == "critical"
        elif balance < outflow * 3:
            assert day["risk_level"] in ["high", "critical"]
        elif balance < outflow * 7:
            assert day["risk_level"] in ["medium", "high", "critical"]
        else:
            assert day["risk_level"] == "low"
