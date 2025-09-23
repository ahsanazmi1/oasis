"""
Smoke tests for MCP (Model Context Protocol) endpoints.
"""

from fastapi.testclient import TestClient

from oasis.api import app

client = TestClient(app)


def test_mcp_get_status() -> None:
    """Test MCP getStatus verb returns expected response."""
    response = client.post("/mcp/invoke", json={"verb": "getStatus", "args": {}})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["ok"] is True
    assert data["data"]["agent"] == "oasis"


def test_mcp_get_treasury_plan() -> None:
    """Test MCP getTreasuryPlan verb returns expected response."""
    response = client.post("/mcp/invoke", json={"verb": "getTreasuryPlan", "args": {}})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    treasury_data = data["data"]
    assert treasury_data["plan_id"] == "oasis-treasury-v1"
    assert treasury_data["plan_name"] == "Oasis Conservative Treasury Plan"
    assert treasury_data["risk_tolerance"] == "conservative"
    assert "allocation" in treasury_data
    assert "rebalancing_frequency" in treasury_data


def test_mcp_invalid_verb() -> None:
    """Test MCP endpoint returns 400 for invalid verb."""
    response = client.post("/mcp/invoke", json={"verb": "invalidVerb", "args": {}})

    assert response.status_code == 400
    data = response.json()
    assert "Unsupported verb" in data["detail"]


def test_mcp_missing_verb() -> None:
    """Test MCP endpoint returns validation error for missing verb."""
    response = client.post("/mcp/invoke", json={"args": {}})

    assert response.status_code == 422  # Validation error


def test_mcp_deterministic_responses() -> None:
    """Test that MCP responses are deterministic."""
    # Test getStatus multiple times
    responses = []
    for _ in range(3):
        response = client.post("/mcp/invoke", json={"verb": "getStatus", "args": {}})
        assert response.status_code == 200
        responses.append(response.json())

    # All responses should be identical
    assert all(resp == responses[0] for resp in responses)

    # Test getTreasuryPlan multiple times
    treasury_responses = []
    for _ in range(3):
        response = client.post(
            "/mcp/invoke", json={"verb": "getTreasuryPlan", "args": {}}
        )
        assert response.status_code == 200
        treasury_responses.append(response.json())

    # All treasury responses should be identical
    assert all(resp == treasury_responses[0] for resp in treasury_responses)
