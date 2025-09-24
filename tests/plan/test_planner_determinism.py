"""
Tests for treasury planner determinism.
"""

from oasis.planner import plan_treasury, validate_planning_inputs


def test_planner_determinism_same_inputs() -> None:
    """Test that same inputs always produce identical results."""
    inputs = {
        "current_balance": 150000.0,
        "risk_tolerance": "medium",
        "expected_inflows": {"multiplier": 1.2},
        "expected_outflows": {"multiplier": 0.9},
        "vendor_payment_schedule": {"upcoming_payments": [5000, 3000, 2000]},
    }

    # Run planning multiple times with same inputs
    results = []
    for _ in range(3):
        result = plan_treasury(inputs)
        results.append(result)

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result["forecast"] == first_result["forecast"]
        assert result["buckets"] == first_result["buckets"]
        assert result["notes"] == first_result["notes"]
        assert result["inputs"] == first_result["inputs"]


def test_planner_determinism_forecast_series() -> None:
    """Test that forecast series is deterministic for fixed inputs."""
    inputs = {
        "current_balance": 100000.0,
        "risk_tolerance": "low",
        "expected_inflows": {},
        "expected_outflows": {},
    }

    # Generate two forecasts
    result1 = plan_treasury(inputs)
    result2 = plan_treasury(inputs)

    forecast1 = result1["forecast"]
    forecast2 = result2["forecast"]

    # Should have exactly 14 days
    assert len(forecast1) == 14
    assert len(forecast2) == 14

    # Each day should be identical
    for day1, day2 in zip(forecast1, forecast2, strict=False):
        assert day1["day"] == day2["day"]
        assert day1["inflow"] == day2["inflow"]
        assert day1["outflow"] == day2["outflow"]
        assert day1["net"] == day2["net"]
        assert day1["balance"] == day2["balance"]
        assert day1["risk_level"] == day2["risk_level"]


def test_planner_determinism_buckets() -> None:
    """Test that bucket allocation is deterministic."""
    inputs = {
        "current_balance": 200000.0,
        "risk_tolerance": "high",
        "vendor_payment_schedule": {"upcoming_payments": [10000, 5000]},
    }

    # Generate two plans
    result1 = plan_treasury(inputs)
    result2 = plan_treasury(inputs)

    buckets1 = result1["buckets"]
    buckets2 = result2["buckets"]

    # All bucket amounts should be identical
    for bucket_name in ["reserve", "operating", "vendor"]:
        assert buckets1[bucket_name]["amount"] == buckets2[bucket_name]["amount"]
        assert buckets1[bucket_name]["ratio"] == buckets2[bucket_name]["ratio"]


def test_planner_determinism_with_events() -> None:
    """Test determinism with specific inflow/outflow events."""
    inputs = {
        "current_balance": 50000.0,
        "risk_tolerance": "medium",
        "expected_inflows": {
            "multiplier": 1.5,
            "events": [{"day": 3, "amount": 25000}, {"day": 7, "amount": 15000}],
        },
        "expected_outflows": {
            "multiplier": 0.8,
            "events": [{"day": 5, "amount": 10000}, {"day": 10, "amount": 8000}],
        },
    }

    # Generate two plans with events
    result1 = plan_treasury(inputs)
    result2 = plan_treasury(inputs)

    forecast1 = result1["forecast"]
    forecast2 = result2["forecast"]

    # Check specific days with events
    assert forecast1[2]["inflow"] == forecast2[2]["inflow"]  # Day 3 inflow event
    assert forecast1[6]["inflow"] == forecast2[6]["inflow"]  # Day 7 inflow event
    assert forecast1[4]["outflow"] == forecast2[4]["outflow"]  # Day 5 outflow event
    assert forecast1[9]["outflow"] == forecast2[9]["outflow"]  # Day 10 outflow event

    # All balances should be identical
    for day1, day2 in zip(forecast1, forecast2, strict=False):
        assert day1["balance"] == day2["balance"]


def test_planner_determinism_different_risk_tolerances() -> None:
    """Test that different risk tolerances produce different but deterministic results."""
    base_inputs = {
        "current_balance": 100000.0,
        "expected_inflows": {},
        "expected_outflows": {},
    }

    # Test different risk tolerances
    risk_levels = ["low", "medium", "high"]
    results = {}

    for risk in risk_levels:
        inputs = {**base_inputs, "risk_tolerance": risk}
        result1 = plan_treasury(inputs)
        result2 = plan_treasury(inputs)

        # Same risk tolerance should produce identical results
        assert result1["buckets"] == result2["buckets"]
        results[risk] = result1

    # Different risk tolerances should produce different bucket allocations
    low_reserve = results["low"]["buckets"]["reserve"]["ratio"]
    medium_reserve = results["medium"]["buckets"]["reserve"]["ratio"]
    high_reserve = results["high"]["buckets"]["reserve"]["ratio"]

    # Low risk should have highest reserve ratio
    assert low_reserve > medium_reserve
    assert medium_reserve > high_reserve


def test_planner_determinism_balance_variations() -> None:
    """Test determinism across different balance amounts."""
    balances = [50000.0, 100000.0, 250000.0, 500000.0]

    for balance in balances:
        inputs = {
            "current_balance": balance,
            "risk_tolerance": "medium",
            "expected_inflows": {},
            "expected_outflows": {},
        }

        # Generate two plans with same balance
        result1 = plan_treasury(inputs)
        result2 = plan_treasury(inputs)

        # Should be identical
        assert result1["buckets"] == result2["buckets"]
        assert result1["forecast"] == result2["forecast"]

        # Bucket amounts should scale with balance
        total_allocated = (
            result1["buckets"]["reserve"]["amount"]
            + result1["buckets"]["operating"]["amount"]
            + result1["buckets"]["vendor"]["amount"]
        )
        assert abs(total_allocated - balance) < 0.01  # Allow for rounding


def test_planner_determinism_forecast_length() -> None:
    """Test that forecast always has exactly 14 days."""
    test_cases = [
        {"current_balance": 100000.0, "risk_tolerance": "low"},
        {"current_balance": 50000.0, "risk_tolerance": "high"},
        {"current_balance": 1000000.0, "risk_tolerance": "medium"},
    ]

    for inputs in test_cases:
        result = plan_treasury(inputs)
        forecast = result["forecast"]

        assert len(forecast) == 14

        # Each day should have correct structure
        for i, day in enumerate(forecast):
            assert day["day"] == i + 1
            assert "date" in day
            assert "inflow" in day
            assert "outflow" in day
            assert "net" in day
            assert "balance" in day
            assert "risk_level" in day
            assert day["risk_level"] in ["low", "medium", "high", "critical"]


def test_planner_determinism_metadata_consistency() -> None:
    """Test that metadata is consistent across runs."""
    inputs = {"current_balance": 75000.0, "risk_tolerance": "medium"}

    result1 = plan_treasury(inputs)
    result2 = plan_treasury(inputs)

    metadata1 = result1["metadata"]
    metadata2 = result2["metadata"]

    # Model version and seed should be identical
    assert metadata1["model_version"] == metadata2["model_version"]
    assert metadata1["seed"] == metadata2["seed"]

    # Generated timestamps may differ, but other fields should be same
    assert metadata1["seed"] == 42  # Fixed seed value


def test_planner_determinism_notes_consistency() -> None:
    """Test that planning notes are consistent for same inputs."""
    inputs = {
        "current_balance": 120000.0,
        "risk_tolerance": "low",
        "vendor_payment_schedule": {"upcoming_payments": [8000, 5000]},
    }

    result1 = plan_treasury(inputs)
    result2 = plan_treasury(inputs)

    notes1 = result1["notes"]
    notes2 = result2["notes"]

    # Notes should be identical
    assert notes1 == notes2

    # Notes should contain key information
    assert "Treasury planning analysis" in notes1
    assert "14-day forecast" in notes1
    assert "Bucket allocation" in notes1
    assert "risk tolerance: low" in notes1


def test_validate_planning_inputs_determinism() -> None:
    """Test that input validation is deterministic."""
    raw_inputs = {
        "current_balance": "150000",  # String that needs conversion
        "risk_tolerance": "LOW",  # Case that needs normalization
        "expected_inflows": {"multiplier": "2.5"},  # String multiplier
        "expected_outflows": {"multiplier": "1.8"},
        "vendor_payment_schedule": {"upcoming_payments": [1000, 2000, 3000]},
    }

    # Validate multiple times
    validated1 = validate_planning_inputs(raw_inputs)
    validated2 = validate_planning_inputs(raw_inputs)

    # Should be identical
    assert validated1 == validated2

    # Should have correct types
    assert isinstance(validated1["current_balance"], float)
    assert validated1["current_balance"] == 150000.0
    assert validated1["risk_tolerance"] == "low"
    assert validated1["expected_inflows"]["multiplier"] == 2.5
    assert validated1["expected_outflows"]["multiplier"] == 1.8
