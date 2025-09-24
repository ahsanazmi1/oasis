"""
Oasis treasury planning module.

Provides deterministic treasury planning and liquidity forecasting.
"""

import random
from datetime import datetime, timedelta
from typing import Any

# Treasury planning constants
FORECAST_DAYS = 14
SEED_VALUE = 42
MIN_RESERVE_RATIO = 0.15  # 15% minimum reserve
OPERATING_RATIO = 0.70  # 70% for operations
VENDOR_RATIO = 0.15  # 15% for vendor payments

# Base daily patterns (deterministic)
BASE_INFLOW_PATTERN = [
    10000,
    12000,
    8000,
    15000,
    9000,
    11000,
    13000,
    14000,
    7000,
    16000,
    10000,
    12000,
    8500,
    14500,
]
BASE_OUTFLOW_PATTERN = [
    8000,
    10000,
    6000,
    12000,
    7000,
    9000,
    11000,
    10000,
    5000,
    13000,
    8000,
    10000,
    6500,
    12000,
]


def plan_treasury(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Generate deterministic treasury plan with 14-day liquidity forecast.

    Args:
        inputs: Treasury planning inputs containing current_balance,
                expected_inflows, expected_outflows, etc.

    Returns:
        Dictionary with forecast, buckets, and notes
    """
    # Set deterministic seed for reproducible results
    random.seed(SEED_VALUE)

    # Extract and validate inputs
    current_balance = float(inputs.get("current_balance", 100000.0))
    expected_inflows = inputs.get("expected_inflows", {})
    expected_outflows = inputs.get("expected_outflows", {})
    risk_tolerance = inputs.get("risk_tolerance", "medium")
    vendor_payment_schedule = inputs.get("vendor_payment_schedule", {})

    # Generate 14-day forecast
    forecast = _generate_forecast(current_balance, expected_inflows, expected_outflows)

    # Calculate optimal bucket allocation
    buckets = _calculate_buckets(
        current_balance, risk_tolerance, vendor_payment_schedule
    )

    # Generate planning notes
    notes = _generate_notes(current_balance, forecast, buckets, risk_tolerance)

    return {
        "forecast": forecast,
        "buckets": buckets,
        "notes": notes,
        "inputs": {
            "current_balance": current_balance,
            "risk_tolerance": risk_tolerance,
            "forecast_days": FORECAST_DAYS,
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model_version": "1.0.0",
            "seed": SEED_VALUE,
        },
    }


def _generate_forecast(
    current_balance: float,
    expected_inflows: dict[str, Any],
    expected_outflows: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate 14-day liquidity forecast."""
    forecast = []
    running_balance = current_balance

    # Apply inflow/outflow adjustments based on inputs
    inflow_multiplier = expected_inflows.get("multiplier", 1.0)
    outflow_multiplier = expected_outflows.get("multiplier", 1.0)

    # Add specific inflow/outflow events
    inflow_events = expected_inflows.get("events", [])
    outflow_events = expected_outflows.get("events", [])

    for day in range(FORECAST_DAYS):
        date = datetime.now() + timedelta(days=day)

        # Base daily amounts with deterministic variation
        base_inflow = BASE_INFLOW_PATTERN[day % len(BASE_INFLOW_PATTERN)]
        base_outflow = BASE_OUTFLOW_PATTERN[day % len(BASE_OUTFLOW_PATTERN)]

        # Apply multipliers
        daily_inflow = base_inflow * inflow_multiplier
        daily_outflow = base_outflow * outflow_multiplier

        # Add specific events for this day
        for event in inflow_events:
            if event.get("day", 0) == day:
                daily_inflow += event.get("amount", 0)

        for event in outflow_events:
            if event.get("day", 0) == day:
                daily_outflow += event.get("amount", 0)

        # Calculate daily net and running balance
        daily_net = daily_inflow - daily_outflow
        running_balance += daily_net

        forecast_day = {
            "day": day + 1,
            "date": date.strftime("%Y-%m-%d"),
            "inflow": round(daily_inflow, 2),
            "outflow": round(daily_outflow, 2),
            "net": round(daily_net, 2),
            "balance": round(running_balance, 2),
            "risk_level": _assess_daily_risk(running_balance, daily_outflow),
        }

        forecast.append(forecast_day)

    return forecast


def _calculate_buckets(
    current_balance: float, risk_tolerance: str, vendor_payment_schedule: dict[str, Any]
) -> dict[str, Any]:
    """Calculate optimal bucket allocation based on balance and risk tolerance."""

    # Adjust ratios based on risk tolerance
    if risk_tolerance == "low":
        reserve_ratio = 0.25  # Higher reserves for low risk
        operating_ratio = 0.60
        vendor_ratio = 0.15
    elif risk_tolerance == "high":
        reserve_ratio = 0.10  # Lower reserves for high risk
        operating_ratio = 0.75
        vendor_ratio = 0.15
    else:  # medium risk
        reserve_ratio = MIN_RESERVE_RATIO
        operating_ratio = OPERATING_RATIO
        vendor_ratio = VENDOR_RATIO

    # Calculate bucket amounts
    reserve_amount = current_balance * reserve_ratio
    operating_amount = current_balance * operating_ratio
    vendor_amount = current_balance * vendor_ratio

    # Adjust vendor bucket based on payment schedule
    total_vendor_commitments = sum(vendor_payment_schedule.get("upcoming_payments", []))
    if total_vendor_commitments > 0:
        vendor_amount = max(vendor_amount, total_vendor_commitments * 1.1)  # 10% buffer

    # Ensure buckets don't exceed total balance
    total_allocated = reserve_amount + operating_amount + vendor_amount
    if total_allocated > current_balance:
        # Scale down proportionally
        scale_factor = current_balance / total_allocated
        reserve_amount *= scale_factor
        operating_amount *= scale_factor
        vendor_amount *= scale_factor

    return {
        "reserve": {
            "amount": round(reserve_amount, 2),
            "ratio": round(reserve_ratio, 3),
            "purpose": "Emergency liquidity and regulatory requirements",
            "target_min": round(current_balance * 0.10, 2),
        },
        "operating": {
            "amount": round(operating_amount, 2),
            "ratio": round(operating_ratio, 3),
            "purpose": "Daily operations and working capital",
            "target_min": round(current_balance * 0.50, 2),
        },
        "vendor": {
            "amount": round(vendor_amount, 2),
            "ratio": round(vendor_ratio, 3),
            "purpose": "Vendor payments and scheduled obligations",
            "upcoming_commitments": total_vendor_commitments,
        },
    }


def _assess_daily_risk(balance: float, outflow: float) -> str:
    """Assess daily risk level based on balance and outflow."""
    if balance <= 0:
        return "critical"
    elif balance < outflow * 3:  # Less than 3 days of coverage
        return "high"
    elif balance < outflow * 7:  # Less than 1 week of coverage
        return "medium"
    else:
        return "low"


def _generate_notes(
    current_balance: float,
    forecast: list[dict[str, Any]],
    buckets: dict[str, Any],
    risk_tolerance: str,
) -> str:
    """Generate human-readable planning notes."""

    # Analyze forecast
    min_balance = min(day["balance"] for day in forecast)
    max_balance = max(day["balance"] for day in forecast)
    avg_balance = sum(day["balance"] for day in forecast) / len(forecast)

    high_risk_days = sum(
        1 for day in forecast if day["risk_level"] in ["high", "critical"]
    )

    notes = (
        f"Treasury planning analysis for ${current_balance:,.2f} current balance:\n\n"
    )

    # Balance analysis
    notes += f"• 14-day forecast shows balance range: ${min_balance:,.2f} - ${max_balance:,.2f}\n"
    notes += f"• Average projected balance: ${avg_balance:,.2f}\n"

    # Risk assessment
    if high_risk_days == 0:
        notes += "• Risk assessment: Low risk period - adequate liquidity maintained\n"
    elif high_risk_days <= 3:
        notes += f"• Risk assessment: Moderate risk - {high_risk_days} high-risk days identified\n"
    else:
        notes += f"• Risk assessment: High risk - {high_risk_days} high-risk days require attention\n"

    # Bucket recommendations
    notes += f"\nBucket allocation (risk tolerance: {risk_tolerance}):\n"
    notes += f"• Reserve: ${buckets['reserve']['amount']:,.2f} ({buckets['reserve']['ratio']:.1%})\n"
    notes += f"• Operating: ${buckets['operating']['amount']:,.2f} ({buckets['operating']['ratio']:.1%})\n"
    notes += f"• Vendor: ${buckets['vendor']['amount']:,.2f} ({buckets['vendor']['ratio']:.1%})\n"

    # Recommendations
    if min_balance < buckets["reserve"]["target_min"]:
        notes += "\n⚠️  Recommendation: Consider increasing reserve buffer or reducing planned outflows\n"
    elif min_balance > buckets["reserve"]["target_min"] * 2:
        notes += "\n💡 Recommendation: Excess liquidity available for investment or debt reduction\n"
    else:
        notes += "\n✅ Recommendation: Current allocation appears optimal for risk tolerance\n"

    return notes


def validate_planning_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize treasury planning inputs.

    Args:
        inputs: Raw planning inputs

    Returns:
        Validated and normalized inputs
    """
    validated = {
        "current_balance": max(0.0, float(inputs.get("current_balance", 100000.0))),
        "risk_tolerance": str(inputs.get("risk_tolerance", "medium")).lower(),
        "expected_inflows": inputs.get("expected_inflows", {}),
        "expected_outflows": inputs.get("expected_outflows", {}),
        "vendor_payment_schedule": inputs.get("vendor_payment_schedule", {}),
    }

    # Validate risk tolerance
    if validated["risk_tolerance"] not in ["low", "medium", "high", "conservative"]:
        validated["risk_tolerance"] = "medium"
    elif validated["risk_tolerance"] == "conservative":
        validated["risk_tolerance"] = "low"  # Map conservative to low risk

    # Validate inflow/outflow multipliers
    if "multiplier" in validated["expected_inflows"]:
        validated["expected_inflows"]["multiplier"] = max(
            0.1, min(5.0, float(validated["expected_inflows"]["multiplier"]))
        )
    else:
        validated["expected_inflows"]["multiplier"] = 1.0

    if "multiplier" in validated["expected_outflows"]:
        validated["expected_outflows"]["multiplier"] = max(
            0.1, min(5.0, float(validated["expected_outflows"]["multiplier"]))
        )
    else:
        validated["expected_outflows"]["multiplier"] = 1.0

    return validated
