"""
FastAPI application for Oasis service.
"""

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Add the project root to the Python path to import mcp
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server import mcp_router
from oasis.planner import plan_treasury, validate_planning_inputs

# Create FastAPI application
app = FastAPI(
    title="Oasis Service",
    description="Oasis service for the Open Checkout Network (OCN)",
    version="0.1.0",
    contact={
        "name": "OCN Team",
        "email": "team@ocn.ai",
        "url": "https://github.com/ahsanazmi1/oasis",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Include MCP router
app.include_router(mcp_router)


class TreasuryPlanningRequest(BaseModel):
    """Treasury planning request model."""

    current_balance: float = Field(..., ge=0, description="Current treasury balance")
    risk_tolerance: str = Field(
        "medium", description="Risk tolerance: low, medium, high"
    )
    expected_inflows: dict[str, Any] = Field(
        default_factory=dict, description="Expected inflow patterns"
    )
    expected_outflows: dict[str, Any] = Field(
        default_factory=dict, description="Expected outflow patterns"
    )
    vendor_payment_schedule: dict[str, Any] = Field(
        default_factory=dict, description="Vendor payment schedule"
    )


class TreasuryPlanningResponse(BaseModel):
    """Treasury planning response model."""

    forecast: list[dict[str, Any]] = Field(..., description="14-day liquidity forecast")
    buckets: dict[str, Any] = Field(..., description="Treasury bucket allocation")
    notes: str = Field(..., description="Planning analysis and recommendations")
    inputs: dict[str, Any] = Field(..., description="Validated planning inputs")
    metadata: dict[str, Any] = Field(..., description="Planning metadata")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {"ok": True, "repo": "oasis"}


@app.post("/treasury/plan", response_model=TreasuryPlanningResponse)
async def plan_treasury_endpoint(
    request: TreasuryPlanningRequest,
) -> TreasuryPlanningResponse:
    """
    Generate deterministic treasury plan with 14-day liquidity forecast.

    Args:
        request: Treasury planning request with balance, risk tolerance, and flows

    Returns:
        Treasury planning response with forecast, buckets, and recommendations
    """
    try:
        # Validate and normalize inputs
        validated_inputs = validate_planning_inputs(request.model_dump())

        # Generate treasury plan
        plan_result = plan_treasury(validated_inputs)

        # Create response
        response = TreasuryPlanningResponse(
            forecast=plan_result["forecast"],
            buckets=plan_result["buckets"],
            notes=plan_result["notes"],
            inputs=plan_result["inputs"],
            metadata=plan_result["metadata"],
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating treasury plan: {str(e)}",
        ) from e


def main() -> None:
    """Main entry point for running the application."""
    import uvicorn

    uvicorn.run(
        "oasis.api:app",
        host="127.0.0.1",  # Use localhost for development security
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
