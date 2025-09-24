"""
MCP (Model Context Protocol) server implementation for Oasis service.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from oasis.planner import plan_treasury, validate_planning_inputs


class MCPRequest(BaseModel):
    """MCP request model."""

    verb: str
    args: dict[str, Any] = {}


class MCPResponse(BaseModel):
    """MCP response model."""

    success: bool = True
    data: dict[str, Any] = {}


# Create MCP router
mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])


@mcp_router.post("/invoke")
async def invoke_mcp(request: MCPRequest) -> MCPResponse:
    """
    MCP protocol endpoint for Oasis service operations.

    Supported verbs:
    - getStatus: Get the current status of the Oasis agent
    - getTreasuryPlan: Generate deterministic treasury plan with liquidity forecast
    """
    if request.verb == "getStatus":
        return MCPResponse(data={"ok": True, "agent": "oasis"})
    elif request.verb == "getTreasuryPlan":
        try:
            # Extract planning inputs from args
            planning_inputs = {
                "current_balance": request.args.get("current_balance", 100000.0),
                "risk_tolerance": request.args.get("risk_tolerance", "medium"),
                "expected_inflows": request.args.get("expected_inflows", {}),
                "expected_outflows": request.args.get("expected_outflows", {}),
                "vendor_payment_schedule": request.args.get(
                    "vendor_payment_schedule", {}
                ),
            }

            # Validate inputs
            validated_inputs = validate_planning_inputs(planning_inputs)

            # Generate treasury plan
            plan_result = plan_treasury(validated_inputs)

            return MCPResponse(
                data={
                    "plan_id": "oasis-treasury-plan-v1",
                    "forecast": plan_result["forecast"],
                    "buckets": plan_result["buckets"],
                    "notes": plan_result["notes"],
                    "inputs": plan_result["inputs"],
                    "metadata": plan_result["metadata"],
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating treasury plan: {str(e)}"
            ) from e
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported verb: {request.verb}. Supported verbs: getStatus, getTreasuryPlan",
        )
