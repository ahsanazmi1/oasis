"""
MCP (Model Context Protocol) server implementation for Oasis service.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


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
    - getTreasuryPlan: Get treasury management plan and configuration
    """
    if request.verb == "getStatus":
        return MCPResponse(data={"ok": True, "agent": "oasis"})
    elif request.verb == "getTreasuryPlan":
        return MCPResponse(
            data={
                "plan_id": "oasis-treasury-v1",
                "plan_name": "Oasis Conservative Treasury Plan",
                "risk_tolerance": "conservative",
                "allocation": {
                    "cash": 0.4,
                    "bonds": 0.35,
                    "equities": 0.2,
                    "alternative_investments": 0.05,
                },
                "rebalancing_frequency": "quarterly",
                "minimum_balance": 10000,
                "maximum_withdrawal_rate": 0.04,
                "emergency_reserve_months": 6,
            }
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported verb: {request.verb}. Supported verbs: getStatus, getTreasuryPlan",
        )
