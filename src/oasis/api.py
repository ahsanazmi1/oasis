"""
FastAPI application for Oasis service.
"""

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI

# Add the project root to the Python path to import mcp
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server import mcp_router

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


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {"ok": True, "repo": "oasis"}


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
