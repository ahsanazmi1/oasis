# Oasis Service

[![CI](https://github.com/ahsanazmi1/oasis/workflows/CI/badge.svg)](https://github.com/ahsanazmi1/oasis/actions/workflows/ci.yml)
[![Contracts](https://github.com/ahsanazmi1/oasis/workflows/Contracts/badge.svg)](https://github.com/ahsanazmi1/oasis/actions/workflows/contracts.yml)
[![Security](https://github.com/ahsanazmi1/oasis/workflows/Security/badge.svg)](https://github.com/ahsanazmi1/oasis/actions/workflows/security.yml)

**Oasis** is the **Treasury and Liquidity service** for the [Open Checkout Network (OCN)](https://github.com/ahsanazmi1/ocn-common).

## Phase 4 — Payment Instruction & Visibility

🚧 **Currently in development** - Phase 4 focuses on payment instruction generation, settlement visibility, and comprehensive payment tracking for treasury operations.

- **Status**: Active development on `phase-4-instruction` branch
- **Features**: Payment instruction schemas, settlement visibility, payment tracking, instruction validation
- **Issue Tracker**: [Phase 4 Issues](https://github.com/ahsanazmi1/oasis/issues?q=is%3Aopen+is%3Aissue+label%3Aphase-4)
- **Timeline**: Weeks 12-16 of OCN development roadmap

See [CHANGELOG.md](CHANGELOG.md) for detailed Phase 4 progress and features.

Oasis provides intelligent treasury management and liquidity planning for the OCN ecosystem. Unlike traditional black-box treasury systems, Oasis offers:

## Quickstart (≤ 60s)

Get up and running with Oasis OCN Agent in under a minute:

```bash
# Clone the repository
git clone https://github.com/ahsanazmi1/oasis.git
cd oasis

# Setup everything (venv, deps, pre-commit hooks)
make setup

# Run tests to verify everything works
make test

# Start the service
make run
```

**That's it!** 🎉

The service will be running at `http://localhost:8000`. Test the endpoints:

```bash
# Health check
curl http://localhost:8000/health

# MCP getStatus
curl -X POST http://localhost:8000/mcp/invoke \
  -H "Content-Type: application/json" \
  -d '{"verb": "getStatus", "args": {}}'

# MCP getTreasuryPlan
curl -X POST http://localhost:8000/mcp/invoke \
  -H "Content-Type: application/json" \
  -d '{"verb": "getTreasuryPlan", "args": {}}'
```

### Additional Makefile Targets

```bash
make lint        # Run code quality checks
make fmt         # Format code with black/ruff
make clean       # Remove virtual environment and cache
make help        # Show all available targets
```

## Manual Setup (Alternative)

If you prefer manual setup over the Makefile:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest -q

# Start the service
uvicorn oasis.api:app --reload
```

## API Endpoints

### Core Endpoints
- `GET /health` - Health check endpoint

### Treasury Planning
- `POST /treasury/plan` - Generate deterministic treasury plan with 14-day liquidity forecast

#### Treasury Planning Request Example
```bash
curl -X POST http://localhost:8000/treasury/plan \
  -H "Content-Type: application/json" \
  -d '{
    "current_balance": 150000.0,
    "risk_tolerance": "medium",
    "expected_inflows": {
      "multiplier": 1.2,
      "events": [
        {"day": 3, "amount": 25000},
        {"day": 7, "amount": 15000}
      ]
    },
    "expected_outflows": {
      "multiplier": 0.9,
      "events": [
        {"day": 5, "amount": 10000}
      ]
    },
    "vendor_payment_schedule": {
      "upcoming_payments": [8000, 5000, 3000]
    }
  }'
```

#### Treasury Planning Response Example
```json
{
  "forecast": [
    {
      "day": 1,
      "date": "2024-01-15",
      "inflow": 12000.0,
      "outflow": 9000.0,
      "net": 3000.0,
      "balance": 153000.0,
      "risk_level": "low"
    }
  ],
  "buckets": {
    "reserve": {
      "amount": 22500.0,
      "ratio": 0.15,
      "purpose": "Emergency liquidity and regulatory requirements",
      "target_min": 15000.0
    },
    "operating": {
      "amount": 105000.0,
      "ratio": 0.70,
      "purpose": "Daily operations and working capital",
      "target_min": 75000.0
    },
    "vendor": {
      "amount": 22500.0,
      "ratio": 0.15,
      "purpose": "Vendor payments and scheduled obligations",
      "upcoming_commitments": 16000.0
    }
  },
  "notes": "Treasury planning analysis for $150,000.00 current balance:\n\n• 14-day forecast shows balance range: $145,000.00 - $165,000.00\n• Average projected balance: $155,000.00\n• Risk assessment: Low risk period - adequate liquidity maintained\n\nBucket allocation (risk tolerance: medium):\n• Reserve: $22,500.00 (15.0%)\n• Operating: $105,000.00 (70.0%)\n• Vendor: $22,500.00 (15.0%)\n\n✅ Recommendation: Current allocation appears optimal for risk tolerance",
  "inputs": {
    "current_balance": 150000.0,
    "risk_tolerance": "medium",
    "forecast_days": 14
  },
  "metadata": {
    "generated_at": "2024-01-15T10:30:00Z",
    "model_version": "1.0.0",
    "seed": 42
  }
}
```

### MCP (Model Context Protocol)
- `POST /mcp/invoke` - MCP protocol endpoint for Oasis service operations
  - `getStatus` - Get the current status of the Oasis agent
  - `getTreasuryPlan` - Generate deterministic treasury plan with liquidity forecast

## Development

This project uses:
- **FastAPI** for the web framework
- **pytest** for testing
- **ruff** and **black** for code formatting
- **mypy** for type checking

### Pre-commit Hooks

Install and run pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

The hooks include:
- **ruff** - Fast Python linter and formatter
- **black** - Code formatting
- **end-of-file-fixer** - Ensures files end with newlines
- **trailing-whitespace** - Removes trailing whitespace
- **check-yaml** - Validates YAML files
- **check-json** - Validates JSON files
- **check-toml** - Validates TOML files

## Phase 3 — Negotiation & Live Fee Bidding

Treasury constraints influence negotiation.

### Phase 3 — Negotiation & Live Fee Bidding
- [ ] Feeds liquidity/treasury constraints into negotiation logic
- [ ] Constraints visible via CE (ocn.oasis.constraint.v1)
- [ ] Tests validating constraint-influenced outcomes

## License

MIT License - see [LICENSE](LICENSE) file for details.
