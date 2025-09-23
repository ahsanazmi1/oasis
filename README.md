# Oasis Service

[![CI](https://github.com/ahsanazmi1/oasis/workflows/CI/badge.svg)](https://github.com/ahsanazmi1/oasis/actions/workflows/ci.yml)
[![Contracts](https://github.com/ahsanazmi1/oasis/workflows/Contracts/badge.svg)](https://github.com/ahsanazmi1/oasis/actions/workflows/contracts.yml)
[![Security](https://github.com/ahsanazmi1/oasis/workflows/Security/badge.svg)](https://github.com/ahsanazmi1/oasis/actions/workflows/security.yml)

Oasis is a minimal Python service for the [Open Checkout Network (OCN)](https://github.com/ahsanazmi1/ocn-common). It provides core functionality and serves as a foundation for building more complex services within the OCN ecosystem. Oasis follows modern Python development practices with FastAPI, comprehensive testing, and automated CI/CD workflows.

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

### MCP (Model Context Protocol)
- `POST /mcp/invoke` - MCP protocol endpoint for Oasis service operations
  - `getStatus` - Get the current status of the Oasis agent
  - `getTreasuryPlan` - Get treasury management plan and configuration

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

## License

MIT License - see [LICENSE](LICENSE) file for details.
