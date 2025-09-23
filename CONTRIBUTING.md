# Contributing to Oasis

Thank you for your interest in contributing to Oasis! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/oasis.git
   cd oasis
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Run the formatting and linting tools:
```bash
black .
ruff check .
mypy src/
```

## Testing

Run the test suite:
```bash
pytest
```

Make sure all tests pass before submitting a pull request.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Run the linting tools
6. Submit a pull request

## Commit Messages

Use clear, descriptive commit messages. Follow the conventional commit format:
```
feat: add new feature
fix: fix bug
docs: update documentation
test: add tests
```

## Issues

When creating issues, please provide:
- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
