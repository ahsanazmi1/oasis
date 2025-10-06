# Changelog

All notable changes to this project will be documented in this file.

## v0.3.0 — Phase 3: Negotiation & Live Fee Bidding
- New branch: phase-3-bidding
- Prep for negotiation, bidding, policy DSL, and processor connectors
- README updated with Phase 3 checklist

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Phase 4

### 🚀 Phase 4 — Payment Instruction & Visibility

This release initiates Phase 4 development, focusing on payment instruction generation, settlement visibility, and comprehensive payment tracking for treasury operations.

#### Planned Features
- **Payment Instruction Generation**: Generate standardized payment instructions for treasury operations
- **Settlement Visibility**: Real-time tracking and visibility into treasury settlement processes
- **Payment Tracking**: Comprehensive payment tracking across treasury workflows
- **Instruction Validation**: Enhanced validation for payment instructions and settlement confirmations
- **CloudEvents Integration**: New event types for payment instruction and settlement visibility

#### Timeline
- **Status**: Active development on `phase-4-instruction` branch
- **Timeline**: Weeks 12-16 of OCN development roadmap
- **Issue Tracker**: [Phase 4 Issues](https://github.com/ahsanazmi1/oasis/issues?q=is%3Aopen+is%3Aissue+label%3Aphase-4)

## [Unreleased]

### Added
- Phase 2 — Explainability scaffolding
- PR template for Phase 2 development

## [0.2.0] - 2025-01-25

### 🚀 Phase 2 Complete: Treasury Planning & Explainability

This release completes Phase 2 development, delivering deterministic treasury planning, AI-powered liquidity decision explanations, and production-ready infrastructure for transparent treasury management.

#### Highlights
- **Deterministic Treasury Planning**: Fixed seed (42) for reproducible 14-day liquidity forecasting
- **AI-Powered Liquidity Decisions**: Azure OpenAI integration for human-readable treasury reasoning
- **CloudEvents Integration**: Complete CloudEvent emission for treasury planning decisions with schema validation
- **Production Infrastructure**: Robust CI/CD workflows with security scanning
- **MCP Integration**: Enhanced Model Context Protocol verbs for explainability features

#### Core Features
- **Treasury Planning Engine**: Advanced 14-day liquidity forecasting with deterministic outputs
- **Bucket Allocation**: Operating (70%), reserve (15%), and vendor (15%) allocation management
- **Risk Assessment**: Daily risk level assessment with liquidity management
- **API Endpoints**: RESTful endpoints for treasury planning and liquidity management
- **Event Processing**: Advanced event handling and CloudEvent emission

#### Quality & Infrastructure
- **Test Coverage**: Comprehensive test suite with treasury planning and API validation
- **Security Hardening**: Enhanced security validation and risk assessment
- **CI/CD Pipeline**: Complete GitHub Actions workflow with security scanning
- **Documentation**: Comprehensive API and contract documentation

### Added
- Deterministic treasury planning with fixed seed for reproducible 14-day forecasts
- AI-powered treasury decision explanations with Azure OpenAI integration
- LLM integration for human-readable liquidity reasoning
- Explainability API endpoints for treasury plan decisions
- Decision audit trail with explanations
- CloudEvents integration for treasury planning decisions
- Enhanced MCP verbs for explainability features
- Comprehensive bucket allocation management (operating, reserve, vendor)
- Advanced risk assessment and liquidity management
- Production-ready CI/CD infrastructure

### Changed
- Enhanced treasury planning with deterministic forecasting
- Improved bucket allocation with transparent decision logic
- Streamlined MCP integration for better explainability
- Optimized API performance and accuracy

### Deprecated
- None

### Removed
- None

### Fixed
- Resolved MCP smoke test failures
- Fixed code formatting and type hint issues
- Enhanced error handling and validation
- Improved code quality and consistency

### Security
- Enhanced security validation for treasury planning decisions
- Comprehensive risk assessment and mitigation
- Secure API endpoints with proper authentication
- Robust treasury management security measures

## [Unreleased] — Phase 2

### Added
- AI-powered treasury decision explanations
- LLM integration for human-readable liquidity reasoning
- Explainability API endpoints for treasury plan decisions
- Decision audit trail with explanations
- Integration with Azure OpenAI for explanations
- Enhanced MCP verbs for explainability features

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2024-09-22

### Added
- Initial release
- Health check endpoint at `/health`
- FastAPI application setup
- Basic project structure and documentation
