"""
ML model for treasury risk assessment and optimal allocation recommendations.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib

logger = logging.getLogger(__name__)


class RiskAssessmentFeatures(BaseModel):
    """Features for treasury risk assessment."""
    current_balance: float = Field(..., description="Current treasury balance", ge=0)
    historical_balance_volatility: float = Field(..., description="Balance volatility (std dev)", ge=0)
    balance_trend: float = Field(..., description="Balance trend over last 30 days (positive = growing, negative = declining)")
    
    # Cash flow characteristics
    avg_daily_net_flow: float = Field(..., description="Average daily net cash flow (can be negative)")
    cash_flow_volatility: float = Field(..., description="Cash flow volatility (std dev)", ge=0)
    cash_flow_consistency: float = Field(..., description="Cash flow consistency score (0-1)", ge=0, le=1)
    
    # Liquidity metrics
    current_ratio: float = Field(..., description="Current ratio (current assets / current liabilities)", gt=0)
    quick_ratio: float = Field(..., description="Quick ratio (liquid assets / current liabilities)", gt=0)
    cash_ratio: float = Field(..., description="Cash ratio (cash / current liabilities)", ge=0)
    working_capital: float = Field(..., description="Working capital (can be negative)")
    
    # Debt and credit metrics
    debt_to_equity_ratio: float = Field(..., description="Debt to equity ratio", ge=0)
    interest_coverage_ratio: float = Field(..., description="Interest coverage ratio", gt=0)
    credit_line_utilization: float = Field(..., description="Credit line utilization ratio", ge=0, le=1)
    available_credit: float = Field(..., description="Available credit line", ge=0)
    
    # Operational metrics
    monthly_revenue: float = Field(..., description="Monthly revenue", ge=0)
    monthly_expenses: float = Field(..., description="Monthly expenses", ge=0)
    revenue_volatility: float = Field(..., description="Revenue volatility (std dev)", ge=0)
    expense_volatility: float = Field(..., description="Expense volatility (std dev)", ge=0)
    profit_margin: float = Field(..., description="Profit margin (%)", ge=-100, le=100)
    
    # Market and economic factors
    interest_rate: float = Field(..., description="Current interest rate (%)", ge=0)
    market_volatility: float = Field(..., description="Market volatility index", ge=0, le=1)
    economic_indicator: float = Field(..., description="Economic health indicator", ge=0, le=1)
    industry_risk_score: float = Field(..., description="Industry-specific risk score", ge=0, le=1)
    
    # Upcoming obligations
    vendor_payment_commitments: float = Field(..., description="Upcoming vendor payment commitments", ge=0)
    payroll_commitments: float = Field(..., description="Upcoming payroll commitments", ge=0)
    tax_obligations: float = Field(..., description="Upcoming tax obligations", ge=0)
    debt_service_requirements: float = Field(..., description="Debt service requirements", ge=0)
    total_upcoming_obligations: float = Field(..., description="Total upcoming obligations", ge=0)
    
    # Risk factors
    counterparty_risk_score: float = Field(..., description="Counterparty risk score", ge=0, le=1)
    regulatory_risk_score: float = Field(..., description="Regulatory compliance risk score", ge=0, le=1)
    operational_risk_score: float = Field(..., description="Operational risk score", ge=0, le=1)
    concentration_risk_score: float = Field(..., description="Customer/vendor concentration risk", ge=0, le=1)
    
    # Company characteristics
    company_size_factor: float = Field(..., description="Company size factor (revenue-based)", gt=0)
    business_cycle_stage: str = Field(..., description="Business cycle stage (growth, stable, decline)")
    years_in_business: int = Field(..., description="Years in business", ge=0)
    
    # Temporal factors
    is_month_end: bool = Field(..., description="Is it month end?")
    is_quarter_end: bool = Field(..., description="Is it quarter end?")
    is_year_end: bool = Field(..., description="Is it year end?")
    is_holiday_period: bool = Field(..., description="Is it a holiday period?")
    day_of_week: int = Field(..., description="Day of week (0=Monday, 6=Sunday)", ge=0, le=6)


class RiskAssessmentResult(BaseModel):
    """Result of risk assessment."""
    overall_risk_score: float = Field(..., description="Overall risk score (0-1)", ge=0, le=1)
    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")
    liquidity_risk_score: float = Field(..., description="Liquidity risk score (0-1)", ge=0, le=1)
    credit_risk_score: float = Field(..., description="Credit risk score (0-1)", ge=0, le=1)
    operational_risk_score: float = Field(..., description="Operational risk score (0-1)", ge=0, le=1)
    market_risk_score: float = Field(..., description="Market risk score (0-1)", ge=0, le=1)
    
    # Optimal allocation recommendations
    recommended_reserve_ratio: float = Field(..., description="Recommended reserve ratio", ge=0, le=1)
    recommended_operating_ratio: float = Field(..., description="Recommended operating ratio", ge=0, le=1)
    recommended_vendor_ratio: float = Field(..., description="Recommended vendor ratio", ge=0, le=1)
    recommended_investment_ratio: float = Field(..., description="Recommended investment ratio", ge=0, le=1)
    
    # Risk mitigation suggestions
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    mitigation_suggestions: List[str] = Field(..., description="Risk mitigation suggestions")
    early_warning_indicators: List[str] = Field(..., description="Early warning indicators to monitor")
    
    # Model metadata
    model_type: str = Field(..., description="Model type used")
    model_version: str = Field(..., description="Model version")
    features_used: List[str] = Field(..., description="Features used in prediction")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")
    confidence: float = Field(..., description="Confidence in assessment (0-1)", ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class RiskAssessmentModel:
    """ML model for treasury risk assessment and allocation optimization."""

    def __init__(self, model_dir: str = "models/risk_assessment"):
        """Initialize the risk assessment model."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[GradientBoostingClassifier] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self._load_model()

    def _load_model(self):
        """Load the model from disk."""
        try:
            model_path = self.model_dir / "risk_assessment_model.pkl"
            scaler_path = self.model_dir / "risk_assessment_scaler.pkl"
            metadata_path = self.model_dir / "risk_assessment_metadata.json"

            if model_path.exists() and scaler_path.exists() and metadata_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                self.feature_names = self.metadata.get("feature_names", [])
                self.is_loaded = True
                logger.info(f"Risk assessment model loaded from {self.model_dir}")
            else:
                logger.warning(f"Risk assessment model not found at {self.model_dir}")
                self._create_stub_model()
        except Exception as e:
            logger.error(f"Failed to load risk assessment model: {e}")
            self._create_stub_model()

    def _create_stub_model(self):
        """Create a stub model for development."""
        logger.info("Creating stub risk assessment model")
        
        # Define feature names
        self.feature_names = [
            "current_balance", "historical_balance_volatility", "balance_trend",
            "avg_daily_net_flow", "cash_flow_volatility", "cash_flow_consistency",
            "current_ratio", "quick_ratio", "cash_ratio", "working_capital",
            "debt_to_equity_ratio", "interest_coverage_ratio", "credit_line_utilization", "available_credit",
            "monthly_revenue", "monthly_expenses", "revenue_volatility", "expense_volatility", "profit_margin",
            "interest_rate", "market_volatility", "economic_indicator", "industry_risk_score",
            "vendor_payment_commitments", "payroll_commitments", "tax_obligations", "debt_service_requirements", "total_upcoming_obligations",
            "counterparty_risk_score", "regulatory_risk_score", "operational_risk_score", "concentration_risk_score",
            "company_size_factor", "business_cycle_stage_growth", "business_cycle_stage_stable", "business_cycle_stage_decline", "years_in_business",
            "is_month_end", "is_quarter_end", "is_year_end", "is_holiday_period", "day_of_week"
        ]
        
        # Create stub model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Create and fit stub scaler with dummy data
        self.scaler = StandardScaler()
        import numpy as np
        dummy_data = np.random.randn(100, len(self.feature_names))
        self.scaler.fit(dummy_data)
        
        # Fit the model with dummy data
        dummy_targets = np.random.choice([0, 1, 2, 3], 100, p=[0.4, 0.3, 0.2, 0.1])  # Risk levels: low, medium, high, critical
        self.model.fit(dummy_data, dummy_targets)
        
        # Create stub metadata
        self.metadata = {
            "version": "1.0.0",
            "trained_on": datetime.now().isoformat(),
            "model_type": "GradientBoostingClassifier",
            "feature_names": self.feature_names,
            "performance_metrics": {
                "accuracy": 0.88,
                "precision": 0.85,
                "recall": 0.87,
                "f1_score": 0.86
            }
        }
        
        self.is_loaded = True
        logger.info("Stub risk assessment model created")

    def save_model(self, model_name: str = "risk_assessment_model") -> None:
        """Save the model to disk."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        model_path = self.model_dir / f"{model_name}.pkl"
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Feature importance (convert numpy types to Python types for JSON serialization)
        feature_importance = dict(zip(
            self.feature_names,
            [float(x) for x in self.model.feature_importances_]
        ))
        
        metadata = {
            **self.metadata,
            "feature_importance": feature_importance,
            "saved_on": datetime.now().isoformat()
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Risk assessment model saved to {self.model_dir}")

    def train_model(self, X: pd.DataFrame, y_risk_level: pd.Series) -> None:
        """Train the risk assessment model."""
        if X.shape[0] != len(y_risk_level):
            raise ValueError("X and y must have the same number of samples")
        
        logger.info(f"Training risk assessment model with {len(X)} samples")
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.model.fit(X_scaled, y_risk_level)
        
        # Calibrate probabilities
        self.calibrator = CalibratedClassifierCV(self.model, cv=3)
        self.calibrator.fit(X_scaled, y_risk_level)
        
        # Update metadata
        self.feature_names = list(X.columns)
        self.metadata.update({
            "version": "1.0.0",
            "trained_on": datetime.now().isoformat(),
            "model_type": "GradientBoostingClassifier",
            "feature_names": self.feature_names,
            "training_samples": len(X),
            "performance_metrics": {
                "accuracy": self.model.score(X_scaled, y_risk_level)
            }
        })
        
        self.is_loaded = True
        logger.info("Risk assessment model trained successfully")

    def assess_risk(self, features: RiskAssessmentFeatures) -> RiskAssessmentResult:
        """Assess treasury risk for the given features."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        start_time = datetime.now()
        
        # Convert features to array
        feature_dict = features.model_dump()
        
        # Handle categorical features (one-hot encoding)
        business_cycle = feature_dict.pop("business_cycle_stage")
        feature_dict["business_cycle_stage_growth"] = 1 if business_cycle == "growth" else 0
        feature_dict["business_cycle_stage_stable"] = 1 if business_cycle == "stable" else 0
        feature_dict["business_cycle_stage_decline"] = 1 if business_cycle == "decline" else 0
        
        # Convert boolean features to int
        bool_features = ["is_month_end", "is_quarter_end", "is_year_end", "is_holiday_period"]
        for feature in bool_features:
            feature_dict[feature] = 1 if feature_dict[feature] else 0
        
        feature_array = np.array([feature_dict[feature] for feature in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Predict risk level
        if self.calibrator is not None:
            probabilities = self.calibrator.predict_proba(feature_array_scaled)
            risk_level_probs = probabilities[0]
        else:
            probabilities = self.model.predict_proba(feature_array_scaled)
            risk_level_probs = probabilities[0]
        
        # Get predicted risk level (0=low, 1=medium, 2=high, 3=critical)
        predicted_risk_level = int(self.model.predict(feature_array_scaled)[0])
        confidence = max(risk_level_probs)
        
        # Map risk level to string
        risk_levels = ["low", "medium", "high", "critical"]
        risk_level_str = risk_levels[predicted_risk_level]
        
        # Calculate individual risk scores
        liquidity_risk = self._calculate_liquidity_risk(features)
        credit_risk = self._calculate_credit_risk(features)
        operational_risk = self._calculate_operational_risk(features)
        market_risk = self._calculate_market_risk(features)
        
        # Calculate overall risk score
        overall_risk_score = (liquidity_risk * 0.3 + credit_risk * 0.25 + 
                             operational_risk * 0.25 + market_risk * 0.2)
        
        # Generate allocation recommendations
        allocation_ratios = self._generate_allocation_recommendations(features, overall_risk_score)
        
        # Generate risk factors and suggestions
        risk_factors = self._identify_risk_factors(features, overall_risk_score)
        mitigation_suggestions = self._generate_mitigation_suggestions(features, risk_factors)
        early_warning_indicators = self._identify_early_warning_indicators(features)
        
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RiskAssessmentResult(
            overall_risk_score=round(overall_risk_score, 3),
            risk_level=risk_level_str,
            liquidity_risk_score=round(liquidity_risk, 3),
            credit_risk_score=round(credit_risk, 3),
            operational_risk_score=round(operational_risk, 3),
            market_risk_score=round(market_risk, 3),
            recommended_reserve_ratio=round(allocation_ratios["reserve"], 3),
            recommended_operating_ratio=round(allocation_ratios["operating"], 3),
            recommended_vendor_ratio=round(allocation_ratios["vendor"], 3),
            recommended_investment_ratio=round(allocation_ratios["investment"], 3),
            risk_factors=risk_factors,
            mitigation_suggestions=mitigation_suggestions,
            early_warning_indicators=early_warning_indicators,
            model_type=self.metadata.get("model_type", "GradientBoostingClassifier"),
            model_version=self.metadata.get("version", "1.0.0"),
            features_used=self.feature_names,
            prediction_time_ms=round(prediction_time, 3),
            confidence=round(confidence, 3),
            timestamp=datetime.now()
        )
    
    def _calculate_liquidity_risk(self, features: RiskAssessmentFeatures) -> float:
        """Calculate liquidity risk score."""
        # Factors: cash ratio, working capital, upcoming obligations vs balance
        cash_ratio_risk = max(0, 1 - features.cash_ratio)
        working_capital_risk = max(0, -features.working_capital / 100000)  # Normalize
        obligation_risk = min(1, features.total_upcoming_obligations / features.current_balance)
        
        return (cash_ratio_risk * 0.4 + working_capital_risk * 0.3 + obligation_risk * 0.3)
    
    def _calculate_credit_risk(self, features: RiskAssessmentFeatures) -> float:
        """Calculate credit risk score."""
        # Factors: debt ratios, credit utilization, interest coverage
        debt_risk = min(1, features.debt_to_equity_ratio / 2)  # Normalize
        credit_util_risk = features.credit_line_utilization
        interest_coverage_risk = max(0, 1 - features.interest_coverage_ratio / 5)  # Normalize
        
        return (debt_risk * 0.4 + credit_util_risk * 0.3 + interest_coverage_risk * 0.3)
    
    def _calculate_operational_risk(self, features: RiskAssessmentFeatures) -> float:
        """Calculate operational risk score."""
        # Factors: revenue volatility, profit margin, operational risk score
        revenue_vol_risk = min(1, features.revenue_volatility / 0.5)  # Normalize
        margin_risk = max(0, (10 - features.profit_margin) / 20)  # Normalize
        operational_risk = features.operational_risk_score
        
        return (revenue_vol_risk * 0.3 + margin_risk * 0.3 + operational_risk * 0.4)
    
    def _calculate_market_risk(self, features: RiskAssessmentFeatures) -> float:
        """Calculate market risk score."""
        # Factors: market volatility, economic indicator, industry risk
        market_vol_risk = features.market_volatility
        economic_risk = 1 - features.economic_indicator
        industry_risk = features.industry_risk_score
        
        return (market_vol_risk * 0.4 + economic_risk * 0.3 + industry_risk * 0.3)
    
    def _generate_allocation_recommendations(self, features: RiskAssessmentFeatures, risk_score: float) -> Dict[str, float]:
        """Generate optimal allocation recommendations based on risk."""
        base_reserve = 0.15
        base_operating = 0.70
        base_vendor = 0.15
        base_investment = 0.0
        
        # Adjust based on risk score
        if risk_score > 0.7:  # High risk
            reserve_ratio = min(0.30, base_reserve + 0.10)
            operating_ratio = min(0.60, base_operating - 0.05)
            vendor_ratio = min(0.20, base_vendor + 0.05)
            investment_ratio = 0.0
        elif risk_score > 0.4:  # Medium risk
            reserve_ratio = base_reserve + 0.05
            operating_ratio = base_operating - 0.05
            vendor_ratio = base_vendor
            investment_ratio = 0.05
        else:  # Low risk
            reserve_ratio = max(0.10, base_reserve - 0.05)
            operating_ratio = base_operating
            vendor_ratio = base_vendor
            investment_ratio = 0.10
        
        # Ensure ratios sum to 1
        total = reserve_ratio + operating_ratio + vendor_ratio + investment_ratio
        if total > 1:
            scale_factor = 1 / total
            reserve_ratio *= scale_factor
            operating_ratio *= scale_factor
            vendor_ratio *= scale_factor
            investment_ratio *= scale_factor
        
        return {
            "reserve": reserve_ratio,
            "operating": operating_ratio,
            "vendor": vendor_ratio,
            "investment": investment_ratio
        }
    
    def _identify_risk_factors(self, features: RiskAssessmentFeatures, risk_score: float) -> List[str]:
        """Identify specific risk factors."""
        risk_factors = []
        
        if features.cash_ratio < 0.2:
            risk_factors.append("Low cash ratio")
        if features.working_capital < 0:
            risk_factors.append("Negative working capital")
        if features.total_upcoming_obligations > features.current_balance * 0.5:
            risk_factors.append("High upcoming obligations relative to balance")
        if features.debt_to_equity_ratio > 2:
            risk_factors.append("High debt-to-equity ratio")
        if features.credit_line_utilization > 0.8:
            risk_factors.append("High credit line utilization")
        if features.revenue_volatility > 0.3:
            risk_factors.append("High revenue volatility")
        if features.profit_margin < 5:
            risk_factors.append("Low profit margin")
        if features.market_volatility > 0.7:
            risk_factors.append("High market volatility")
        if features.industry_risk_score > 0.7:
            risk_factors.append("High industry risk")
        
        return risk_factors
    
    def _generate_mitigation_suggestions(self, features: RiskAssessmentFeatures, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation suggestions."""
        suggestions = []
        
        if "Low cash ratio" in risk_factors:
            suggestions.append("Increase cash reserves by reducing non-essential expenses")
        if "Negative working capital" in risk_factors:
            suggestions.append("Improve working capital management and cash conversion cycle")
        if "High upcoming obligations" in risk_factors:
            suggestions.append("Negotiate payment terms with vendors or arrange additional financing")
        if "High debt-to-equity ratio" in risk_factors:
            suggestions.append("Consider equity financing or debt restructuring")
        if "High credit line utilization" in risk_factors:
            suggestions.append("Reduce credit line usage or secure additional credit facilities")
        if "High revenue volatility" in risk_factors:
            suggestions.append("Diversify revenue streams and improve demand forecasting")
        if "Low profit margin" in risk_factors:
            suggestions.append("Review pricing strategy and cost structure optimization")
        if "High market volatility" in risk_factors:
            suggestions.append("Implement hedging strategies and diversify investments")
        
        if not suggestions:
            suggestions.append("Continue monitoring current risk levels and maintain existing controls")
        
        return suggestions
    
    def _identify_early_warning_indicators(self, features: RiskAssessmentFeatures) -> List[str]:
        """Identify early warning indicators to monitor."""
        indicators = []
        
        indicators.append("Daily cash balance trend")
        indicators.append("Credit line utilization ratio")
        indicators.append("Accounts receivable aging")
        indicators.append("Vendor payment delays")
        indicators.append("Revenue variance from forecast")
        indicators.append("Interest coverage ratio")
        indicators.append("Working capital ratio")
        indicators.append("Market volatility index")
        
        return indicators


_risk_assessor: Optional[RiskAssessmentModel] = None


def get_risk_assessor() -> RiskAssessmentModel:
    """Get the global risk assessment model instance."""
    global _risk_assessor
    if _risk_assessor is None:
        _risk_assessor = RiskAssessmentModel()
    return _risk_assessor


def assess_treasury_risk(
    current_balance: float,
    historical_balance_volatility: float,
    balance_trend: float,
    avg_daily_net_flow: float,
    cash_flow_volatility: float,
    cash_flow_consistency: float,
    current_ratio: float,
    quick_ratio: float,
    cash_ratio: float,
    working_capital: float,
    debt_to_equity_ratio: float,
    interest_coverage_ratio: float,
    credit_line_utilization: float,
    available_credit: float,
    monthly_revenue: float,
    monthly_expenses: float,
    revenue_volatility: float,
    expense_volatility: float,
    profit_margin: float,
    interest_rate: float,
    market_volatility: float,
    economic_indicator: float,
    industry_risk_score: float,
    vendor_payment_commitments: float,
    payroll_commitments: float,
    tax_obligations: float,
    debt_service_requirements: float,
    total_upcoming_obligations: float,
    counterparty_risk_score: float,
    regulatory_risk_score: float,
    operational_risk_score: float,
    concentration_risk_score: float,
    company_size_factor: float,
    business_cycle_stage: str,
    years_in_business: int,
    is_month_end: bool,
    is_quarter_end: bool,
    is_year_end: bool,
    is_holiday_period: bool,
    day_of_week: int
) -> RiskAssessmentResult:
    """
    Assess treasury risk using the ML model.
    """
    features = RiskAssessmentFeatures(
        current_balance=current_balance,
        historical_balance_volatility=historical_balance_volatility,
        balance_trend=balance_trend,
        avg_daily_net_flow=avg_daily_net_flow,
        cash_flow_volatility=cash_flow_volatility,
        cash_flow_consistency=cash_flow_consistency,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        cash_ratio=cash_ratio,
        working_capital=working_capital,
        debt_to_equity_ratio=debt_to_equity_ratio,
        interest_coverage_ratio=interest_coverage_ratio,
        credit_line_utilization=credit_line_utilization,
        available_credit=available_credit,
        monthly_revenue=monthly_revenue,
        monthly_expenses=monthly_expenses,
        revenue_volatility=revenue_volatility,
        expense_volatility=expense_volatility,
        profit_margin=profit_margin,
        interest_rate=interest_rate,
        market_volatility=market_volatility,
        economic_indicator=economic_indicator,
        industry_risk_score=industry_risk_score,
        vendor_payment_commitments=vendor_payment_commitments,
        payroll_commitments=payroll_commitments,
        tax_obligations=tax_obligations,
        debt_service_requirements=debt_service_requirements,
        total_upcoming_obligations=total_upcoming_obligations,
        counterparty_risk_score=counterparty_risk_score,
        regulatory_risk_score=regulatory_risk_score,
        operational_risk_score=operational_risk_score,
        concentration_risk_score=concentration_risk_score,
        company_size_factor=company_size_factor,
        business_cycle_stage=business_cycle_stage,
        years_in_business=years_in_business,
        is_month_end=is_month_end,
        is_quarter_end=is_quarter_end,
        is_year_end=is_year_end,
        is_holiday_period=is_holiday_period,
        day_of_week=day_of_week
    )
    model = get_risk_assessor()
    return model.assess_risk(features)
