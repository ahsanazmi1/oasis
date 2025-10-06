"""
ML-enhanced treasury planner that integrates liquidity forecasting and risk assessment.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from oasis.planner import plan_treasury, validate_planning_inputs
from oasis.ml.liquidity_forecasting import get_liquidity_forecaster, LiquidityForecastingFeatures
from oasis.ml.risk_assessment import get_risk_assessor, RiskAssessmentFeatures

logger = logging.getLogger(__name__)


class MLEnhancedTreasuryPlanner:
    """ML-enhanced treasury planner with liquidity forecasting and risk assessment."""

    def __init__(self, ml_weight: float = 0.7, use_ml: bool = True):
        self.ml_weight = ml_weight
        self.use_ml = use_ml
        self.liquidity_forecaster = get_liquidity_forecaster()
        self.risk_assessor = get_risk_assessor()
        logger.info(f"MLEnhancedTreasuryPlanner initialized with ml_weight={self.ml_weight}, use_ml={self.use_ml}")

    def plan_treasury_enhanced(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ML-enhanced treasury plan with liquidity forecasting and risk assessment.
        
        Args:
            inputs: Treasury planning inputs containing current_balance,
                    expected_inflows, expected_outflows, etc.
        
        Returns:
            Dictionary with forecast, buckets, notes, and ML insights
        """
        if not self.use_ml:
            logger.info("ML is disabled, falling back to base treasury planner.")
            return plan_treasury(inputs)

        start_time = datetime.now()
        logger.info(
            f"Generating ML-enhanced treasury plan",
            extra={
                "current_balance": inputs.get("current_balance"),
                "risk_tolerance": inputs.get("risk_tolerance"),
                "ml_weight": self.ml_weight
            }
        )

        # 1. Run base treasury planning first
        base_result = plan_treasury(inputs)
        
        # 2. Prepare features for ML models
        liquidity_features = self._prepare_liquidity_forecasting_features(inputs)
        risk_features = self._prepare_risk_assessment_features(inputs)
        
        # 3. Get ML predictions
        liquidity_forecast = self.liquidity_forecaster.predict_liquidity_forecast(liquidity_features)
        risk_assessment = self.risk_assessor.assess_risk(risk_features)
        
        # 4. Enhance the base forecast with ML insights
        enhanced_forecast = self._enhance_forecast_with_ml(base_result["forecast"], liquidity_forecast)
        
        # 5. Enhance bucket allocation with ML recommendations
        enhanced_buckets = self._enhance_buckets_with_ml(base_result["buckets"], risk_assessment, inputs)
        
        # 6. Generate enhanced notes with ML insights
        enhanced_notes = self._generate_enhanced_notes(
            base_result["notes"], 
            liquidity_forecast, 
            risk_assessment, 
            inputs
        )
        
        planning_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            f"ML-enhanced treasury plan generated successfully",
            extra={
                "planning_time_ms": planning_time,
                "liquidity_risk_score": liquidity_forecast.liquidity_risk_score,
                "overall_risk_score": risk_assessment.overall_risk_score,
                "risk_level": risk_assessment.risk_level
            }
        )
        
        return {
            "forecast": enhanced_forecast,
            "buckets": enhanced_buckets,
            "notes": enhanced_notes,
            "inputs": base_result["inputs"],
            "metadata": {
                **base_result["metadata"],
                "ml_enhanced": True,
                "ml_weight": self.ml_weight,
                "planning_time_ms": planning_time,
                "liquidity_forecast": liquidity_forecast.dict(),
                "risk_assessment": risk_assessment.dict()
            }
        }

    def _prepare_liquidity_forecasting_features(self, inputs: Dict[str, Any]) -> LiquidityForecastingFeatures:
        """Prepare features for the Liquidity Forecasting Model."""
        current_balance = float(inputs.get("current_balance", 100000.0))
        expected_inflows = inputs.get("expected_inflows", {})
        expected_outflows = inputs.get("expected_outflows", {})
        
        # Extract or estimate historical data (in real system, this would come from data warehouse)
        historical_avg_balance = current_balance * 1.1  # Estimate 10% higher than current
        historical_balance_volatility = current_balance * 0.15  # Estimate 15% volatility
        
        # Calculate flow metrics
        inflow_multiplier = expected_inflows.get("multiplier", 1.0)
        outflow_multiplier = expected_outflows.get("multiplier", 1.0)
        avg_daily_inflow = 10000 * inflow_multiplier  # Base estimate
        avg_daily_outflow = 8000 * outflow_multiplier  # Base estimate
        
        # Calculate current date features
        now = datetime.now()
        day_of_week = now.weekday()
        day_of_month = now.day
        month_of_quarter = ((now.month - 1) % 3) + 1
        is_month_end = day_of_month >= 28
        is_quarter_end = now.month in [3, 6, 9, 12] and day_of_month >= 28
        is_year_end = now.month == 12 and day_of_month >= 28
        is_holiday_period = now.month in [11, 12]  # Holiday season
        
        return LiquidityForecastingFeatures(
            current_balance=current_balance,
            historical_avg_balance=historical_avg_balance,
            historical_balance_volatility=historical_balance_volatility,
            avg_daily_inflow=avg_daily_inflow,
            avg_daily_outflow=avg_daily_outflow,
            inflow_volatility=avg_daily_inflow * 0.2,  # 20% volatility
            outflow_volatility=avg_daily_outflow * 0.15,  # 15% volatility
            monthly_revenue=avg_daily_inflow * 30,  # Monthly estimate
            monthly_expenses=avg_daily_outflow * 30,  # Monthly estimate
            seasonal_factor=1.05 if is_holiday_period else 1.0,
            interest_rate=5.25,  # Current estimate
            market_volatility=0.3,  # Market volatility index
            economic_indicator=0.7,  # Economic health indicator
            vendor_payment_commitments=inputs.get("vendor_payment_schedule", {}).get("total_commitments", 0),
            payroll_commitments=avg_daily_outflow * 15,  # Estimate 15 days of payroll
            tax_obligations=avg_daily_outflow * 5,  # Estimate tax obligations
            debt_service_requirements=current_balance * 0.02,  # 2% of balance
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            month_of_quarter=month_of_quarter,
            is_month_end=is_month_end,
            is_quarter_end=is_quarter_end,
            is_year_end=is_year_end,
            is_holiday_period=is_holiday_period,
            credit_line_utilization=0.3,  # Estimate
            counterparty_risk_score=0.2,  # Estimate
            regulatory_risk_score=0.1,  # Estimate
            business_cycle_stage="stable",  # Default assumption
            industry_risk_score=0.3,  # Estimate
            company_size_factor=1.0  # Normalize
        )

    def _prepare_risk_assessment_features(self, inputs: Dict[str, Any]) -> RiskAssessmentFeatures:
        """Prepare features for the Risk Assessment Model."""
        current_balance = float(inputs.get("current_balance", 100000.0))
        risk_tolerance = inputs.get("risk_tolerance", "medium")
        
        # Calculate financial ratios (estimates for demo)
        current_assets = current_balance * 2  # Estimate current assets
        current_liabilities = current_balance * 0.8  # Estimate current liabilities
        total_debt = current_balance * 0.5  # Estimate total debt
        equity = current_balance * 1.5  # Estimate equity
        
        # Calculate current date features
        now = datetime.now()
        day_of_week = now.weekday()
        day_of_month = now.day
        is_month_end = day_of_month >= 28
        is_quarter_end = now.month in [3, 6, 9, 12] and day_of_month >= 28
        is_year_end = now.month == 12 and day_of_month >= 28
        is_holiday_period = now.month in [11, 12]
        
        return RiskAssessmentFeatures(
            current_balance=current_balance,
            historical_balance_volatility=current_balance * 0.15,
            balance_trend=current_balance * 0.05,  # 5% growth trend
            avg_daily_net_flow=2000,  # Net positive flow
            cash_flow_volatility=current_balance * 0.1,
            cash_flow_consistency=0.8,  # High consistency
            current_ratio=current_assets / current_liabilities,
            quick_ratio=(current_balance + current_balance * 0.3) / current_liabilities,  # Cash + receivables
            cash_ratio=current_balance / current_liabilities,
            working_capital=current_assets - current_liabilities,
            debt_to_equity_ratio=total_debt / equity,
            interest_coverage_ratio=5.0,  # Good coverage
            credit_line_utilization=0.3,
            available_credit=current_balance * 2,  # Available credit line
            monthly_revenue=current_balance * 0.4,  # Monthly revenue estimate
            monthly_expenses=current_balance * 0.35,  # Monthly expenses estimate
            revenue_volatility=0.2,  # 20% revenue volatility
            expense_volatility=0.15,  # 15% expense volatility
            profit_margin=12.5,  # 12.5% profit margin
            interest_rate=5.25,
            market_volatility=0.3,
            economic_indicator=0.7,
            industry_risk_score=0.3,
            vendor_payment_commitments=inputs.get("vendor_payment_schedule", {}).get("total_commitments", 0),
            payroll_commitments=current_balance * 0.15,
            tax_obligations=current_balance * 0.05,
            debt_service_requirements=current_balance * 0.02,
            total_upcoming_obligations=current_balance * 0.25,  # 25% of balance in obligations
            counterparty_risk_score=0.2,
            regulatory_risk_score=0.1,
            operational_risk_score=0.25,
            concentration_risk_score=0.3,
            company_size_factor=1.0,
            business_cycle_stage="stable",
            years_in_business=10,  # Assume established business
            is_month_end=is_month_end,
            is_quarter_end=is_quarter_end,
            is_year_end=is_year_end,
            is_holiday_period=is_holiday_period,
            day_of_week=day_of_week
        )

    def _enhance_forecast_with_ml(self, base_forecast: List[Dict[str, Any]], ml_forecast) -> List[Dict[str, Any]]:
        """Enhance base forecast with ML predictions."""
        enhanced_forecast = []
        
        for i, base_day in enumerate(base_forecast):
            if i < len(ml_forecast.forecast_days):
                ml_day = ml_forecast.forecast_days[i]
                
                # Blend base and ML predictions
                blended_balance = (
                    base_day["balance"] * (1 - self.ml_weight) + 
                    ml_day["predicted_balance"] * self.ml_weight
                )
                
                # Add ML insights
                enhanced_day = {
                    **base_day,
                    "balance": round(blended_balance, 2),
                    "ml_predicted_balance": round(ml_day["predicted_balance"], 2),
                    "ml_daily_change": round(ml_day["daily_change"], 2),
                    "ml_confidence": round(ml_day["confidence_score"], 3)
                }
            else:
                enhanced_day = base_day
            
            enhanced_forecast.append(enhanced_day)
        
        return enhanced_forecast

    def _enhance_buckets_with_ml(self, base_buckets: Dict[str, Any], risk_assessment, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance bucket allocation with ML recommendations."""
        current_balance = float(inputs.get("current_balance", 100000.0))
        
        # Blend base and ML recommendations
        ml_reserve_ratio = risk_assessment.recommended_reserve_ratio
        ml_operating_ratio = risk_assessment.recommended_operating_ratio
        ml_vendor_ratio = risk_assessment.recommended_vendor_ratio
        ml_investment_ratio = risk_assessment.recommended_investment_ratio
        
        # Get base ratios
        base_reserve_ratio = base_buckets["reserve"]["ratio"]
        base_operating_ratio = base_buckets["operating"]["ratio"]
        base_vendor_ratio = base_buckets["vendor"]["ratio"]
        
        # Blend ratios
        blended_reserve_ratio = base_reserve_ratio * (1 - self.ml_weight) + ml_reserve_ratio * self.ml_weight
        blended_operating_ratio = base_operating_ratio * (1 - self.ml_weight) + ml_operating_ratio * self.ml_weight
        blended_vendor_ratio = base_vendor_ratio * (1 - self.ml_weight) + ml_vendor_ratio * self.ml_weight
        
        # Calculate amounts
        reserve_amount = current_balance * blended_reserve_ratio
        operating_amount = current_balance * blended_operating_ratio
        vendor_amount = current_balance * blended_vendor_ratio
        investment_amount = current_balance * ml_investment_ratio  # Use ML recommendation directly
        
        # Ensure amounts don't exceed total balance
        total_allocated = reserve_amount + operating_amount + vendor_amount + investment_amount
        if total_allocated > current_balance:
            scale_factor = current_balance / total_allocated
            reserve_amount *= scale_factor
            operating_amount *= scale_factor
            vendor_amount *= scale_factor
            investment_amount *= scale_factor
        
        return {
            "reserve": {
                **base_buckets["reserve"],
                "amount": round(reserve_amount, 2),
                "ratio": round(blended_reserve_ratio, 3),
                "ml_recommendation": round(ml_reserve_ratio, 3),
                "risk_adjusted": True
            },
            "operating": {
                **base_buckets["operating"],
                "amount": round(operating_amount, 2),
                "ratio": round(blended_operating_ratio, 3),
                "ml_recommendation": round(ml_operating_ratio, 3),
                "risk_adjusted": True
            },
            "vendor": {
                **base_buckets["vendor"],
                "amount": round(vendor_amount, 2),
                "ratio": round(blended_vendor_ratio, 3),
                "ml_recommendation": round(ml_vendor_ratio, 3),
                "risk_adjusted": True
            },
            "investment": {
                "amount": round(investment_amount, 2),
                "ratio": round(ml_investment_ratio, 3),
                "purpose": "Low-risk investments and yield optimization",
                "ml_recommended": True,
                "risk_level": risk_assessment.risk_level
            }
        }

    def _generate_enhanced_notes(self, base_notes: str, liquidity_forecast, risk_assessment, inputs: Dict[str, Any]) -> str:
        """Generate enhanced notes with ML insights."""
        enhanced_notes = base_notes + "\n\n🤖 ML-Enhanced Analysis:\n\n"
        
        # Liquidity insights
        enhanced_notes += f"📊 Liquidity Forecasting:\n"
        enhanced_notes += f"• ML predicts minimum balance: ${liquidity_forecast.min_balance_forecast:,.2f}\n"
        enhanced_notes += f"• ML predicts maximum balance: ${liquidity_forecast.max_balance_forecast:,.2f}\n"
        enhanced_notes += f"• Liquidity risk score: {liquidity_forecast.liquidity_risk_score:.3f}\n"
        enhanced_notes += f"• Recommended buffer: ${liquidity_forecast.recommended_buffer_amount:,.2f}\n"
        enhanced_notes += f"• Confidence interval: ${liquidity_forecast.confidence_interval_lower:,.2f} - ${liquidity_forecast.confidence_interval_upper:,.2f}\n\n"
        
        # Risk assessment insights
        enhanced_notes += f"🎯 Risk Assessment:\n"
        enhanced_notes += f"• Overall risk level: {risk_assessment.risk_level.upper()}\n"
        enhanced_notes += f"• Overall risk score: {risk_assessment.overall_risk_score:.3f}\n"
        enhanced_notes += f"• Liquidity risk: {risk_assessment.liquidity_risk_score:.3f}\n"
        enhanced_notes += f"• Credit risk: {risk_assessment.credit_risk_score:.3f}\n"
        enhanced_notes += f"• Operational risk: {risk_assessment.operational_risk_score:.3f}\n"
        enhanced_notes += f"• Market risk: {risk_assessment.market_risk_score:.3f}\n\n"
        
        # ML allocation recommendations
        enhanced_notes += f"💡 ML-Recommended Allocation:\n"
        enhanced_notes += f"• Reserve: {risk_assessment.recommended_reserve_ratio:.1%}\n"
        enhanced_notes += f"• Operating: {risk_assessment.recommended_operating_ratio:.1%}\n"
        enhanced_notes += f"• Vendor: {risk_assessment.recommended_vendor_ratio:.1%}\n"
        enhanced_notes += f"• Investment: {risk_assessment.recommended_investment_ratio:.1%}\n\n"
        
        # Risk factors and mitigation
        if risk_assessment.risk_factors:
            enhanced_notes += f"⚠️ Key Risk Factors:\n"
            for factor in risk_assessment.risk_factors[:5]:  # Show top 5
                enhanced_notes += f"• {factor}\n"
            enhanced_notes += "\n"
        
        if risk_assessment.mitigation_suggestions:
            enhanced_notes += f"🛡️ Mitigation Suggestions:\n"
            for suggestion in risk_assessment.mitigation_suggestions[:3]:  # Show top 3
                enhanced_notes += f"• {suggestion}\n"
            enhanced_notes += "\n"
        
        # Early warning indicators
        enhanced_notes += f"📈 Monitor These Indicators:\n"
        for indicator in risk_assessment.early_warning_indicators[:5]:  # Show top 5
            enhanced_notes += f"• {indicator}\n"
        
        return enhanced_notes


# Global instance
ml_enhanced_planner = MLEnhancedTreasuryPlanner()
