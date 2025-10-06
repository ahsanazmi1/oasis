"""
Training script for Oasis ML models.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oasis.ml.liquidity_forecasting import LiquidityForecastingModel
from oasis.ml.risk_assessment import RiskAssessmentModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_all_models():
    logger.info("🚀 Starting Oasis ML model training...")

    # Train Liquidity Forecasting Model
    logger.info("🔧 Training liquidity forecasting model...")
    liquidity_forecaster = LiquidityForecastingModel()
    
    # Create synthetic data for liquidity forecasting
    n_samples = 5000
    np.random.seed(42)
    
    data_liquidity = {
        "current_balance": np.random.uniform(50000, 500000, n_samples),
        "historical_avg_balance": np.random.uniform(60000, 550000, n_samples),
        "historical_balance_volatility": np.random.uniform(5000, 75000, n_samples),
        "avg_daily_inflow": np.random.uniform(5000, 25000, n_samples),
        "avg_daily_outflow": np.random.uniform(4000, 20000, n_samples),
        "inflow_volatility": np.random.uniform(500, 5000, n_samples),
        "outflow_volatility": np.random.uniform(400, 4000, n_samples),
        "monthly_revenue": np.random.uniform(150000, 750000, n_samples),
        "monthly_expenses": np.random.uniform(120000, 600000, n_samples),
        "seasonal_factor": np.random.uniform(0.8, 1.3, n_samples),
        "interest_rate": np.random.uniform(3.0, 8.0, n_samples),
        "market_volatility": np.random.uniform(0.1, 0.8, n_samples),
        "economic_indicator": np.random.uniform(0.3, 0.9, n_samples),
        "vendor_payment_commitments": np.random.uniform(10000, 100000, n_samples),
        "payroll_commitments": np.random.uniform(15000, 150000, n_samples),
        "tax_obligations": np.random.uniform(5000, 50000, n_samples),
        "debt_service_requirements": np.random.uniform(2000, 20000, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "day_of_month": np.random.randint(1, 32, n_samples),
        "month_of_quarter": np.random.randint(1, 4, n_samples),
        "is_month_end": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        "is_quarter_end": np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        "is_year_end": np.random.choice([True, False], n_samples, p=[0.02, 0.98]),
        "is_holiday_period": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        "credit_line_utilization": np.random.uniform(0.0, 1.0, n_samples),
        "counterparty_risk_score": np.random.uniform(0.0, 0.8, n_samples),
        "regulatory_risk_score": np.random.uniform(0.0, 0.5, n_samples),
        "business_cycle_stage_growth": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        "business_cycle_stage_stable": np.random.choice([True, False], n_samples, p=[0.5, 0.5]),
        "business_cycle_stage_decline": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        "industry_risk_score": np.random.uniform(0.1, 0.8, n_samples),
        "company_size_factor": np.random.uniform(0.5, 3.0, n_samples),
    }
    df_liquidity = pd.DataFrame(data_liquidity)
    
    # Generate synthetic liquidity targets (14-day forecast min balance)
    # Higher volatility, lower cash flows, higher obligations = lower min balance
    avg_daily_net_flow = df_liquidity['avg_daily_inflow'] - df_liquidity['avg_daily_outflow']
    df_liquidity['min_balance_14d'] = (
        df_liquidity['current_balance'] * 0.8 +  # Base 80% of current
        (avg_daily_net_flow * 14) +  # 14 days of net flow
        -df_liquidity['vendor_payment_commitments'] * 0.8 +  # Reduce by commitments
        -df_liquidity['payroll_commitments'] * 0.6 +  # Reduce by payroll
        -df_liquidity['historical_balance_volatility'] * 0.3  # Reduce by volatility
    )
    df_liquidity['min_balance_14d'] = np.maximum(df_liquidity['min_balance_14d'], 10000)  # Minimum $10k
    
    liquidity_forecaster.train_model(
        df_liquidity[liquidity_forecaster.feature_names], 
        df_liquidity['min_balance_14d']
    )
    logger.info("✅ Liquidity forecasting model trained and saved")

    # Train Risk Assessment Model
    logger.info("🔧 Training risk assessment model...")
    risk_assessor = RiskAssessmentModel()
    
    # Create synthetic data for risk assessment
    data_risk = {
        "current_balance": np.random.uniform(50000, 500000, n_samples),
        "historical_balance_volatility": np.random.uniform(5000, 75000, n_samples),
        "balance_trend": np.random.uniform(-20000, 30000, n_samples),
        "avg_daily_net_flow": np.random.uniform(-5000, 8000, n_samples),
        "cash_flow_volatility": np.random.uniform(1000, 15000, n_samples),
        "cash_flow_consistency": np.random.uniform(0.3, 1.0, n_samples),
        "current_ratio": np.random.uniform(0.5, 3.0, n_samples),
        "quick_ratio": np.random.uniform(0.3, 2.5, n_samples),
        "cash_ratio": np.random.uniform(0.1, 1.5, n_samples),
        "working_capital": np.random.uniform(-100000, 200000, n_samples),
        "debt_to_equity_ratio": np.random.uniform(0.1, 3.0, n_samples),
        "interest_coverage_ratio": np.random.uniform(1.0, 10.0, n_samples),
        "credit_line_utilization": np.random.uniform(0.0, 1.0, n_samples),
        "available_credit": np.random.uniform(50000, 500000, n_samples),
        "monthly_revenue": np.random.uniform(150000, 750000, n_samples),
        "monthly_expenses": np.random.uniform(120000, 600000, n_samples),
        "revenue_volatility": np.random.uniform(0.05, 0.5, n_samples),
        "expense_volatility": np.random.uniform(0.03, 0.4, n_samples),
        "profit_margin": np.random.uniform(-20, 30, n_samples),
        "interest_rate": np.random.uniform(3.0, 8.0, n_samples),
        "market_volatility": np.random.uniform(0.1, 0.8, n_samples),
        "economic_indicator": np.random.uniform(0.3, 0.9, n_samples),
        "industry_risk_score": np.random.uniform(0.1, 0.8, n_samples),
        "vendor_payment_commitments": np.random.uniform(10000, 100000, n_samples),
        "payroll_commitments": np.random.uniform(15000, 150000, n_samples),
        "tax_obligations": np.random.uniform(5000, 50000, n_samples),
        "debt_service_requirements": np.random.uniform(2000, 20000, n_samples),
        "total_upcoming_obligations": np.random.uniform(30000, 300000, n_samples),
        "counterparty_risk_score": np.random.uniform(0.0, 0.8, n_samples),
        "regulatory_risk_score": np.random.uniform(0.0, 0.5, n_samples),
        "operational_risk_score": np.random.uniform(0.0, 0.8, n_samples),
        "concentration_risk_score": np.random.uniform(0.0, 0.9, n_samples),
        "company_size_factor": np.random.uniform(0.5, 3.0, n_samples),
        "business_cycle_stage_growth": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        "business_cycle_stage_stable": np.random.choice([True, False], n_samples, p=[0.5, 0.5]),
        "business_cycle_stage_decline": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        "years_in_business": np.random.randint(1, 50, n_samples),
        "is_month_end": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        "is_quarter_end": np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        "is_year_end": np.random.choice([True, False], n_samples, p=[0.02, 0.98]),
        "is_holiday_period": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        "day_of_week": np.random.randint(0, 7, n_samples),
    }
    df_risk = pd.DataFrame(data_risk)
    
    # Generate synthetic risk level labels
    # Factors that increase risk: low cash ratio, high debt, negative working capital, high volatility
    risk_score = (
        (1 - df_risk['cash_ratio'] / 2) * 0.25 +  # Low cash ratio increases risk
        (df_risk['debt_to_equity_ratio'] / 3) * 0.2 +  # High debt increases risk
        (df_risk['working_capital'] < 0).astype(int) * 0.15 +  # Negative working capital
        (df_risk['revenue_volatility'] / 0.5) * 0.15 +  # High revenue volatility
        (df_risk['credit_line_utilization']) * 0.1 +  # High credit utilization
        (df_risk['profit_margin'] < 5).astype(int) * 0.1 +  # Low profit margin
        (df_risk['market_volatility'] / 0.8) * 0.05  # Market volatility
    )
    
    # Convert to risk levels
    df_risk['risk_level'] = 0  # Low
    df_risk.loc[risk_score > 0.3, 'risk_level'] = 1  # Medium
    df_risk.loc[risk_score > 0.6, 'risk_level'] = 2  # High
    df_risk.loc[risk_score > 0.8, 'risk_level'] = 3  # Critical
    
    risk_assessor.train_model(
        df_risk[risk_assessor.feature_names], 
        df_risk['risk_level']
    )
    logger.info("✅ Risk assessment model trained and saved")

    logger.info("🎉 All ML models trained successfully!")


if __name__ == "__main__":
    train_all_models()
