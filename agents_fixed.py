"""
Fix for Clustering Risk Classification
This updates the SegmentationAgent to use financial indicators for better risk assessment
"""

import numpy as np
import pandas as pd

class SegmentationAgentFixed:
    """Agent 1 — Customer Segmentation Agent (FIXED VERSION)"""
    def __init__(self, kmeans_model, scaler, risk_map, cluster_features):
        self.kmeans = kmeans_model
        self.scaler = scaler
        self.risk_map = risk_map  # Legacy - kept for compatibility
        self.cluster_features = cluster_features

    def _build_feature_vector(self, customer: dict) -> np.ndarray:
        income       = customer['income_annum']
        loan_amt     = customer['loan_amount']
        cibil        = customer['cibil_score']
        loan_term    = customer['loan_term']
        dependents   = customer['no_of_dependents']
        total_assets = (customer['residential_assets_value'] +
                        customer['commercial_assets_value'] +
                        customer['luxury_assets_value'] +
                        customer['bank_asset_value'])

        emi_estimate       = loan_amt / max(loan_term, 1)
        debt_to_income     = (emi_estimate * 12) / max(income, 1)
        net_worth          = total_assets - loan_amt
        loan_to_income     = loan_amt / max(income, 1)
        cibil_normalized   = (cibil - 300) / 600
        risk_pressure      = debt_to_income * (1 - cibil_normalized)
        affordability      = (income - emi_estimate * 12) / max(income, 1)

        feature_values = {
            'income_annum': income,
            'cibil_score': cibil,
            'total_assets': total_assets,
            'loan_to_income_ratio': loan_to_income,
            'debt_to_income': debt_to_income,
            'net_worth': net_worth,
            'risk_pressure_index': risk_pressure,
            'affordability_score': affordability,
        }
        return np.array([feature_values[f] for f in self.cluster_features]).reshape(1, -1), feature_values

    def _calculate_financial_risk_score(self, feature_values: dict) -> tuple:
        """
        Calculate actual financial risk using indicators
        Returns (segment, confidence_score)
        """
        loan_to_income = feature_values['loan_to_income_ratio']
        debt_to_income = feature_values['debt_to_income']
        risk_pressure = feature_values['risk_pressure_index']
        net_worth = feature_values['net_worth']
        cibil = feature_values['cibil_score']
        affordability = feature_values['affordability_score']
        
        # Calculate risk score (0-100, higher = more risky)
        risk_score = 0
        
        # 1. Loan-to-Income Risk (0-30 points)
        if loan_to_income > 8:
            risk_score += 30
        elif loan_to_income > 6:
            risk_score += 25
        elif loan_to_income > 4:
            risk_score += 15
        elif loan_to_income > 2:
            risk_score += 5
        
        # 2. Debt-to-Income Risk (0-25 points)
        if debt_to_income > 3:
            risk_score += 25
        elif debt_to_income > 2:
            risk_score += 20
        elif debt_to_income > 1.5:
            risk_score += 12
        elif debt_to_income > 0.5:
            risk_score += 5
        
        # 3. Net Worth Risk (0-25 points)
        if net_worth < 0:
            risk_score += 25
        elif net_worth < 1000000:
            risk_score += 20
        elif net_worth < 5000000:
            risk_score += 10
        elif net_worth < 10000000:
            risk_score += 5
        
        # 4. Credit Score Risk (0-15 points)
        if cibil < 400:
            risk_score += 15
        elif cibil < 500:
            risk_score += 12
        elif cibil < 600:
            risk_score += 8
        elif cibil < 700:
            risk_score += 3
        
        # 5. Risk Pressure Index (0-10 points)
        if risk_pressure > 5:
            risk_score += 10
        elif risk_pressure > 2:
            risk_score += 6
        elif risk_pressure > 1:
            risk_score += 3
        
        risk_score = min(100, max(0, risk_score))
        
        # Classify into segments
        if risk_score < 25:
            segment = 'Low Risk'
        elif risk_score < 55:
            segment = 'Medium Risk'
        else:
            segment = 'High Risk'
        
        return segment, risk_score

    def run(self, customer: dict) -> dict:
        fv, feature_values = self._build_feature_vector(customer)
        
        # Use original clustering for cluster_id (for compatibility)
        fv_scaled = self.scaler.transform(fv)
        cluster_raw = self.kmeans.predict(fv_scaled)[0]
        
        # FIXED: Use financial indicators for actual risk classification
        segment, financial_risk_score = self._calculate_financial_risk_score(feature_values)
        
        lti = customer['loan_amount'] / max(customer['income_annum'], 1)
        cibil = customer['cibil_score']
        cibil_band = 'Excellent' if cibil>=750 else 'Good' if cibil>=650 else 'Fair' if cibil>=550 else 'Poor'

        return {
            'segment': segment,
            'cluster_id': int(cluster_raw),
            'cibil_band': cibil_band,
            'loan_to_income_ratio': round(lti, 2),
            'financial_risk_score': round(financial_risk_score, 1),
            'interpretation': (
                f"Customer classified as {segment} (Financial Risk: {financial_risk_score:.0f}/100). "
                f"CIBIL score {cibil} ({cibil_band}). "
                f"Loan-to-income ratio: {lti:.1f}× — "
                f"{'within acceptable range' if lti < 5 else 'elevated — requires careful review'}."
            )
        }
