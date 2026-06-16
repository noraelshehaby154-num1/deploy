import numpy as np
import pandas as pd

class SegmentationAgent:
    """Agent 1 — Customer Segmentation Agent (IMPROVED)"""
    def __init__(self, kmeans_model, scaler, risk_map, cluster_features):
        self.kmeans = kmeans_model
        self.scaler = scaler
        self.risk_map = risk_map
        self.cluster_features = cluster_features

    def _build_feature_vector(self, customer: dict) -> tuple:
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
        Calculate actual financial risk using indicators.
        Returns (segment, risk_score_0_to_100)
        """
        loan_to_income = feature_values['loan_to_income_ratio']
        debt_to_income = feature_values['debt_to_income']
        risk_pressure = feature_values['risk_pressure_index']
        net_worth = feature_values['net_worth']
        cibil = feature_values['cibil_score']
        
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
        
        # Classify into segments based on financial risk
        if risk_score < 25:
            segment = 'Low Risk'
        elif risk_score < 55:
            segment = 'Medium Risk'
        else:
            segment = 'High Risk'
        
        return segment, risk_score

    def run(self, customer: dict) -> dict:
        fv, feature_values = self._build_feature_vector(customer)
        fv_scaled = self.scaler.transform(fv)
        cluster_raw = self.kmeans.predict(fv_scaled)[0]
        
        # IMPROVED: Use financial indicators for more accurate risk classification
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

class DecisionAgent:
    """Agent 2 — Loan Decision Agent"""
    def __init__(self, clf_model, reg_model, clf_scaler, reg_scaler, feature_cols):
        self.clf       = clf_model
        self.reg       = reg_model
        self.clf_scaler = clf_scaler
        self.reg_scaler = reg_scaler
        self.feature_cols = feature_cols

    def _build_clf_features(self, customer: dict) -> pd.DataFrame:
        income      = customer['income_annum']
        loan_amt    = customer['loan_amount']
        cibil       = customer['cibil_score']
        loan_term   = customer['loan_term']
        dependents  = customer['no_of_dependents']
        total_assets = (customer['residential_assets_value'] +
                        customer['commercial_assets_value'] +
                        customer['luxury_assets_value'] +
                        customer['bank_asset_value'])
        emi_est        = loan_amt / max(loan_term, 1)
        debt_to_income = (emi_est * 12) / max(income, 1)
        lti            = loan_amt / max(income, 1)
        cibil_norm     = (cibil - 300) / 600
        affordability  = (income - emi_est * 12) / max(income, 1)
        net_worth      = total_assets - loan_amt
        asset_to_loan  = total_assets / max(loan_amt, 1)
        inc_per_dep    = income / max(dependents + 1, 1)
        liquid         = customer['bank_asset_value'] / max(total_assets, 1)
        rpi            = debt_to_income * (1 - cibil_norm)

        row = {
            'no_of_dependents': dependents,
            'education': int(customer.get('education_graduate', 1)),
            'self_employed': int(customer.get('self_employed', 0)),
            'income_annum': income,
            'loan_term': loan_term,
            'cibil_score': cibil,
            'residential_assets_value': customer['residential_assets_value'],
            'commercial_assets_value': customer['commercial_assets_value'],
            'luxury_assets_value': customer['luxury_assets_value'],
            'bank_asset_value': customer['bank_asset_value'],
            'total_assets': total_assets,
            'loan_to_income_ratio': lti,
            'net_worth': net_worth,
            'emi_estimate': emi_est,
            'debt_to_income': debt_to_income,
            'asset_to_loan': asset_to_loan,
            'income_per_dependent': inc_per_dep,
            'liquid_asset_ratio': liquid,
            'cibil_normalized': cibil_norm,
            'risk_pressure_index': rpi,
            'affordability_score': affordability,
        }
        return pd.DataFrame([row])[self.feature_cols]

    def run(self, customer: dict) -> dict:
        feat_df = self._build_clf_features(customer)
        feat_sc = self.clf_scaler.transform(feat_df)

        approved   = bool(self.clf.predict(feat_sc)[0])
        prob_approve = float(self.clf.predict_proba(feat_sc)[0][1])
        confidence = 'High' if abs(prob_approve - 0.5) > 0.35 else 'Medium' if abs(prob_approve - 0.5) > 0.15 else 'Low'

        feat_sc_reg = self.reg_scaler.transform(feat_df)
        predicted_amount = max(0, float(self.reg.predict(feat_sc_reg)[0]))
        risk_score = round((1 - prob_approve) * 100, 1)

        return {
            'approved': approved,
            'approval_probability': round(prob_approve, 4),
            'risk_score': risk_score,
            'confidence': confidence,
            'predicted_loan_amount': round(predicted_amount, -3),
            'requested_loan_amount': customer['loan_amount'],
            'decision': 'APPROVED' if approved else 'REJECTED',
            'flag_manual_review': confidence == 'Low',
        }

class MarketingCampaignAgent:
    """Agent 3 — Marketing Campaign Agent (Innovation Layer)"""
    CAMPAIGN_MATRIX = {
        ('Low Risk', True): {
            'offer_type': 'Premium Personal Loan',
            'interest_rate_band': '8.5% – 10.5% p.a.',
            'max_multiplier': 6.0,
            'message_tone': 'exclusive / premium',
            'key_benefits': ['No prepayment penalty', 'Top-up loan facility', 'Dedicated relationship manager'],
            'campaign_title': 'Elite Credit Program — You qualify for our best rates',
        },
        ('Medium Risk', True): {
            'offer_type': 'Standard Personal Loan',
            'interest_rate_band': '11.5% – 14.5% p.a.',
            'max_multiplier': 4.0,
            'message_tone': 'supportive / reassuring',
            'key_benefits': ['Flexible repayment schedule', 'Online account management', 'EMI holiday option'],
            'campaign_title': 'Smart Finance Program — A loan designed for your needs',
        },
        ('High Risk', True): {
            'offer_type': 'Secured Loan (Against Assets)',
            'interest_rate_band': '15.5% – 18.5% p.a.',
            'max_multiplier': 2.5,
            'message_tone': 'cautious / conditional',
            'key_benefits': ['Asset-backed approval', 'Credit score improvement program', 'Shorter tenure option'],
            'campaign_title': 'Secured Credit Program — Build your credit history with us',
        },
        ('Low Risk', False): {
            'offer_type': 'Pre-Approved Savings Plan',
            'interest_rate_band': 'N/A',
            'max_multiplier': None,
            'message_tone': 'advisory / future-focused',
            'key_benefits': ['Auto-approval in 6 months if profile maintained', 'Free credit counseling', 'CIBIL monitoring alerts'],
            'campaign_title': 'Future Ready Program — You are almost there',
        },
        ('Medium Risk', False): {
            'offer_type': 'Credit Builder Program',
            'interest_rate_band': 'N/A',
            'max_multiplier': None,
            'message_tone': 'advisory',
            'key_benefits': ['Secured credit card', 'Monthly credit coaching', 'Re-evaluation in 3 months'],
            'campaign_title': 'Credit Recovery Program — Let\'s strengthen your profile',
        },
        ('High Risk', False): {
            'offer_type': 'Financial Health Program',
            'interest_rate_band': 'N/A',
            'max_multiplier': None,
            'message_tone': 'empathetic / educational',
            'key_benefits': ['Free debt counseling', 'Financial literacy resources', 'Micro-savings account'],
            'campaign_title': 'New Start Program — We\'re here to help you improve',
        },
    }

    def run(self, segment_result: dict, decision_result: dict, customer: dict) -> dict:
        segment  = segment_result['segment']
        approved = decision_result['approved']
        key = (segment, approved)
        campaign = self.CAMPAIGN_MATRIX.get(key, self.CAMPAIGN_MATRIX[('Medium Risk', approved)])

        if approved and campaign['max_multiplier']:
            max_offer = customer['income_annum'] * campaign['max_multiplier']
            offer_amount = min(decision_result['predicted_loan_amount'], max_offer)
        else:
            offer_amount = None

        return {
            'campaign_title': campaign['campaign_title'],
            'offer_type': campaign['offer_type'],
            'interest_rate_band': campaign['interest_rate_band'],
            'offer_amount': round(offer_amount, -3) if offer_amount else None,
            'key_benefits': campaign['key_benefits'],
            'message_tone': campaign['message_tone'],
            'next_action': 'Proceed to loan documentation' if approved else 'Schedule financial counseling session',
        }