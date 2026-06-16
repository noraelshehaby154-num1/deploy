"""
تحقق من منطق التجميع و التصنيف إلى فئات المخاطر
Verification script for clustering and risk classification logic
"""

import pickle
import pandas as pd
import numpy as np
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add agents module to path
sys.path.insert(0, os.path.dirname(__file__))
from agents import SegmentationAgent, DecisionAgent, MarketingCampaignAgent

try:
    # Load the trained agents
    with open("trained_agents.pkl", "rb") as f:
        agent1, agent2, agent3 = pickle.load(f)
    print("✅ Loaded agents from trained_agents.pkl")
except Exception as e:
    print(f"❌ Error loading agents: {e}")
    print(f"   Exception: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    exit(1)

# Print the risk_map used in segmentation agent
print("\n" + "="*70)
print("CLUSTERING RISK MAP (agent1.risk_map):")
print("="*70)
for cluster_id, risk_label in sorted(agent1.risk_map.items()):
    print(f"  Cluster {cluster_id} → {risk_label}")

print("\n" + "="*70)
print("FEATURE COLUMNS USED FOR CLUSTERING:")
print("="*70)
for i, feat in enumerate(agent1.cluster_features, 1):
    print(f"  {i}. {feat}")

# Test with different customer profiles to verify clustering
test_cases = [
    {
        'name': '✅ APPROVED: High Income, Excellent Credit, High Assets',
        'income_annum': 10000000, 'loan_amount': 5000000, 'cibil_score': 800,
        'loan_term': 12, 'no_of_dependents': 2, 'education_graduate': 1, 'self_employed': 0,
        'residential_assets_value': 20000000, 'commercial_assets_value': 10000000,
        'luxury_assets_value': 8000000, 'bank_asset_value': 5000000,
    },
    {
        'name': '⚠️ MEDIUM: Medium Income, Fair Credit',
        'income_annum': 5000000, 'loan_amount': 12000000, 'cibil_score': 600,
        'loan_term': 15, 'no_of_dependents': 3, 'education_graduate': 0, 'self_employed': 0,
        'residential_assets_value': 8000000, 'commercial_assets_value': 3000000,
        'luxury_assets_value': 2000000, 'bank_asset_value': 1000000,
    },
    {
        'name': '🔴 HIGH RISK: Low Income, Poor Credit, High Loan-to-Income',
        'income_annum': 2000000, 'loan_amount': 15000000, 'cibil_score': 400,
        'loan_term': 8, 'no_of_dependents': 5, 'education_graduate': 0, 'self_employed': 1,
        'residential_assets_value': 2000000, 'commercial_assets_value': 500000,
        'luxury_assets_value': 1000000, 'bank_asset_value': 200000,
    },
    {
        'name': '🔴 HIGH RISK: Negative Net Worth, High Risk Pressure',
        'income_annum': 3000000, 'loan_amount': 25000000, 'cibil_score': 350,
        'loan_term': 6, 'no_of_dependents': 4, 'education_graduate': 0, 'self_employed': 1,
        'residential_assets_value': 1000000, 'commercial_assets_value': 0,
        'luxury_assets_value': 500000, 'bank_asset_value': 100000,
    },
]

print("\n" + "="*70)
print("TESTING CUSTOMER PROFILES:")
print("="*70)

for customer in test_cases:
    name = customer.pop('name')
    print(f"\n{name}")
    print("-" * 70)
    
    # Calculate feature values to show what's being used
    income = customer['income_annum']
    loan_amt = customer['loan_amount']
    cibil = customer['cibil_score']
    loan_term = customer['loan_term']
    total_assets = (customer['residential_assets_value'] +
                   customer['commercial_assets_value'] +
                   customer['luxury_assets_value'] +
                   customer['bank_asset_value'])
    
    emi_estimate = loan_amt / max(loan_term, 1)
    debt_to_income = (emi_estimate * 12) / max(income, 1)
    loan_to_income = loan_amt / max(income, 1)
    net_worth = total_assets - loan_amt
    cibil_normalized = (cibil - 300) / 600
    risk_pressure = debt_to_income * (1 - cibil_normalized)
    affordability = (income - emi_estimate * 12) / max(income, 1)
    
    print(f"  Income: ${income:,} | Loan: ${loan_amt:,} | CIBIL: {cibil}")
    print(f"  Loan-to-Income: {loan_to_income:.2f}x | Debt-to-Income: {debt_to_income:.2f}x")
    print(f"  Total Assets: ${total_assets:,} | Net Worth: ${net_worth:,}")
    print(f"  Risk Pressure Index: {risk_pressure:.3f}")
    print(f"  Affordability Score: {affordability:.3f}")
    
    # Run segmentation
    r1 = agent1.run(customer)
    print(f"\n  📊 SEGMENTATION RESULT:")
    print(f"     Segment: {r1['segment']}")
    print(f"     Cluster ID: {r1['cluster_id']}")
    print(f"     CIBIL Band: {r1['cibil_band']}")
    print(f"     Financial Risk Score: {r1.get('financial_risk_score', 'N/A')}/100")
    
    # Run decision
    r2 = agent2.run(customer)
    print(f"\n  ⚡ DECISION RESULT:")
    print(f"     Decision: {r2['decision']}")
    print(f"     Approval Probability: {r2['approval_probability']:.1%}")
    print(f"     Risk Score: {r2['risk_score']}/100")
    print(f"     Confidence: {r2['confidence']}")
    
    if r1['segment'] == 'Low Risk' and r2['risk_score'] > 60:
        print(f"\n  ⚠️ WARNING: Marked as Low Risk but Risk Score is {r2['risk_score']}/100")
    elif r1['segment'] == 'High Risk' and r2['risk_score'] < 40:
        print(f"\n  ⚠️ WARNING: Marked as High Risk but Risk Score is {r2['risk_score']}/100")

print("\n" + "="*70)
print("✅ CLUSTERING VERIFICATION COMPLETE")
print("="*70)
