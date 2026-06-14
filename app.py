import streamlit as st
import pandas as pd
import numpy as np
import pickle
# استيراد الهيكلة أولاً للتأكد من فك الموديلات بشكل سليم
import agents 

from agents import SegmentationAgent, DecisionAgent, MarketingCampaignAgent

st.set_page_config(page_title="Credit Risk Engine", page_icon="🏦", layout="wide")

# تحميل الـ Agents الجاهزة بالكامل من ملف التخزين الموحد
@st.cache_resource
def load_all_agents():
    with open("trained_agents.pkl", "rb") as f:
        agent1, agent2, agent3 = pickle.load(f)
    return agent1, agent2, agent3

try:
    agent1, agent2, agent3 = load_all_agents()
    agents_ready = True
except Exception as e:
    agents_ready = False
    error_msg = str(e)

# الـ Interface
st.markdown("<h1 style='color:#185FA5;'>🏦 AI Credit Risk Scoring & Loan Decision System</h1>", unsafe_allow_html=True)
st.markdown("**Multi-Agent Intelligent Banking Decision Support Platform**")
st.divider()

with st.sidebar:
    st.header("📊 Mode Selection")
    mode = st.radio("Choose Mode", ["Single Application", "Portfolio Analytics"])

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("👤 Customer Profile")
    income = st.number_input("Annual Income ($)", min_value=100000, max_value=20000000, value=5000000, step=100000, format="%d")
    cibil = st.slider("CIBIL Score", 300, 900, 650)
    dependents = st.selectbox("No. of Dependents", [0,1,2,3,4,5])
    education = st.radio("Education", ["Graduate", "Not Graduate"])
    self_employed = st.checkbox("Self Employed")

with col_b:
    st.subheader("💰 Loan Parameters")
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=100000, max_value=50000000, value=10000000, step=100000, format="%d")
    loan_term = st.slider("Loan Term (months)", 2, 20, 10)
    res_assets   = st.number_input("Residential Assets ($)", min_value=0, value=5000000, step=100000, format="%d")
    comm_assets  = st.number_input("Commercial Assets ($)", min_value=0, value=2000000, step=100000, format="%d")
    lux_assets   = st.number_input("Luxury Assets ($)", min_value=0, value=3000000, step=100000, format="%d")
    bank_assets  = st.number_input("Bank Assets ($)", min_value=0, value=1000000, step=100000, format="%d")

analyze = st.button("⚡ Analyze Application", type="primary", use_container_width=True)

if analyze:
    if not agents_ready:
        st.error(f"❌ خطأ في تحميل الأيجنتس الجاهزة: {error_msg}")
    else:
        # تجهيز القاموس بكافة المفاتيح المطلوبة هندسياً داخل الكلاسات
        customer_payload = {
            "income_annum": income, 
            "loan_amount": loan_amount, 
            "cibil_score": cibil,
            "loan_term": loan_term, 
            "no_of_dependents": dependents,
            "education_graduate": 1 if education == "Graduate" else 0,
            "self_employed": int(self_employed),
            "residential_assets_value": res_assets, 
            "commercial_assets_value": comm_assets,
            "luxury_assets_value": lux_assets, 
            "bank_asset_value": bank_assets
        }

        # تشغيل الـ Pipeline بالترتيب الصحيح
        r1 = agent1.run(customer_payload)       # 1. Segmentation
        r2 = agent2.run(customer_payload)       # 2. Decision & Risk
        r3 = agent3.run(r1, r2, customer_payload) # 3. Marketing Offers

        st.divider()
        st.subheader("📊 Decision & Risk Analysis (Multi-Agent Output)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Decision", r2.get("decision", "N/A"))
        
        proba = r2.get("approval_probability", 0.0)
        m2.metric("Approval Probability", f"{proba:.1%}")
        m3.metric("Risk Score", f"{r2.get('risk_score', 0)}/100")
        m4.metric("Segment", r1.get("segment", "N/A"))

        # طباعة التفسير النصي التلقائي للأيجنت الأول
        st.info(f"**Agent Interpretation:** {r1.get('interpretation', '')}")

        # عرض تفاصيل العرض التسويقي المخصص
        st.subheader("📣 Personalized Campaign Recommendation")
        st.success(f"**🎯 Campaign Title:** {r3.get('campaign_title', 'N/A')}")
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.write(f"**Offer Product:** {r3.get('offer_type', 'N/A')}")
            st.write(f"**Interest Rate:** {r3.get('interest_rate_band', 'N/A')}")
        with c_col2:
            off_amt = r3.get('offer_amount')
            st.write(f"**Max Approved Offer Amount:** ${off_amt:,.2f}" if off_amt else "**Max Approved Offer Amount:** N/A")
            st.write(f"**Next Recommended Action:** `{r3.get('next_action', 'N/A')}`")

        st.markdown("**Key Product Benefits Included:**")
        for benefit in r3.get("key_benefits", []):
            st.markdown(f" * {benefit}")