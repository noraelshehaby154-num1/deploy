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

        # ===== KPI DASHBOARD =====
        st.subheader("📊 Executive Summary - Key Metrics")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        with kpi_col1:
            decision_color = "🟢" if r2.get("decision") == "APPROVED" else "🔴"
            st.metric("✅ Decision", r2.get("decision", "N/A"))
        
        with kpi_col2:
            proba = r2.get("approval_probability", 0.0)
            st.metric("📈 Approval Probability", f"{proba:.1%}")
        
        with kpi_col3:
            risk_score = r2.get('risk_score', 0)
            st.metric("⚠️ Model Risk Score", f"{risk_score}/100")
        
        with kpi_col4:
            financial_risk = r1.get('financial_risk_score', 0)
            st.metric("💰 Financial Risk", f"{financial_risk}/100")
        
        with kpi_col5:
            st.metric("🏷️ Customer Segment", r1.get("segment", "N/A"))

        st.divider()

        # ===== TABBED VIEW FOR AGENTS =====
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔍 Agent 1: Segmentation", "⚡ Agent 2: Decision", "🎯 Agent 3: Marketing", "📋 Full Report"]
        )

        # ===== TAB 1: SEGMENTATION AGENT =====
        with tab1:
            st.subheader("Customer Segmentation Analysis")
            st.info(f"**📝 Interpretation:** {r1.get('interpretation', 'N/A')}")
            

            seg_col1, seg_col2, seg_col3 = st.columns(3)
            with seg_col1:
                st.metric("Customer Segment", r1.get("segment", "N/A"))
            with seg_col2:
                st.metric("CIBIL Band", r1.get("cibil_band", "N/A"))
            with seg_col3:
                st.metric("Loan-to-Income Ratio", f"{r1.get('loan_to_income_ratio', 0):.2f}×")
            
            # عرض بيانات التقسيم
            st.subheader("Segmentation Details")
            seg_data = {
                "Metric": [
                    "Customer Segment",
                    "Cluster ID",
                    "CIBIL Score Band",
                    "Loan-to-Income Ratio",
                    "Financial Risk Score",
                ],
                "Value": [
                    r1.get("segment", "N/A"),
                    r1.get("cluster_id", "N/A"),
                    r1.get("cibil_band", "N/A"),
                    f"{r1.get('loan_to_income_ratio', 0):.2f}",
                    f"{r1.get('financial_risk_score', 0):.1f}/100",
                ]
            }
            st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

        # ===== TAB 2: DECISION AGENT =====
        with tab2:
            st.subheader("Loan Decision & Risk Assessment")
            
            dec_col1, dec_col2, dec_col3 = st.columns(3)
            with dec_col1:
                decision = r2.get("decision", "N/A")
                color = "green" if decision == "APPROVED" else "red"
                st.metric("Final Decision", decision)
            with dec_col2:
                confidence = r2.get("confidence", "N/A")
                st.metric("Confidence Level", confidence)
            with dec_col3:
                flag = "⚠️ Yes" if r2.get("flag_manual_review", False) else "✅ No"
                st.metric("Manual Review Required", flag)
            
            # تفاصيل القرار
            st.subheader("Decision Metrics")
            decision_data = {
                "Metric": [
                    "Decision Status",
                    "Approval Probability",
                    "Risk Score (0-100)",
                    "Confidence Level",
                    "Predicted Loan Amount",
                    "Requested Loan Amount",
                    "Manual Review Flag",
                ],
                "Value": [
                    r2.get("decision", "N/A"),
                    f"{r2.get('approval_probability', 0):.1%}",
                    f"{r2.get('risk_score', 0)}/100",
                    r2.get("confidence", "N/A"),
                    f"${r2.get('predicted_loan_amount', 0):,.0f}",
                    f"${r2.get('requested_loan_amount', 0):,.0f}",
                    "Yes" if r2.get("flag_manual_review", False) else "No",
                ]
            }
            st.dataframe(pd.DataFrame(decision_data), use_container_width=True)

        # ===== TAB 3: MARKETING CAMPAIGN AGENT =====
        with tab3:
            st.subheader("📣 Personalized Campaign Recommendation")
            
            st.success(f"**🎯 Campaign Title:** {r3.get('campaign_title', 'N/A')}")
            
            camp_col1, camp_col2 = st.columns(2)
            with camp_col1:
                st.write(f"**📦 Offer Type:** {r3.get('offer_type', 'N/A')}")
                st.write(f"**💰 Interest Rate:** {r3.get('interest_rate_band', 'N/A')}")
                st.write(f"**🎤 Tone:** {r3.get('message_tone', 'N/A')}")
            
            with camp_col2:
                off_amt = r3.get('offer_amount')
                st.write(f"**💵 Max Approved Offer:** ${off_amt:,.2f}" if off_amt else "**💵 Max Approved Offer:** N/A")
                st.write(f"**✅ Recommended Action:** {r3.get('next_action', 'N/A')}")
            
            # عرض المزايا
            st.subheader("🎁 Key Product Benefits")
            for idx, benefit in enumerate(r3.get("key_benefits", []), 1):
                st.markdown(f"**{idx}.** {benefit}")
            
            # جدول العرض التسويقي
            st.subheader("Campaign Details")
            campaign_data = {
                "Attribute": [
                    "Campaign Title",
                    "Offer Type",
                    "Interest Rate Band",
                    "Max Offer Amount",
                    "Message Tone",
                    "Next Action",
                ],
                "Details": [
                    r3.get("campaign_title", "N/A"),
                    r3.get("offer_type", "N/A"),
                    r3.get("interest_rate_band", "N/A"),
                    f"${r3.get('offer_amount', 0):,.0f}" if r3.get('offer_amount') else "N/A",
                    r3.get("message_tone", "N/A"),
                    r3.get("next_action", "N/A"),
                ]
            }
            st.dataframe(pd.DataFrame(campaign_data), use_container_width=True)

        # ===== TAB 4: COMPREHENSIVE REPORT =====
        with tab4:
            st.subheader("📋 Complete Analysis Report")
            
            # Input Summary
            st.write("**🔹 Customer Input Summary:**")
            input_summary = {
                "Parameter": [
                    "Annual Income",
                    "CIBIL Score",
                    "Number of Dependents",
                    "Education",
                    "Self Employed",
                    "Loan Amount",
                    "Loan Term (months)",
                    "Total Assets",
                ],
                "Value": [
                    f"${income:,.0f}",
                    f"{cibil}",
                    f"{dependents}",
                    education,
                    "Yes" if self_employed else "No",
                    f"${loan_amount:,.0f}",
                    f"{loan_term}",
                    f"${res_assets + comm_assets + lux_assets + bank_assets:,.0f}",
                ]
            }
            st.dataframe(pd.DataFrame(input_summary), use_container_width=True)
            
            st.divider()
            
            # All Agents Output
            st.write("**🔹 Agent 1 - Segmentation Results:**")
            st.json(r1)
            
            st.write("**🔹 Agent 2 - Decision Results:**")
            st.json(r2)
            
            st.write("**🔹 Agent 3 - Marketing Results:**")
            st.json(r3)
            
            # Export Data
            st.divider()
            st.subheader("📥 Export Options")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # CSV Export
                csv_data = pd.DataFrame({
                    "Agent": ["Segmentation", "Decision", "Marketing"],
                    "Result": [str(r1), str(r2), str(r3)]
                })
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data.to_csv(index=False),
                    file_name="loan_analysis_report.csv",
                    mime="text/csv"
                )
            
            with export_col2:
                # JSON Export
                json_export = {
                    "customer_input": customer_payload,
                    "segmentation_agent": r1,
                    "decision_agent": r2,
                    "marketing_agent": r3,
                }
                st.download_button(
                    label="📥 Download as JSON",
                    data=pd.Series(json_export).to_json(),
                    file_name="loan_analysis_report.json",
                    mime="application/json"
                )