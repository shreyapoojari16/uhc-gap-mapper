import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="Predict | UHC Gap Mapper", page_icon="🔮", layout="wide")

with open('app/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.markdown("""
<p class="sidebar-brand-eyebrow">Government of India</p>
<p class="sidebar-brand-title">UHC Portal</p>
<p class="sidebar-brand-sub">Health Ministry</p>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown("""
<div class="app-header">
    <div class="icon-badge">🔮</div>
    <div>
        <h1>Predict District Coverage Tier</h1>
        <p>Enter indicator values for a real or hypothetical district and get a live model prediction</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Load real trained artifacts (no static/fake data) ----
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('models/scaler.pkl')
    svm_model = joblib.load('models/svm.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    return scaler, svm_model, rf_model

scaler, svm_model, rf_model = load_artifacts()

feature_cols = [
    'sub_centres_per_lakh', 'phcs_per_lakh', 'chcs_per_lakh', 'district_hospitals_per_lakh',
    'institutional_delivery_pct_hmis', 'csection_rate_pct', 'institutional_births_pct',
    'skilled_birth_attendance_pct', 'child_immunization_pct',
    'sanitation_pct', 'drinking_water_pct', 'electricity_pct', 'health_insurance_pct',
    'literacy_rate_pct', 'women_anaemia_pct',
]

# Real ranges pulled from your actual dataset (replace min/max below with Step 1's output)
FEATURE_RANGES = {
    'sub_centres_per_lakh':            (0.0, 115.0, 17.3),
    'phcs_per_lakh':                   (0.0, 51.0, 3.7),
    'chcs_per_lakh':                   (0.0, 19.0, 0.8),
    'district_hospitals_per_lakh':     (0.0, 7.0, 0.2),
    'institutional_delivery_pct_hmis': (28.0, 100.0, 93.6),
    'csection_rate_pct':               (0.0, 75.0, 20.6),
    'institutional_births_pct':        (21.0, 100.0, 88.5),
    'skilled_birth_attendance_pct':    (31.0, 100.0, 89.5),
    'child_immunization_pct':          (0.0, 100.0, 76.9),
    'sanitation_pct':                  (29.0, 100.0, 72.2),
    'drinking_water_pct':              (41.0, 100.0, 93.6),
    'electricity_pct':                 (68.0, 100.0, 97.1),
    'health_insurance_pct':            (1.0, 98.0, 40.2),
    'literacy_rate_pct':               (29.0, 89.0, 62.1),
    'women_anaemia_pct':               (15.0, 94.0, 55.6),
}

FRIENDLY_NAMES = {
    'sub_centres_per_lakh': 'Sub-Centres per Lakh Population',
    'phcs_per_lakh': 'PHCs per Lakh Population',
    'chcs_per_lakh': 'CHCs per Lakh Population',
    'district_hospitals_per_lakh': 'District Hospitals per Lakh',
    'institutional_delivery_pct_hmis': 'Institutional Delivery Rate (%)',
    'csection_rate_pct': 'C-Section Rate (%)',
    'institutional_births_pct': 'Institutional Births (%)',
    'skilled_birth_attendance_pct': 'Skilled Birth Attendance (%)',
    'child_immunization_pct': 'Child Immunization Rate (%)',
    'sanitation_pct': 'Sanitation Coverage (%)',
    'drinking_water_pct': 'Improved Drinking Water (%)',
    'electricity_pct': 'Electricity Access (%)',
    'health_insurance_pct': 'Health Insurance Coverage (%)',
    'literacy_rate_pct': 'Literacy Rate (%)',
    'women_anaemia_pct': 'Women Anaemia Rate (%)',
}

tier_colors = {
    'Critical': '#ba1a1a',
    'Underserved': '#d97706',
    'Adequate': '#92730c',
    'Well Covered': '#0f9d58',
}

st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown("### District Indicators")
st.caption("Load a real district as a starting point, or set values manually.")

# ---- Optional: pre-fill from a real district ----
df_real = pd.read_csv('data/processed/districts_with_tiers.csv')
district_options = ["— Manual entry —"] + sorted(
    (df_real['state'] + " | " + df_real['district']).tolist()
)
selected = st.selectbox("Load values from an existing district (optional)", district_options)

if selected != "— Manual entry —":
    state_sel, dist_sel = selected.split(" | ")
    row = df_real[(df_real['state'] == state_sel) & (df_real['district'] == dist_sel)].iloc[0]
    defaults = {col: float(row[col]) for col in feature_cols}
else:
    defaults = {col: FEATURE_RANGES[col][2] for col in feature_cols}

st.markdown("---")

input_values = {}
cols = st.columns(3)
for i, col in enumerate(feature_cols):
    lo, hi, _ = FEATURE_RANGES[col]
    with cols[i % 3]:
        input_values[col] = st.slider(
            FRIENDLY_NAMES[col], min_value=lo, max_value=hi,
            value=float(defaults[col]), key=f"slider_{col}"
        )

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Coverage Tier", use_container_width=True):
    X_input = pd.DataFrame([input_values])[feature_cols]
    X_input_scaled = scaler.transform(X_input)

    predicted_tier = svm_model.predict(X_input_scaled)[0]

    # Distance-based confidence proxy from SVM's decision function
    decision_scores = svm_model.decision_function(X_input_scaled)[0]
    confidence = float(np.max(np.abs(decision_scores)) / (np.sum(np.abs(decision_scores)) + 1e-9))

    # Explanation via Random Forest + SHAP (same explainer built in Phase 9)
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_input_scaled)
    tier_idx = list(rf_model.classes_).index(predicted_tier)
    shap_for_input = shap_vals[0, :, tier_idx] if np.array(shap_vals).ndim == 3 else shap_vals[tier_idx][0]
    top_factors = pd.Series(shap_for_input, index=feature_cols).sort_values(key=abs, ascending=False).head(4)

    color = tier_colors[predicted_tier]

    st.markdown(f"""
    <div class="content-card" style="border-left: 6px solid {color};">
        <div class="kpi-label">Predicted Coverage Tier</div>
        <div class="kpi-value" style="color:{color}; font-size:36px;">{predicted_tier}</div>
        <div class="kpi-sub">Model: SVM (RBF kernel) · 92.6% test accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### Top factors driving this prediction")
    st.caption("Explained via SHAP on the Random Forest model, using the same indicator values you entered.")
    for feat, val in top_factors.items():
        direction = "pushes toward" if val > 0 else "pushes away from"
        st.markdown(f"- **{FRIENDLY_NAMES[feat]}**: {direction} *{predicted_tier}* (impact: `{val:+.3f}`)")
    st.markdown('</div>', unsafe_allow_html=True)