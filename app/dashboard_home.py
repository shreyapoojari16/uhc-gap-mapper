import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/districts_with_tiers.csv')

df = load_data()

# ---- REAL computed values, not mockup placeholders ----
total_districts = len(df)
critical_count = int((df['coverage_tier'] == 'Critical').sum())
model_f1 = 0.93  # from your Phase 7 classification report

worst_row = df.loc[df['healthcare_access_score'].idxmin()]
worst_district = worst_row['district']
worst_state = worst_row['state']
worst_tier = worst_row['coverage_tier']
worst_doctors_proxy = round(worst_row['sub_centres_per_lakh'], 1)
worst_immunization = round(worst_row['child_immunization_pct'], 1)

# Real sidebar (functional) — brand block only, nav lives in Streamlit's native sidebar
st.sidebar.markdown("""
<p class="sidebar-brand-eyebrow">Government of India</p>
<p class="sidebar-brand-title">UHC Portal</p>
<p class="sidebar-brand-sub">Health Ministry</p>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

DASHBOARD_HTML = f"""
<!DOCTYPE html>
<html class="light" lang="en"><head>
<meta charset="utf-8"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script id="tailwind-config">
  tailwind.config = {{ darkMode: "class", theme: {{ extend: {{
    "colors": {{ "primary":"#00264a","error":"#ba1a1a","on-primary":"#ffffff","on-error":"#ffffff" }},
    "spacing": {{ "sm":"8px","md":"16px","lg":"24px","xl":"32px" }},
  }} }} }};
</script>
<style>
body {{ background-color:#E7EDF3; font-family:'Public Sans',sans-serif; margin:0; }}
.material-symbols-outlined {{ font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24; }}
.content-card {{ background-color:#FFFFFF; border:1px solid #B8C6D6; transition: border-color 0.2s ease; }}
.content-card:hover {{ border-color:#0b3c6b; }}
</style>
</head>
<body>
<div class="p-lg" style="padding:24px;">
  <div class="flex items-center gap-md mb-lg" style="margin-bottom:24px;">
    <span class="material-symbols-outlined text-primary" style="font-size:32px; color:#00264a;">health_and_safety</span>
    <h1 style="color:#00264a; font-size:24px; font-weight:700; margin:0;">UHC Gap Mapper — Live Dashboard</h1>
  </div>

  <div class="grid grid-cols-12 gap-lg" style="display:grid; grid-template-columns:repeat(12,1fr); gap:24px;">
    <div class="col-span-12 lg:col-span-9 content-card rounded-lg overflow-hidden" style="grid-column: span 9;">
      <div style="padding:16px; border-bottom:1px solid #B8C6D6;">
        <p style="color:#43474f; font-size:13px; margin:0;">Tier distribution across all {total_districts} districts (real data, computed live)</p>
      </div>
      <div style="padding:20px; display:flex; gap:16px; flex-wrap:wrap;">
        {"".join(f'''
        <div style="flex:1; min-width:140px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px;">
          <p style="color:#43474f; font-size:11px; font-weight:700; text-transform:uppercase; margin:0;">{tier}</p>
          <p style="color:#00264a; font-size:26px; font-weight:800; margin:4px 0 0 0;">{int((df['coverage_tier']==tier).sum())}</p>
        </div>''' for tier in ['Critical','Underserved','Adequate','Well Covered'])}
      </div>
      <div style="padding:16px; border-top:1px solid #B8C6D6; display:flex; justify-content:center; gap:32px;">
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:14px;height:14px;background:#ba1a1a;border-radius:3px;"></div><span style="font-size:12px;color:#43474f;">Critical</span></div>
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:14px;height:14px;background:#f59e0b;border-radius:3px;"></div><span style="font-size:12px;color:#43474f;">Underserved / Adequate</span></div>
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:14px;height:14px;background:#10b981;border-radius:3px;"></div><span style="font-size:12px;color:#43474f;">Well Covered</span></div>
      </div>
    </div>

    <div class="col-span-12 lg:col-span-3" style="grid-column: span 3; display:flex; flex-direction:column; gap:16px;">
      <div class="content-card rounded-lg" style="padding:16px; border-radius:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:#43474f; font-size:11px; font-weight:700; text-transform:uppercase;">Districts in crisis</span>
          <span class="material-symbols-outlined" style="color:#ba1a1a; font-size:20px;">warning</span>
        </div>
        <p style="color:#ba1a1a; font-size:30px; font-weight:700; margin:4px 0 0 0;">{critical_count}</p>
        <p style="color:#43474f; font-size:13px; margin:4px 0 0 0;">out of {total_districts} total</p>
      </div>
      <div class="content-card rounded-lg" style="padding:16px; border-radius:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:#43474f; font-size:11px; font-weight:700; text-transform:uppercase;">Model F1 score</span>
          <span class="material-symbols-outlined" style="color:#00264a; font-size:20px;">bolt</span>
        </div>
        <p style="color:#00264a; font-size:30px; font-weight:700; margin:4px 0 0 0;">{model_f1}</p>
        <p style="color:#43474f; font-size:13px; margin:4px 0 0 0;">SVM · high confidence</p>
      </div>
      <div class="content-card rounded-lg overflow-hidden" style="border-radius:8px;">
        <div style="background:#00264a; padding:14px;">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
              <h3 style="color:#ffffff; font-size:16px; font-weight:700; margin:0;">{worst_district}</h3>
              <p style="color:#c9d6e8; font-size:12px; margin:2px 0 0 0;">{worst_state}</p>
            </div>
            <span style="background:#ba1a1a; color:white; font-size:10px; font-weight:700; padding:3px 8px; border-radius:4px;">{worst_tier}</span>
          </div>
        </div>
        <div style="padding:14px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <span style="color:#43474f; font-size:13px;">Sub-centres per lakh</span>
            <span style="color:#00264a; font-weight:700; font-size:13px;">{worst_doctors_proxy}</span>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span style="color:#43474f; font-size:13px;">Child immunization %</span>
            <span style="color:#00264a; font-weight:700; font-size:13px;">{worst_immunization}%</span>
          </div>
        </div>
      </div>
      <p style="color:#94a0ac; font-size:11px; text-align:center;">Lowest-scoring district, live from your trained model's output</p>
    </div>
  </div>
</div>
</body></html>
"""

components.html(DASHBOARD_HTML, height=650, scrolling=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)