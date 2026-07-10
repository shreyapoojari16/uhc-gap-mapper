from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import joblib
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
from datetime import datetime
import folium
import json

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'districts_with_tiers.csv')
MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURE_COLS = [
    'sub_centres_per_lakh', 'phcs_per_lakh', 'chcs_per_lakh', 'district_hospitals_per_lakh',
    'institutional_delivery_pct_hmis', 'csection_rate_pct', 'institutional_births_pct',
    'skilled_birth_attendance_pct', 'child_immunization_pct',
    'sanitation_pct', 'drinking_water_pct', 'electricity_pct', 'health_insurance_pct',
    'literacy_rate_pct', 'women_anaemia_pct',
]
RADAR_FEATURES = ['institutional_births_pct', 'skilled_birth_attendance_pct', 'electricity_pct', 'sanitation_pct', 'literacy_rate_pct']
FRIENDLY = {
    'institutional_births_pct': 'Inst. Births %', 'skilled_birth_attendance_pct': 'Skilled Birth Att. %',
    'electricity_pct': 'Electricity %', 'sanitation_pct': 'Sanitation %', 'literacy_rate_pct': 'Literacy %',
    'sub_centres_per_lakh': 'Sub-Centres/Lakh', 'phcs_per_lakh': 'PHCs/Lakh', 'chcs_per_lakh': 'CHCs/Lakh',
    'district_hospitals_per_lakh': 'District Hospitals/Lakh',
    'institutional_delivery_pct_hmis': 'Institutional Delivery %', 'csection_rate_pct': 'C-Section Rate %',
    'drinking_water_pct': 'Drinking Water %',
    'child_immunization_pct': 'Child Immunization %',
    'health_insurance_pct': 'Health Insurance %', 'women_anaemia_pct': 'Women Anaemia %',
}

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
TIER_COLORS = {'Critical': '#ba1a1a', 'Underserved': '#d97706', 'Adequate': '#92730c', 'Well Covered': '#0f9d58'}
TIER_DESCRIPTIONS = {
    'Critical': "This district has some of the weakest healthcare access in the country. Most residents likely face serious gaps in maternal care, basic infrastructure, or living conditions that put health outcomes at real risk.",
    'Underserved': "This district falls below the national average on healthcare access. There are meaningful gaps that need attention, though the situation isn't as severe as the lowest tier.",
    'Adequate': "This district performs close to or slightly above the national average. Healthcare access is reasonable but there's clear room for improvement in specific areas.",
    'Well Covered': "This district has strong healthcare access, performing above the national average across most indicators. This doesn't mean it's perfect — just that residents are relatively well-served compared to most of India.",
}
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), 'static', 'india_district.geojson')
TIER_MAP_COLORS = {
    'Critical': '#ba1a1a',
    'Underserved': '#f59e0b',
    'Adequate': '#fde047',
    'Well Covered': '#10b981',
}

def load_data():
    return pd.read_csv(DATA_PATH)


def get_default_values():
    return {f: FEATURE_RANGES[f][2] for f in FEATURE_COLS}


def make_radar_chart(district_row, national_avg):
    labels = [FRIENDLY[f] for f in RADAR_FEATURES]
    district_vals = [district_row[f] for f in RADAR_FEATURES]
    national_vals = [national_avg[f] for f in RADAR_FEATURES]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    district_vals += district_vals[:1]
    national_vals += national_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, district_vals, color='#ba1a1a', linewidth=2, label='This District')
    ax.fill(angles, district_vals, color='#ba1a1a', alpha=0.15)
    ax.plot(angles, national_vals, color='#00264a', linewidth=2, linestyle='--', label='National Avg')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/dashboard')
def dashboard():
    df = load_data()
    tier_counts = df['coverage_tier'].value_counts().reindex(
        ['Critical', 'Underserved', 'Adequate', 'Well Covered']
    ).to_dict()

    worst = df.nsmallest(6, 'healthcare_access_score')
    worst_districts = [
        {'district': r.district, 'state': r.state, 'score': round(r.healthcare_access_score, 2), 'tier': r.coverage_tier}
        for r in worst.itertuples()
    ]

    # ---- Build the choropleth map ----
    lookup = {}
    for r in df.itertuples():
        key = (r.state.strip().lower(), r.district.strip().lower())
        lookup[key] = r.coverage_tier

    with open(GEOJSON_PATH, encoding='utf-8') as f:
        geojson_data = json.load(f)

    matched_count = 0
    for feature in geojson_data['features']:
        props = feature['properties']
        state_name = str(props.get('NAME_1', '')).strip().lower()
        district_name = str(props.get('NAME_2', '')).strip().lower()
        tier = lookup.get((state_name, district_name))
        if tier:
            matched_count += 1
        feature['properties']['tier'] = tier if tier else 'Unmatched'
        feature['properties']['fill_color'] = TIER_MAP_COLORS.get(tier, '#d1d5db')

    m = folium.Map(location=[22.5, 80], zoom_start=4, tiles='CartoDB positron', width='100%', height='500px')

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': feature['properties']['fill_color'],
            'color': '#94a3b8',
            'weight': 0.5,
            'fillOpacity': 0.75,
        },
        highlight_function=lambda feature: {'weight': 2, 'color': '#00264a'},
        tooltip=folium.GeoJsonTooltip(fields=['NAME_2', 'NAME_1', 'tier'], aliases=['District:', 'State:', 'Tier:']),
    ).add_to(m)

    map_html = m._repr_html_()
    map_total = len(geojson_data['features'])
    match_pct = round(matched_count / map_total * 100, 1) if map_total else 0

    return render_template(
        'dashboard.html',
        active_page='dashboard',
        total_districts=len(df),
        critical_count=int((df['coverage_tier'] == 'Critical').sum()),
        avg_score=round(df['healthcare_access_score'].mean(), 2),
        tier_counts=tier_counts,
        worst_districts=worst_districts,
        map_html=map_html,
        matched_count=matched_count,
        map_total=map_total,
        match_pct=match_pct,
    )

@app.route('/drilldown')
def drilldown():
    df = load_data()
    all_districts = sorted(df['district'].tolist())
    requested = request.args.get('district', df.loc[df['healthcare_access_score'].idxmin(), 'district'])

    selected_row = df[df['district'] == requested].iloc[0]
    national_avg = df[FEATURE_COLS].mean()

    radar_chart = make_radar_chart(selected_row, national_avg)

    comparison = []
    for f in ['institutional_births_pct', 'skilled_birth_attendance_pct', 'electricity_pct', 'sanitation_pct', 'child_immunization_pct', 'health_insurance_pct']:
        d_val = round(selected_row[f], 1)
        n_val = round(national_avg[f], 1)
        comparison.append({'label': FRIENDLY[f], 'district_val': d_val, 'national_val': n_val, 'below_avg': d_val < n_val})

    scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_PATH, 'random_forest.pkl'))
    X_row = pd.DataFrame([selected_row[FEATURE_COLS]])
    X_scaled = scaler.transform(X_row)
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_scaled)
    tier_idx = list(rf_model.classes_).index(selected_row['coverage_tier'])
    shap_for_row = shap_vals[0, :, tier_idx] if np.array(shap_vals).ndim == 3 else shap_vals[tier_idx][0]
    top_factor_idx = np.argmax(np.abs(shap_for_row))
    top_factor = FEATURE_COLS[top_factor_idx]
    insight_text = (
        f"{selected_row['district']} is classified as {selected_row['coverage_tier']} primarily due to "
        f"{FRIENDLY.get(top_factor, top_factor)} ({round(selected_row[top_factor], 1)}, vs national average "
        f"{round(national_avg[top_factor], 1)}). This is the single strongest factor identified by SHAP for this district."
    )

    return render_template(
        'drilldown.html',
        active_page='drilldown',
        total_districts=len(df),
        all_districts=all_districts,
        selected={'district': selected_row['district'], 'state': selected_row['state'], 'tier': selected_row['coverage_tier']},
        radar_chart=radar_chart,
        comparison=comparison,
        insight_text=insight_text,
    )


@app.route('/models')
def model_comparison():
    df = load_data()
    X = df[FEATURE_COLS]
    y = df['coverage_tier']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
    X_test_scaled = scaler.transform(X_test)

    model_files = {
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'SVM': 'svm.pkl',
    }

    results = []
    best_acc = 0
    best_name = ''
    for name, fname in model_files.items():
        model = joblib.load(os.path.join(MODELS_PATH, fname))
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        if acc > best_acc:
            best_acc = acc
            best_name = name
        results.append({'name': name, 'accuracy': round(acc * 100, 1), 'f1': round(f1, 3), 'status': 'Baseline' if name == 'Decision Tree' else 'Standard'})

    for r in results:
        r['recommended'] = (r['name'] == best_name)

    best_model = joblib.load(os.path.join(MODELS_PATH, model_files[best_name]))
    y_pred_best = best_model.predict(X_test_scaled)
    tier_order = ['Critical', 'Underserved', 'Adequate', 'Well Covered']
    cm = confusion_matrix(y_test, y_pred_best, labels=tier_order)

    fig1, ax1 = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tier_order, yticklabels=tier_order, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    confusion_matrix_img = fig_to_base64(fig1)

    rf_model = joblib.load(os.path.join(MODELS_PATH, 'random_forest.pkl'))
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    importances.plot(kind='barh', ax=ax2, color='#00264a')
    ax2.invert_yaxis()
    fig2.tight_layout()
    feature_importance_img = fig_to_base64(fig2)

    return render_template(
        'models.html',
        active_page='models',
        total_districts=len(df),
        models=results,
        best_model_name=best_name,
        confusion_matrix_img=confusion_matrix_img,
        feature_importance_img=feature_importance_img,
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    df = load_data()
    all_districts = sorted(df['district'].tolist())

    values = get_default_values()
    prediction = None
    top_factors = []
    loaded_district = None

    if request.method == 'POST':
        for f in FEATURE_COLS:
            values[f] = float(request.form.get(f, values[f]))

        scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
        svm_model = joblib.load(os.path.join(MODELS_PATH, 'svm.pkl'))
        rf_model = joblib.load(os.path.join(MODELS_PATH, 'random_forest.pkl'))

        X_input = pd.DataFrame([values])[FEATURE_COLS]
        X_scaled = scaler.transform(X_input)
        prediction = svm_model.predict(X_scaled)[0]

        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(X_scaled)
        tier_idx = list(rf_model.classes_).index(prediction)
        shap_for_input = shap_vals[0, :, tier_idx] if np.array(shap_vals).ndim == 3 else shap_vals[tier_idx][0]
        top_idx = np.argsort(np.abs(shap_for_input))[::-1][:4]

        for idx in top_idx:
            f = FEATURE_COLS[idx]
            val = shap_for_input[idx]
            entered_value = values[f]
            national_val = df[f].mean()
            comparison_word = "higher than" if entered_value > national_val else "lower than"

            if val > 0:
                plain_explanation = (
                    f"Your entered value of {entered_value:.1f} is {comparison_word} the national "
                    f"average ({national_val:.1f}), and this is one of the main reasons the model "
                    f"leaned toward {prediction}."
                )
            else:
                plain_explanation = (
                    f"Your entered value of {entered_value:.1f} is {comparison_word} the national "
                    f"average ({national_val:.1f}). This value alone would normally suggest a different "
                    f"tier, but other factors outweighed it in the final prediction."
                )

            top_factors.append({
                'label': FRIENDLY.get(f, f),
                'explanation': plain_explanation,
            })

    return render_template(
        'predict.html',
        active_page='predict',
        total_districts=len(df),
        all_districts=all_districts,
        feature_cols=FEATURE_COLS,
        friendly=FRIENDLY,
        ranges=FEATURE_RANGES,
        values=values,
        prediction=prediction,
        tier_colors=TIER_COLORS,
        tier_descriptions=TIER_DESCRIPTIONS,
        top_factors=top_factors,
        loaded_district=loaded_district,
    )


@app.route('/predict/load', methods=['POST'])
def predict_load():
    df = load_data()
    all_districts = sorted(df['district'].tolist())
    district_name = request.form.get('load_district')

    values = get_default_values()
    if district_name:
        row = df[df['district'] == district_name].iloc[0]
        values = {f: float(row[f]) for f in FEATURE_COLS}

    return render_template(
        'predict.html',
        active_page='predict',
        total_districts=len(df),
        all_districts=all_districts,
        feature_cols=FEATURE_COLS,
        friendly=FRIENDLY,
        ranges=FEATURE_RANGES,
        values=values,
        prediction=None,
        tier_colors=TIER_COLORS,
        tier_descriptions=TIER_DESCRIPTIONS,
        top_factors=[],
        loaded_district=district_name,
    )


@app.route('/reports', methods=['GET', 'POST'])
def reports():
    df = load_data()
    all_districts = sorted(df['district'].tolist())
    national_avg = df[FEATURE_COLS].mean()

    df_sorted = df.sort_values('healthcare_access_score', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index + 1

    report = None
    if request.method == 'POST':
        district_name = request.form.get('district')
        row = df_sorted[df_sorted['district'] == district_name].iloc[0]

        state_rows = df_sorted[df_sorted['state'] == row['state']].sort_values('healthcare_access_score', ascending=False).reset_index(drop=True)
        state_rank = state_rows[state_rows['district'] == district_name].index[0] + 1

        indicators = []
        for f in ['institutional_births_pct', 'skilled_birth_attendance_pct', 'electricity_pct', 'sanitation_pct', 'child_immunization_pct', 'health_insurance_pct']:
            val = round(row[f], 1)
            nat = round(national_avg[f], 1)
            gap = round(val - nat, 1)
            indicators.append({
                'label': FRIENDLY[f], 'value': val, 'national': nat,
                'gap': gap, 'gap_display': f"{'+' if gap >= 0 else ''}{gap}"
            })

        report = {
            'district': row['district'], 'state': row['state'], 'tier': row['coverage_tier'],
            'score': round(row['healthcare_access_score'], 2),
            'population': f"{int(row['population']):,}",
            'rank': int(row['rank']), 'state_rank': int(state_rank), 'state_total': len(state_rows),
            'date': datetime.now().strftime('%b %d, %Y'),
            'indicators': indicators,
        }

    recent = [
        {'district': r.district, 'state': r.state, 'tier': r.coverage_tier, 'score': round(r.healthcare_access_score, 2)}
        for r in df_sorted.head(8).itertuples()
    ]

    return render_template(
        'reports.html',
        active_page='reports',
        total_districts=len(df),
        all_districts=all_districts,
        report=report,
        recent=recent,
        tier_colors=TIER_COLORS,
    )


@app.route('/documents', methods=['GET', 'POST'])
def documents():
    df = load_data()

    if 'session_docs' not in globals():
        pass

    base_docs = [
        { 'name': 'RHS 2020 Health Infrastructure Report', 'type': 'Government Dataset', 'size': f"{len(df)} districts", 'author': 'Ministry of Health', 'date': 'FY 2020'},
        { 'name': 'HMIS 2019 Maternal & Child Health Data', 'type': 'Government Dataset', 'size': f"{len(df)} districts", 'author': 'Ministry of Health', 'date': 'FY 2019'},
        { 'name': 'NFHS-5 District Factsheets', 'type': 'Survey Data', 'size': f"{len(df)} districts", 'author': 'Ministry of Health & Family Welfare', 'date': '2019-21'},
        { 'name': 'Census 2011 District Population Data', 'type': 'Census Data', 'size': f"{len(df)} districts", 'author': 'Ministry of Home Affairs', 'date': '2011'},
        { 'name': 'UHC Gap Mapper — Model Comparison Report', 'type': 'Analytics', 'size': '4 models evaluated', 'author': 'System Generated', 'date': datetime.now().strftime('%b %Y')},
        { 'name': 'District Tier Classification Summary', 'type': 'Analytics', 'size': f"{len(df)} districts classified", 'author': 'System Generated', 'date': datetime.now().strftime('%b %Y')},
    ]

    if 'uploaded_docs' not in app.config:
        app.config['uploaded_docs'] = []

    if request.method == 'POST':
        app.config['uploaded_docs'].append({
             'name': request.form.get('doc_name'), 'type': request.form.get('doc_type'),
            'size': 'User uploaded', 'author': 'You', 'date': datetime.now().strftime('%b %d, %Y'),
        })

    all_docs = base_docs + app.config['uploaded_docs']

    return render_template('documents.html', active_page='documents', total_districts=len(df), documents=all_docs)


@app.route('/notifications')
def notifications():
    df = load_data()

    critical_count = int((df['coverage_tier'] == 'Critical').sum())
    worst = df.nsmallest(1, 'healthcare_access_score').iloc[0]
    best = df.nlargest(1, 'healthcare_access_score').iloc[0]

    state_critical = df[df['coverage_tier'] == 'Critical'].groupby('state').size().sort_values(ascending=False)
    top_critical_state = state_critical.index[0] if len(state_critical) else 'N/A'
    top_critical_state_count = int(state_critical.iloc[0]) if len(state_critical) else 0

    notifications_list = [
        {
            'title': f'{worst["district"]} flagged as most critical district',
            'body': f'{worst["district"]}, {worst["state"]} has the lowest healthcare access score nationally ({round(worst["healthcare_access_score"],2)}). Immediate administrative review recommended.',
            'time': 'Live'
        },
        {
            'title': f'{critical_count} districts currently in Critical tier',
            'body': f'Out of {len(df)} total districts, {critical_count} ({round(critical_count/len(df)*100,1)}%) are classified Critical by the trained model.',
            'time': 'Live'
        },
        {
            'title': f'{top_critical_state} has the most Critical districts',
            'body': f'{top_critical_state_count} districts in {top_critical_state} are currently classified as Critical — the highest concentration of any state.',
            'time': 'Live'
        },
        {
            'title': f'{best["district"]} leads national healthcare access',
            'body': f'{best["district"]}, {best["state"]} has the highest healthcare access score nationally ({round(best["healthcare_access_score"],2)}).',
            'time': 'Live'
        },
        {
            'title': 'Model comparison report available',
            'body': 'SVM (RBF kernel) is currently the top-performing model at 92.6% test accuracy across 4 models evaluated.',
            'time': 'Live'
        },
        {
             'title': 'Dataset last refreshed',
            'body': f'District data covers Census 2011, RHS 2020, HMIS 2019, and NFHS-5 2019 — {len(df)} districts total.',
            'time': 'System'
        },
    ]

    return render_template('notifications.html', active_page='notifications', total_districts=len(df), notifications=notifications_list)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    df = load_data()

    if 'app_settings' not in app.config:
        app.config['app_settings'] = {'email_notif': True, 'critical_alerts': True, 'show_confidence': False}

    if request.method == 'POST':
        app.config['app_settings'] = {
            'email_notif': 'email_notif' in request.form,
            'critical_alerts': 'critical_alerts' in request.form,
            'show_confidence': 'show_confidence' in request.form,
        }

    return render_template('settings.html', active_page='settings', total_districts=len(df), settings=app.config['app_settings'])

if __name__ == '__main__':
    app.run(debug=True, port=5050)