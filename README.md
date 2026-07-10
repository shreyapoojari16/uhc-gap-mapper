# UHC Gap Mapper

**Mapping India's Universal Health Coverage gaps with machine learning.**

UHC Gap Mapper classifies all 736 districts of India into healthcare coverage tiers — *Critical*, *Underserved*, *Adequate*, and *Well Covered* — using real public health data and a suite of trained ML models. The project surfaces which districts most urgently need healthcare infrastructure investment, and explains *why* a district falls into a given tier through SHAP-based interpretability.

---

## 🎯 Problem Statement

India's healthcare access is uneven across districts, but there's no single, data-driven view that combines demographic, infrastructure, and service-delivery indicators into one coverage score. UHC Gap Mapper builds that view — a composite **Healthcare Access Score (0–100)** per district, derived from 11 features across multiple national datasets, then classified into actionable tiers.

---

## 📊 Data Sources

- **NFHS-5** (National Family Health Survey) — health & demographic indicators
- **HMIS** (Health Management Information System) — service delivery data
- **Census 2011** — population & infrastructure baselines
- **Hospital directory datasets** — facility density and distribution
- **India district boundaries** (GeoJSON) — for map visualization

---

## 🧠 Machine Learning Pipeline

1. **Data cleaning & merging** — reconciling district names across 4+ disparate government sources
2. **Feature engineering** — 11 features combined into a single Healthcare Access Score via min-max normalization
3. **Dimensionality reduction** — PCA and LDA for structure analysis
4. **Classification** — Decision Tree, Random Forest, Gradient Boosting, and SVM trained to predict coverage tier
   - **Best model: SVM**, selected as the production model
5. **Clustering validation** — KMeans used to cross-check tier assignments against natural data clusters
6. **Explainability** — SHAP values computed per-district to explain individual predictions

---

## 🖥️ Web Application

Built with **Flask + Tailwind CSS**, matching a pixel-accurate Stitch UI design across 9 pages:

| Page | Description |
|---|---|
| **Dashboard** | KPI overview, tier distribution, interactive Folium choropleth map of India, districts needing attention |
| **District Drill-down** | Per-district deep dive with radar chart of contributing features |
| **Model Comparison** | Side-by-side performance of all 4 trained classifiers |
| **Live Predictor** | Real-time tier prediction from user-input feature values, powered by the trained SVM |
| **Reports** | Generated summaries of coverage findings |
| **Documents** | Reference data documentation |
| **Notifications** | System/data update alerts |
| **Settings** | App configuration |

The Dashboard's map colors all matched districts by predicted tier, with an automatically calculated match percentage displayed for transparency (district name conventions differ slightly between the government data sources and the public GeoJSON boundary file — a known, documented limitation).

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** Tailwind CSS, Jinja2 templates
- **ML:** scikit-learn (Decision Tree, Random Forest, Gradient Boosting, SVM, KMeans), SHAP
- **Mapping:** Folium + GeoJSON
- **Data processing:** pandas, NumPy

---

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/uhc-gap-mapper.git
cd uhc-gap-mapper

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python web/app.py
```

Then open `http://localhost:5000` in your browser.

---

## 📁 Project Structure

```
uhc-gap-mapper/
├── assets/                     # Raw datasets, GeoJSON
├── models/                     # Trained .pkl models (Decision Tree, RF, GB, SVM, KMeans, scaler)
├── web/
│   ├── app.py                  # Flask application & routes
│   ├── static/                 # CSS, GeoJSON, static assets
│   └── templates/              # Jinja2 HTML templates (9 pages)
├── notebooks/                  # EDA, feature engineering, model training notebooks
├── requirements.txt
└── README.md
```

---

## 📈 Model Performance

| Model | Notes |
|---|---|
| **SVM** | Best-performing, selected as production model (~92.6% accuracy) |
| Random Forest | Strong baseline, used in SHAP comparisons |
| Gradient Boosting | Competitive performance |
| Decision Tree | Interpretable baseline |
| KMeans | Used for cluster validation (ARI ≈ 0.30 against tier labels) |

---

## ⚠️ Known Limitations

- District boundary GeoJSON uses slightly different naming conventions than the government datasets, so a subset of districts appear unmatched (grey) on the map — the match percentage is displayed live on the Dashboard for transparency.
- Coverage tiers are derived from available public data and are intended as a decision-support signal, not a definitive policy assessment.

---

## 👤 Author

Built by Shreya as a machine learning college project.

---

## 📄 License

This project is for academic purposes.
