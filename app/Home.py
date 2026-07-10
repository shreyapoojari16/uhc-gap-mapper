import streamlit as st

st.set_page_config(page_title="UHC Gap Mapper", page_icon="🏥", layout="wide")

if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    pg = st.navigation([st.Page("auth_pages/splash.py")], position="hidden")
    pg.run()
else:
    pages = {
        "": [
            st.Page("dashboard_home.py", title="Home", icon="🏠"),
            st.Page("pages/1_Analysis.py", title="Analysis", icon="📊"),
            st.Page("pages/2_Map_and_Predict.py", title="Map and Predict", icon="🗺️"),
            st.Page("pages/3_Predict.py", title="Predict", icon="🔮"),
        ]
    }
    pg = st.navigation(pages)
    pg.run()