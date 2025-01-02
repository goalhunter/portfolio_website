import streamlit as st


# --- PAGE SETUP ---
about_page = st.Page(
    "views/about_me.py",
    title="About Me",
    icon=":material/account_circle:",
    default=True,
)
project_1_page = st.Page(
    "views/sales_dashboard.py",
    title="Sales Dashboard",
    icon=":material/bar_chart:",
)
project_2_page = st.Page(
    "views/RAG.py",
    title="RAG powered Chat Bot",
    icon=":material/smart_toy:",
)
project_3_page = st.Page(
    "views/emotion_classification_app.py",
    title="Emotion Classifier",
    icon=":material/sentiment_very_satisfied:",
)
project_4_page = st.Page(
    "views/image_similarity_search.py",
    title="Image Vector Search",
    icon=":material/image_search:",
)
project_5_page = st.Page(
    "views/sportsperson_classification.py",
    title="Sports Legends Image Classification",
    icon=":material/image_search:",
)
# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Live Projects": [project_5_page, project_3_page, project_4_page, project_1_page],
    }
)


# --- SHARED ON ALL PAGES ---
# st.logo("assets/codingisfun_logo.png")
# st.sidebar.markdown("Made with ❤️ by [Sven](https://youtube.com/@codingisfun)")


# --- RUN NAVIGATION ---
pg.run()
