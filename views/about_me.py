from pathlib import Path

import streamlit as st
from PIL import Image


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# current_dir = Path("D:/Github Portfolio/Portfolio_Streamlit/python-multipage-webapp")
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "CV.pdf"
profile_pic = current_dir / "assets" / "profile-pic.jpg"


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Digital CV | Parth Bhanderi"
PAGE_ICON = ":wave:"
NAME = "Parth Bhanderi"
DESCRIPTION = """
Aspiring AI/ML Engineer
"""
EMAIL = "parthbhanderi16@email.com"
SOCIAL_MEDIA = {
    "LinkedIn": {
        "url": "https://www.linkedin.com/in/parth-bhanderi-ai-engineer/",
        "icon": ":material/person_pin:"
    },
    "GitHub": {
        "url": "https://github.com/goalhunter", 
        "icon": ":material/code:"
    }
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)


# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )
    st.write("📫", EMAIL)


# --- SOCIAL LINKS ---
st.write('\n')
# Enhanced Social Media Links with Icons
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, details) in enumerate(SOCIAL_MEDIA.items()):
    with cols[index]:
        st.markdown(f"{details['icon']} [{platform}]({details['url']} '{platform} Profile')")



# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Experience & Qulifications")
st.write(
    """
- ✔️ 2+ Years expereince building applications using Python
- ✔️ Strong hands on experience and knowledge in Machine Learning and AI
- ✔️ Good understanding of LLMs and Generative AI
- ✔️ Excellent team-player and displaying strong sense of initiative on tasks
"""
)


# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Django, Fast API, Flask)
- ✨ AI and ML: PyTorch, TensorFlow, Scikit-learn
- ☁️ Cloud : AWS, GCP, Azure
- 📊 Data Visulization: PowerBi, MS Excel, Plotly
- 📚 Modeling: Logistic regression, linear regression, decition trees
- 🗄️ Databases: SQL, MongoDB, Pinecone, Elasticsearch, ChromaDB
"""
)


# --- WORK HISTORY ---
st.write('\n')
st.subheader("Work History")
st.write("---")

# --- JOB 1
st.write("🚧", "**AI Intern | Critical Start Inc**")
st.write("06/2024 - 08/2024")
st.write(
    """
- ► Developed LLM-based phishing detection system using GPT-4, achieving 97% accuracy across 2,000+ emails 
- ► Reduced SOC analysts' time by 60% in creating phishing email reports by automating with RAG 
- ► Built ML classification models (Logistic Regression, SVM) for phishing email detection with 94% accuracy
- ► Automated security alert summaries using GPT-4, reducing report redundancy and token costs by 20%
"""
)

# --- JOB 2
st.write('\n')
st.write("🚧", "**Research Assistant | University of North Texas**")
st.write("01/2024 - Current")
st.write(
    """
- ► Pioneered state-of-the-art unified voice anti-spoofing system, detecting both physical and logical attacks with 98% accuracy on 
Dev set and 95% accuracy on Evaluation set with 5% average EER. 
- ► Engineered robust acoustic features using ESResNeXt architecture on 300,000+ ASVspoof2019 samples 
- ► Implemented novel deep learning solution for INTERSPEECH 2025 publication, surpassing existing methods in 
voice spoofing detection
"""
)

# --- JOB 3
st.write('\n')
st.write("🚧", "**Python Developer | FlyOnTech Solutions**")
st.write("04/2021 - 12/2022")
st.write(
    """
- ► Reduced server response time by 40% by designing high-performance RESTful APIs with Django and Flask
- ► Established streamlined CI/CD deployment on AWS through GitHub Actions, reducing deployment time by 40% 
- ► Performed comprehensive data cleaning and feature engineering to enhance the dataset's quality and relevance for 
modeling by 20%
"""
)
