import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from resume_utils import extract_text_from_file, find_email, extract_years_of_experience, basic_skill_extract
from scoring import keyword_score, embedding_similarity, experience_score, final_score

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("📄 Resume Screening AI Agent")

st.sidebar.header("Upload Job Description")
jd_text = st.sidebar.text_area("Paste Job Description", height=200)
jd_file = st.sidebar.file_uploader("Or upload JD file (.txt)", type=["txt"])

if jd_file and not jd_text:
    jd_text = jd_file.read().decode("utf-8")

resumes = st.sidebar.file_uploader("Upload resumes (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

kw_weight = st.sidebar.slider("Keyword Weight", 0.0, 1.0, 0.45)
emb_weight = st.sidebar.slider("Embedding Weight", 0.0, 1.0, 0.45)
exp_weight = st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.10)
weights = (kw_weight, emb_weight, exp_weight)

if not jd_text:
    st.info("Please paste or upload a Job Description to continue.")
    st.stop()

st.subheader("Job Description Preview")
st.write(jd_text)

jd_emb = model.encode([jd_text])[0]
keywords = basic_skill_extract(jd_text, top_n=40)

if not resumes:
    st.warning("Upload at least one resume to proceed.")
    st.stop()

results = []

for res in resumes:
    text = extract_text_from_file(res)
    email = find_email(text)
    years = extract_years_of_experience(text)

    kw = keyword_score(jd_text, text, keywords)
    emb = embedding_similarity(jd_emb, model.encode([text])[0])
    exp = experience_score(3, years)  # expected 3 years experience

    score = final_score(kw, emb, exp, weights)

    results.append({
        "Filename": res.name,
        "Email": email,
        "Experience (Years)": years,
        "Keyword Score": round(kw, 3),
        "Embedding Score": round(emb, 3),
        "Experience Score": round(exp, 3),
        "Final Score": round(score, 3)
    })

df = pd.DataFrame(results).sort_values("Final Score", ascending=False)
st.subheader("🏆 Ranked Results")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Results as CSV", csv, "results.csv")
