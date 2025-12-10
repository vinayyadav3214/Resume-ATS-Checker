import streamlit as st
from docx import Document
import pdfplumber  # pip install pdfplumber
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
import os
# -------------------------
# Helper: Extract text from resume
# -------------------------
def extract_resume_text(file):
    if file.name.endswith((".doc", ".docx")):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        return ""

# -------------------------
# Streamlit App UI
# -------------------------
st.title("ATS Resume Evaluator")
st.write("Upload your resume (PDF/DOCX) and job description to get the ATS score and improvement suggestions.")

resume_file = st.file_uploader("Upload Resume", type=["pdf", "doc", "docx"])
job_description = st.text_area("Paste Job Description here", height=250)

# -------------------------
# LLM Model
# -------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-5-chat-latest", temperature=0)

# -------------------------
# Evaluate Button
# -------------------------
if st.button("Evaluate Resume"):
    if not resume_file:
        st.error("Please upload a resume (.doc/.docx/.pdf).")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    # Extract resume text
    resume_text = extract_resume_text(resume_file)
    st.text_area("Resume Text Preview", resume_text, height=300)

    SCORING_RULES = """
    Calculate ATS score as a weighted sum:

    1. Keyword Match (40%)
    - Count number of required JD keywords present in the resume.
    - Score = (matched_keywords / total_keywords) * 40

    2. Skills Match (20%)
    - Count JD skills mentioned in the resume.
    - Score = (matched_skills / total_skills) * 20

    3. Experience Alignment (25%)
    - Rate from 0–25 based on how well the JD responsibilities are reflected.

    4. Resume Structure (15%)
    - Formatting clarity, section organization, bullet points, action verbs.

    Final ATS Score = sum of all four components.
    Round to nearest integer.
    """

    # Build prompt
    prompt = f"""
    You are an expert ATS evaluator.

    You MUST compute the ATS score strictly based on the scoring rules below.

    Return ONLY PURE JSON.
    Do NOT use any markdown formatting, backticks, or explanation text.

    SCORING RULES:
    1. Keyword Match (40%)
    2. Skills Match (20%)
    3. Experience Match (25%)
    4. Resume Structure (15%)

    JSON Format to return:

    {{
    "ats_score": 0,
    "match_summary": "",
    "missing_keywords": [],
    "missing_skills": [],
    "experience_gaps": [],
    "improvement_suggestions": [],
    "sections_to_add_or_fix": []
    }}

    --- Resume ---
    {resume_text}

    --- Job Description ---
    {job_description}
    """

    messages = [
        SystemMessage(content="You evaluate resumes for ATS scoring."),
        HumanMessage(content=prompt)
    ]

    # Call LLM
    with st.spinner("Analyzing resume..."):
        response = llm.invoke(messages)

    # Display result
    st.subheader("ATS Evaluation Result")
    st.json(response.content)
