import streamlit as st
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

# ------------------------------------------------------------
# Extract text from DOCX
# ------------------------------------------------------------
def extract_resume_text(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


# ------------------------------------------------------------
# Clean JSON from LLM
# ------------------------------------------------------------
def clean_json(raw):
    cleaned = raw.strip()

    # remove backticks
    cleaned = re.sub(r"```json|```", "", cleaned).strip()

    # find first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]

    return cleaned


# ------------------------------------------------------------
# Render ATS circular gauge
# ------------------------------------------------------------
def render_gauge(score):
    score = max(0, min(100, score))
    angle = (score / 100) * 360

    html = f"""
    <style>
    .gauge-wrapper {{
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: conic-gradient(#39FF14 {angle}deg, #333 0deg);
        display: flex;
        justify-content: center;
        align-items: center;
        margin: auto;
    }}
    .gauge-inner {{
        width: 160px;
        height: 160px;
        background: #111;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        color: white;
        font-family: Arial, sans-serif;
    }}
    .gauge-score {{
        font-size: 42px;
        font-weight: bold;
        color: #39FF14;
        margin-bottom: -8px;
    }}
    .gauge-label {{
        font-size: 14px;
        color: #ccc;
    }}
    </style>

    <div class="gauge-wrapper">
        <div class="gauge-inner">
            <div class="gauge-score">{score}%</div>
            <div class="gauge-label">Overall Match Score</div>
        </div>
    </div>
    """
    return html



# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="ATS Resume Evaluator", layout="wide")

st.markdown("<h1 style='text-align:center;'>ATS Resume Evaluator<span style='font-size:14px; color:#888;'> by Vinay Yadav</span></h1>", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    resume_file = st.file_uploader("Upload Resume (.docx only)", type=["docx"])
    job_description = st.text_area("Paste Job Description", height=300)

with right:
    if resume_file:
        preview = extract_resume_text(resume_file)
        st.text_area("Resume Preview", preview[:2000], height=300)


# ------------------------------------------------------------
# Load LLM with API key
# ------------------------------------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=groq_api_key
)





def format_report(data):
    report = f"""
### 📌 Match Summary
{data['match_summary']}

---

### ❌ Missing Keywords
{"\n".join("- " + kw for kw in data['missing_keywords']) if data['missing_keywords'] else "None"}

---

### ❌ Missing Skills
{"\n".join("- " + sk for sk in data['missing_skills']) if data['missing_skills'] else "None"}

---

### ⚠️ Experience Gaps
{"\n".join("- " + gap for gap in data['experience_gaps']) if data['experience_gaps'] else "None"}

---

### 🔧 Improvement Suggestions
{"\n".join("- " + sug for sug in data['improvement_suggestions']) if data['improvement_suggestions'] else "None"}

---

### 📝 Sections to Add or Fix
{"\n".join("- " + sec for sec in data['sections_to_add_or_fix']) if data['sections_to_add_or_fix'] else "None"}
"""
    return report



# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------
if st.button("Evaluate Resume"):
    if not resume_file:
        st.error("Upload a DOCX resume.")
        st.stop()

    if not job_description:
        st.error("Paste Job Description.")
        st.stop()

    resume_text = extract_resume_text(resume_file)

    prompt = f"""
You are an ATS evaluator.

Return ONLY this JSON format:

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

--- JD ---
{job_description}
"""

    messages = [
        SystemMessage(content="You evaluate resumes for ATS scoring."),
        HumanMessage(content=prompt)
    ]

    with st.spinner("Processing..."):
        result = llm.invoke(messages)

    cleaned = clean_json(result.content)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        st.error("Could not parse JSON. Showing raw output:")
        st.text(result.content)
        st.stop()

    # Gauge
    st.markdown("### Match Score")
    st.markdown(render_gauge(data["ats_score"]), unsafe_allow_html=True)

    st.markdown("### Detailed Report")
    st.markdown(format_report(data))

