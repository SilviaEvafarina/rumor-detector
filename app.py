import streamlit as st
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types
import requests

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Med-MMHL Fact Checker", page_icon="🩺", layout="wide")

# --- 2. CUSTOM STYLING (The original look you liked) ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SECRETS LOADING ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_SEARCH_KEY = st.secrets["GOOGLE_SEARCH_KEY"]
    SEARCH_ENGINE_ID = st.secrets["SEARCH_ENGINE_ID"]
except KeyError:
    st.error("Missing API Keys! Please add them to Streamlit Secrets.")
    st.stop()

# --- 4. NEW: MODEL CHOICE IN SIDEBAR (Keeps main page clean) ---
with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model Engine:",
        options=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"],
        index=0,
        help="Switch to Flash-Lite if you hit quota limits."
    )
    st.divider()
    st.info("This model choice will be used for the next analysis.")

# --- 5. AGENTIC RAG LOGIC ---
@st.cache_data(show_spinner=False)
def get_external_evidence(query):
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
    try:
        res = requests.get(url, timeout=10).json()
        return " ".join([item['snippet'] for item in res.get('items', [])[:3]])
    except Exception:
        return "Search failed to retrieve evidence."

# --- 6. ORIGINAL MAIN INTERFACE ---
st.title("🩺 Med-MMHL Fact Checker")
st.markdown("### Agentic RAG Verification System")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Claim")
    claim_text = st.text_area("Enter the medical claim text:", height=150)
    uploaded_file = st.file_uploader("Upload associated image", type=["jpg", "jpeg", "png"])
    
with col2:
    st.subheader("Verification Results")
    if st.button("🚀 Run Agentic RAG Verification"):
        if claim_text and uploaded_file:
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            with st.status("Searching & Analyzing...", expanded=True) as status:
                evidence = get_external_evidence(claim_text)
                uploaded_file.seek(0)
                img = Image.open(uploaded_file).convert('RGB')
                
                prompt = f"Claim: {claim_text}\nWeb Evidence: {evidence}\nAnalyze and provide JSON."
                
                try:
                    response = client.models.generate_content(
                        model=model_choice, # ✅ Uses the model you chose in the sidebar
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema={
                                "type": "object", 
                                "properties": {
                                    "prediction": {"type": "string", "enum": ["Real", "Fake"]}, 
                                    "reasoning": {"type": "string"}
                                }
                            }
                        )
                    )
                    result = response.parsed
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Error using {model_choice}: {str(e)}")
                    st.stop()

            # --- Results Display ---
            st.divider()
            if result.prediction == "Real":
                st.success(f"## VERDICT: {result.prediction}")
            else:
                st.error(f"## VERDICT: {result.prediction}")
            st.info(f"**Reasoning:**\n{result.reasoning}")
        else:
            st.warning("Please provide both a claim and an image.")
