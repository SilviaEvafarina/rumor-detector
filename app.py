import streamlit as st
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types
import requests
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Med-Verify AI | Medical Rumor Detector",
    page_icon="🩺",
    layout="wide"
)

# --- 2. CUSTOM STYLING (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SECRETS LOADING ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_SEARCH_KEY = st.secrets["GOOGLE_SEARCH_KEY"]
    SEARCH_ENGINE_ID = st.secrets["SEARCH_ENGINE_ID"]
except KeyError:
    st.error("🔑 API Keys missing! Please set them in Streamlit Secrets.")
    st.stop()

# --- 4. AGENTIC RAG LOGIC ---
@st.cache_data(show_spinner=False)
def get_web_evidence(query):
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
    try:
        res = requests.get(url, timeout=10).json()
        items = res.get('items', [])
        return " ".join([item['snippet'] for item in items[:3]])
    except:
        return "Could not retrieve live web evidence."

# --- 5. SIDEBAR: MODEL SELECTION ---
with st.sidebar:
    st.title("⚙️ Model Settings")
    
    # Dynamic Model Selection
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=[
            "gemini-2.5-flash-lite", 
            "gemini-2.5-flash", 
            "gemini-3-flash-preview", 
            "gemini-2.0-flash"
        ],
        help="Switch models if you hit a quota limit. Flash-Lite has the highest daily limit."
    )
    
    st.divider()
    st.info("""
    **Which one to pick?**
    - **2.5 Flash-Lite:** Best for many requests.
    - **2.5 Flash:** Best balance of logic & speed.
    - **3 Flash Preview:** Latest frontier intelligence.
    """)

# --- 6. MAIN USER INTERFACE ---
st.title("🩺 Med-Verify AI: Agentic Fact-Checking")
st.write(f"Current Engine: `{model_choice}`")

with st.form("inference_form"):
    col_input, col_img = st.columns([2, 1])
    with col_input:
        claim_text = st.text_area("Enter Medical Claim:", height=150)
    with col_img:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Preview", use_container_width=True)

    submit_btn = st.form_submit_button("🚀 Run Verification")

# --- 7. EXECUTION ---
if submit_btn:
    if not (claim_text and uploaded_file):
        st.warning("⚠️ Please provide both a claim and an image.")
    else:
        with st.status(f"🔍 Analyzing with {model_choice}...", expanded=True) as status:
            st.write("Step 1: Searching for evidence...")
            evidence = get_web_evidence(claim_text)
            
            st.write(f"Step 2: Processing with {model_choice}...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            uploaded_file.seek(0)
            img_obj = Image.open(uploaded_file).convert('RGB')
            
            prompt = f"Claim: {claim_text}\nEvidence: {evidence}\n\nAnalyze if Real or Fake."
            
            try:
                response = client.models.generate_content(
                    model=model_choice,  # ✅ Uses dynamic selection
                    contents=[prompt, img_obj],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "prediction": {"type": "string"},
                                "confidence": {"type": "integer"},
                                "explanation": {"type": "string"}
                            },
                            "required": ["prediction", "confidence", "explanation"]
                        }
                    )
                )
                
                if response.parsed is None:
                    st.error("AI failed to parse the schema. Try a different model.")
                    st.stop()
                    
                res = response.parsed
                status.update(label="✅ Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error using {model_choice}: {str(e)}")
                st.stop()

        # Result Display
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            is_real = res.prediction.strip().lower() == "real"
            color = "#d4edda" if is_real else "#f8d7da"
            icon = "✅" if is_real else "❌"
            st.markdown(f'<div class="verdict-box" style="background-color: {color};">{icon} {res.prediction.upper()}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{res.confidence}%")
        with res_col2:
            st.subheader("Reasoning")
            st.write(res.explanation)
