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
    layout="wide",
    initial_sidebar_state="expanded"
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

# --- 5. SIDEBAR: DOCTOR INTRO & MODEL SELECTION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=100)
    st.title("About Med-Verify")
    
    st.subheader("⚙️ Engine Settings")
    model_choice = st.selectbox(
        "Select AI Model:",
        options=["gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash"],
        index=0,
        help="If you hit a 429 error, try switching to Flash-Lite (highest quota)."
    )
    
    st.info("""
    **Capabilities:**
    1. **Multimodal Analysis:** Checks text and image.
    2. **Agentic RAG:** Searches live Google data.
    3. **Evidence Reasoning:** Provides a detailed 'Why'.
    """)
    st.divider()
    st.caption("Developed for CRISP-DM Phase 6: Deployment")

# --- 6. MAIN USER INTERFACE ---
st.title("🩺 Med-Verify AI: Agentic Fact-Checking")
st.write(f"Currently active engine: `{model_choice}`")

with st.form("inference_form"):
    col_input, col_img = st.columns([2, 1])
    with col_input:
        claim_text = st.text_area("Enter the Medical Claim:", placeholder="e.g., 'NIH confirms Vitamin D cures COVID-19...'", height=150)
    with col_img:
        uploaded_file = st.file_uploader("Upload Claim Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Preview", use_container_width=True)

    submit_btn = st.form_submit_button("🚀 Run Agentic RAG Verification")

# --- 7. EXECUTION ---
if submit_btn:
    if not (claim_text and uploaded_file):
        st.warning("⚠️ Please provide both a text claim and an image to proceed.")
    else:
        with st.status(f"🔍 Analyzing with {model_choice}...", expanded=True) as status:
            st.write("Step 1: Searching for evidence...")
            evidence = get_web_evidence(claim_text)
            
            st.write("Step 2: Synthesizing facts...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            uploaded_file.seek(0)
            img_obj = Image.open(uploaded_file).convert('RGB')
            
            prompt = f"Claim: {claim_text}\nEvidence: {evidence}\nAnalyze and provide Verdict and Reasoning."
            
            try:
                response = client.models.generate_content(
                    model=model_choice,
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
                
                # FIX: Defensive check to prevent AttributeErrors if parsing fails
                if response.parsed is None:
                    st.error("AI could not generate a valid verdict. It may have been blocked by safety filters.")
                    st.stop()
                    
                res = response.parsed
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"AI Model Error ({model_choice}): {str(e)}")
                st.stop()

        # Results Display
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            # FIX: Use res.prediction to decide formatting
            is_real = res.prediction.strip().lower() == "real"
            
            # FIX: Changed unsafe_allow_value to unsafe_allow_html
            if is_real:
                st.markdown(f'<div class="verdict-box" style="background-color: #d4edda; color: #155724;">✅ {res.prediction.upper()}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-box" style="background-color: #f8d7da; color: #721c24;">❌ {res.prediction.upper()}</div>', unsafe_allow_html=True)
            
            st.metric("Confidence", f"{res.confidence}%")
            
        with res_col2:
            st.subheader("AI Reasoning & Evidence")
            st.write(res.explanation)

