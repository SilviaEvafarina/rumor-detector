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
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_value=True)

# --- 3. SECRETS LOADING ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_SEARCH_KEY = st.secrets["GOOGLE_SEARCH_KEY"]
    SEARCH_ENGINE_ID = st.secrets["SEARCH_ENGINE_ID"]
except KeyError:
    st.error("🔑 API Keys missing! Please set GEMINI_API_KEY, GOOGLE_SEARCH_KEY, and SEARCH_ENGINE_ID in Streamlit Secrets.")
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

# --- 5. SIDEBAR: INTRODUCTION & CAPABILITIES ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=100)
    st.title("About Med-Verify")
    st.info("""
    **What is a Medical Rumor?**
    A medical rumor is an unverified claim regarding health, diseases, or treatments. In the digital age, these spread fast, often using misleading images to gain trust.
    
    **What can this Dashboard do?**
    1. **Multimodal Analysis:** Analyzes both your text and image simultaneously.
    2. **Agentic RAG:** Searches the live web for scientific consensus (NIH, WHO, CDC).
    3. **Evidence-Based Reasoning:** Provides a detailed 'Why' behind every verdict.
    """)
    st.divider()
    st.caption("Developed for CRISP-DM Phase 6: Deployment")

# --- 6. MAIN USER INTERFACE ---
st.title("🩺 Med-Verify AI: Agentic Fact-Checking")
st.write("Ensuring medical transparency through Multimodal Retrieval Augmented Generation.")

# Use a form to batch inputs and prevent unnecessary reruns
with st.form("inference_form"):
    col_input, col_img = st.columns([2, 1])
    
    with col_input:
        claim_text = st.text_area(
            "Enter the Medical Claim:", 
            placeholder="e.g., 'A new study says ginger cures cataracts overnight...'",
            height=150
        )
    
    with col_img:
        uploaded_file = st.file_uploader("Upload Claim Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Preview", use_container_width=True)

    submit_btn = st.form_submit_button("🚀 Run Agentic RAG Verification")

# --- 7. EXECUTION & OUTPUT ---
if submit_btn:
    if not (claim_text and uploaded_file):
        st.warning("⚠️ Please provide both a text claim and an image to proceed.")
    else:
        # Create a placeholder for real-time status updates
        with st.status("🔍 Agentic RAG in Progress...", expanded=True) as status:
            
            # Step 1: Web Retrieval
            st.write("Step 1: Searching global medical databases for evidence...")
            evidence = get_web_evidence(claim_text)
            
            # Step 2: Gemini Reasoning
            st.write("Step 2: Synthesizing visual cues with search facts...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            img_obj = Image.open(uploaded_file).convert('RGB')
            
            prompt = f"""
            You are a medical fact-checking agent. 
            Claim: {claim_text}
            Evidence: {evidence}
            
            Analyze if the claim is Real or Fake based on the evidence and the image provided.
            Provide a confidence score between 0-100%.
            """
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[prompt, img_obj],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "prediction": {"type": "string", "enum": ["Real", "Fake"]},
                            "confidence": {"type": "integer"},
                            "explanation": {"type": "string"}
                        },
                        "required": ["prediction", "confidence", "explanation"]
                    }
                )
            )
            res = response.parsed
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # Final Result Presentation
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if res.prediction == "Real":
                st.markdown(f'<div class="verdict-box" style="background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;">✅ {res.prediction}</div>', unsafe_allow_value=True)
            else:
                st.markdown(f'<div class="verdict-box" style="background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;">❌ {res.prediction}</div>', unsafe_allow_value=True)
            
            st.metric("Confidence Score", f"{res.confidence}%")
            
        with res_col2:
            st.subheader("AI Reasoning & Evidence")
            st.write(res.explanation)
            with st.expander("View Raw Search Evidence"):
                st.caption(evidence)

# --- 8. FOOTER ---
st.divider()
st.caption("Note: This AI tool is for informational purposes and uses RAG to fetch live data. Always consult a doctor for medical advice.")
