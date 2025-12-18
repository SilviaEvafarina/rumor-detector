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
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; border: none; }
    .stButton>button:hover { background-color: #0056b3; }
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; }
    .evidence-card { padding: 15px; border-radius: 8px; background-color: white; border-left: 5px solid #007bff; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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

# --- 4. MODELING TECHNIQUE: STEP 1 - EVIDENCE RETRIEVAL ---
@st.cache_data(show_spinner=False)
def get_web_evidence(query):
    """Retrieves evidence from Google Custom Search API."""
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
    try:
        res = requests.get(url, timeout=10).json()
        items = res.get('items', [])
        evidence_list = []
        for item in items[:5]:
            evidence_list.append({
                "title": item.get('title'),
                "snippet": item.get('snippet'),
                "link": item.get('link')
            })
        return evidence_list
    except Exception as e:
        st.error(f"Search API Error: {str(e)}")
        return []

# --- 5. SIDEBAR: INTRO & MODEL SELECTION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=100)
    st.title("About Med-Verify")
    
    st.subheader("⚙️ Engine Settings")
    model_choice = st.selectbox(
        "Select AI Model:",
        options=["gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash"],
        index=0,
        help="Switch to Flash-Lite if you hit quota limits (429 errors)."
    )
    
    st.info("""
    **Modeling Technique:**
    1. **Retrieval:** Uses Google Search to find clinical evidence.
    2. **Reasoning:** Uses Gemini to synthesize search data with image analysis.
    """)
    st.divider()
    st.caption("Developed for CRISP-DM Phase 6: Deployment")

# --- 6. MAIN USER INTERFACE ---
st.title("🩺 Med-Verify AI: Agentic Fact-Checking")
st.write(f"Active Reasoning Engine: `{model_choice}`")

with st.form("inference_form"):
    col_input, col_img = st.columns([2, 1])
    with col_input:
        claim_text = st.text_area("Enter Medical Claim:", placeholder="e.g., 'NIH confirms Vitamin D cures COVID-19...'", height=150)
    with col_img:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Preview", use_container_width=True)

    submit_btn = st.form_submit_button("🚀 Run Verification")

# --- 7. EXECUTION & GRACEFUL FALLBACK ---
if submit_btn:
    if not (claim_text and uploaded_file):
        st.warning("⚠️ Please provide both a claim and an image.")
    else:
        # STEP 1: RETRIEVAL (Search Engine)
        with st.status("🔍 Step 1: Retrieving Web Evidence...", expanded=True) as status:
            evidence_data = get_web_evidence(claim_text)
            
            if not evidence_data:
                st.error("No evidence found. Please try a different claim.")
                st.stop()
            
            evidence_text = " ".join([e['snippet'] for e in evidence_data])
            st.write(f"✅ Found {len(evidence_data)} relevant medical sources.")
            
            # STEP 2: REASONING (Gemini)
            st.write(f"🧠 Step 2: Reasoning with {model_choice}...")
            
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                uploaded_file.seek(0)
                img_obj = Image.open(uploaded_file).convert('RGB')
                
                prompt = f"""
                Analyze the following medical claim using the provided search evidence and image.
                
                CLAIM: {claim_text}
                SEARCH EVIDENCE: {evidence_text}
                
                TASK:
                1. Cross-reference the claim with the evidence.
                2. Evaluate if the image supports or contradicts the scientific consensus.
                3. Return a JSON with: 'prediction' (Real/Fake), 'confidence' (0-100), and 'explanation'.
                """
                
                response = client.models.generate_content(
                    model=model_choice,
                    contents=[prompt, img_obj],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        # Safety override to prevent unnecessary blocks on medical debunking
                        safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}],
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
                
                res = response.parsed
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                # --- DISPLAY AI VERDICT ---
                st.divider()
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    is_real = res.prediction.strip().lower() == "real"
                    bg_color = "#d4edda" if is_real else "#f8d7da"
                    txt_color = "#155724" if is_real else "#721c24"
                    st.markdown(f'<div class="verdict-box" style="background-color: {bg_color}; color: {txt_color};">Verdict: {res.prediction.upper()}</div>', unsafe_allow_html=True)
                    st.metric("Reasoning Confidence", f"{res.confidence}%")
                with res_col2:
                    st.subheader("Analysis Summary")
                    st.write(res.explanation)

            except Exception as e:
                # --- GRACEFUL FALLBACK (If Gemini Fails) ---
                status.update(label="⚠️ AI Quota Reached", state="error", expanded=False)
                st.warning("The AI Reasoning engine is at capacity (Quota 429). Displaying retrieved evidence for manual review.")
                
                st.divider()
                st.subheader("🔎 Raw Search Evidence")
                for source in evidence_data:
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>{source['title']}</strong><br>
                        <p>{source['snippet']}</p>
                        <a href="{source['link']}" target="_blank">View Full Article</a>
                    </div>
                    """, unsafe_allow_html=True)
