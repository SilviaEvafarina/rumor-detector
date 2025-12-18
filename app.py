import streamlit as st
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types
import requests

# --- 1. PAGE CONFIG & STYLING (Original Design) ---
st.set_page_config(page_title="Med-Verify AI", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; background-color: #007bff; color: white; }
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 22px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. SEARCH ENGINE LOGIC (Evidence Retrieval) ---
@st.cache_data(show_spinner=False)
def get_external_evidence(query):
    # This fulfills your requirement for Google Search Engine retrieval
    url = f"https://www.googleapis.com/customsearch/v1?key={st.secrets['GOOGLE_SEARCH_KEY']}&cx={st.secrets['SEARCH_ENGINE_ID']}&q={query}"
    try:
        res = requests.get(url, timeout=10).json()
        items = res.get('items', [])
        return " | ".join([f"Source: {i['link']} Content: {i['snippet']}" for i in items[:3]])
    except:
        return "No specific web evidence found."

# --- 3. SIDEBAR (With Model Choice & Doctor Intro) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=80)
    st.title("Settings")
    model_choice = st.selectbox("Select Model Engine:", 
                                options=["gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-2.0-flash"],
                                index=0)
    st.info("This app uses a two-step modeling technique: Google Search for evidence and Gemini for reasoning.")

# --- 4. MAIN UI ---
st.title("🩺 Med-Verify: Medical Rumor Detector")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Step 1: Input Claim & Image")
    claim_text = st.text_area("What is the medical rumor?", height=120)
    uploaded_file = st.file_uploader("Upload associated image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

with col2:
    st.subheader("Step 2: Analysis & Verdict")
    if st.button("🔍 Run Fact-Check"):
        if claim_text and uploaded_file:
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            
            with st.status("Performing Agentic Search...", expanded=True) as status:
                # 1. RETRIEVAL (Search Engine)
                evidence = get_external_evidence(claim_text)
                st.write("✅ Web evidence retrieved.")
                
                # 2. REASONING (Gemini)
                st.write(f"🧠 Reasoning with {model_choice}...")
                img = Image.open(uploaded_file).convert('RGB')
                
                prompt = f"""
                You are a professional fact-checker. 
                CLAIM TO INVESTIGATE: {claim_text}
                GOOGLE SEARCH EVIDENCE: {evidence}
                
                YOUR TASK:
                Analyze if the claim is supported by the search evidence. 
                Look at the uploaded image and see if it provides real proof or if it's misleading.
                
                Provide a structured JSON response. 
                If evidence is contradictory, set a lower confidence score.
                """
                
                try:
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}],
                            response_schema={
                                "type": "object",
                                "properties": {
                                    "prediction": {"type": "string", "enum": ["Real", "Fake", "Inconclusive"]},
                                    "confidence": {"type": "integer"},
                                    "reasoning": {"type": "string"}
                                },
                                "required": ["prediction", "confidence", "reasoning"]
                            }
                        )
                    )
                    res = response.parsed
                    status.update(label="Analysis Finished", state="complete")
                except Exception as e:
                    st.error(f"Reasoning Error: {str(e)}")
                    st.stop()

            # --- DISPLAY RESULTS ---
            if res:
                color = "#d4edda" if res.prediction == "Real" else "#f8d7da"
                text_color = "#155724" if res.prediction == "Real" else "#721c24"
                
                st.markdown(f'<div class="verdict-box" style="background-color: {color}; color: {text_color};">Verdict: {res.prediction.upper()}</div>', unsafe_allow_html=True)
                st.metric("Reasoning Confidence", f"{res.confidence}%")
                st.write("**Detailed Analysis:**")
                st.write(res.reasoning)
        else:
            st.warning("Please provide both a claim and an image.")
