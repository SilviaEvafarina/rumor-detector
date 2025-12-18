import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Rumor & Fake News Detector",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .real { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .fake { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .unsure { background-color: #fff3cd; color: #856404; border: 2px solid #ffeeba; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONFIG & INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2919/2919600.png", width=100)
    st.title("Settings")
    
    # Secure API Key Entry
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get this from Google AI Studio")
    
    st.markdown("---")
    st.markdown("### ℹ️ About this App")
    st.info(
        """
        **What is a Rumor?**
        Unverified information that spreads rapidly, often combining misleading images with sensational text.
        
        **What does this tool do?**
        It uses **AI Agents** (Gemini 2.0) to:
        1. 👀 **See** the image.
        2. 📖 **Read** the claim.
        3. 🌍 **Search** Google for real-time facts.
        4. ⚖️ **Verdict**: Real or Fake?
        """
    )
    st.markdown("---")
    st.caption("Powered by Gemini 2.0 Flash & Google Search")

# --- MAIN PAGE HEADER ---
st.title("🕵️‍♀️ AI Multimedia Fact-Checker")
st.markdown("### Analyze images and text claims in real-time to detect misinformation.")

# --- MAIN CONTENT LAYOUT ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Evidence")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.subheader("2. Enter Claim")
    user_claim = st.text_area("What is the rumor or claim about this image?", 
                              placeholder="Example: This image shows a shark swimming on a highway during the hurricane...",
                              height=150)

# --- ANALYSIS LOGIC ---
with col2:
    st.subheader("3. Investigation Results")
    
    if st.button("🚀 Run Investigation", disabled=not (uploaded_file and user_claim and api_key)):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            try:
                # Initialize Client with Agentic Tools
                client = genai.Client(api_key=api_key)
                search_tool = types.Tool(google_search=types.GoogleSearch())
                
                # Dynamic Spinner
                with st.status("🕵️ Agent is working...", expanded=True) as status:
                    st.write("👀 Analyzing image content...")
                    time.sleep(1)
                    st.write("🌍 Cross-referencing claim with Google Search...")
                    time.sleep(1)
                    st.write("🧠 Synthesizing verdict...")
                    
                    # The Prompt
                    prompt = f"""
                    You are an expert Fact-Checker and Misinformation Detector.
                    
                    TASK:
                    1. Analyze the uploaded image.
                    2. Read the claim: "{user_claim}"
                    3. Use Google Search to verify if this image and claim combination is TRUE or FAKE.
                    4. Check if the image is manipulated or taken out of context.
                    
                    OUTPUT FORMAT:
                    - Verdict: (REAL / FAKE / MISLEADING)
                    - Confidence Score: (0-100%)
                    - Explanation: A concise summary of why.
                    - Sources: List key sources found via search.
                    
                    Return the output in clean Markdown.
                    """
                    
                    # Call Gemini 2.0 Flash
                    response = client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=[image, prompt],
                        config=types.GenerateContentConfig(
                            tools=[search_tool],
                            temperature=0.3
                        )
                    )
                    
                    status.update(label="✅ Investigation Complete!", state="complete", expanded=False)

                # --- PARSE & DISPLAY RESULT ---
                result_text = response.text
                
                # Simple parsing logic for the verdict styling
                if "FAKE" in result_text.upper():
                    st.markdown('<div class="verdict-box fake">🚨 VERDICT: FAKE</div>', unsafe_allow_html=True)
                elif "REAL" in result_text.upper():
                    st.markdown('<div class="verdict-box real">✅ VERDICT: REAL</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict-box unsure">⚠️ VERDICT: MISLEADING / UNVERIFIED</div>', unsafe_allow_html=True)
                
                st.markdown("### 📝 Detailed Report")
                st.markdown(result_text)
                
                # Grounding Metadata (Sources)
                if response.candidates[0].grounding_metadata.search_entry_point:
                    st.markdown("---")
                    st.markdown(response.candidates[0].grounding_metadata.search_entry_point.rendered_content, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Ensure your API key has access to Gemini 2.0 Flash and Google Search Grounding.")

    elif not api_key:
        st.info("👈 Waiting for API Key...")
    elif not uploaded_file or not user_claim:
        st.info("👈 Waiting for image and text input...")