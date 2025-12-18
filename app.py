# --- 5. SIDEBAR: MODEL SELECTION & INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=100)
    st.title("Settings & About")
    
    # NEW: Model Selection Dropdown
    st.subheader("⚙️ Configuration")
    selected_model_name = st.selectbox(
        "Choose Gemini Model:",
        options=[
            "gemini-2.5-flash-lite", 
            "gemini-2.5-flash", 
            "gemini-2.0-flash"
        ],
        index=0, # Default to Flash-Lite for best quota
        help="If you get a 'Resource Exhausted' error, try switching to Flash-Lite."
    )
    
    st.info(f"**Current Model:** {selected_model_name}")
    st.divider()
    
    st.write("**What can this Dashboard do?**")
    st.caption("1. Multimodal Analysis\n2. Agentic RAG\n3. Evidence Reasoning")

# --- 7. EXECUTION (Update the AI call) ---
# Find the part where you call Gemini and update the 'model' parameter:

if submit_btn:
    # ... previous code (searching, image processing) ...
    
    try:
        response = client.models.generate_content(
            model=selected_model_name,  # ✅ Now uses the sidebar selection
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
        res = response.parsed
    except Exception as e:
        st.error(f"Error using {selected_model_name}: {str(e)}")
        st.info("💡 Tip: Try selecting a different model from the sidebar.")
        st.stop()
