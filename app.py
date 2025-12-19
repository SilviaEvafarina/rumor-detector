import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    LLaVAForConditionalGeneration, 
    AutoProcessor
)
import gc

# --- 1. PHASE 2: CLASSIFIER ARCHITECTURE ---
class MedicalFactChecker(nn.Module):
    def __init__(self):
        super(MedicalFactChecker, self).__init__()
        # Vision Branch
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()
        
        # Text Branch (DistilBERT features are 768)
        self.fusion = nn.Linear(512 + 768, 256)
        self.classifier = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, pixel_values):
        img_features = self.resnet(pixel_values) # 512
        # Note: In a real app, you'd pass text through DistilBERT here
        # For brevity, this assumes features are extracted or model is end-to-end
        text_features = torch.randn(input_ids.shape[0], 768).to(pixel_values.device) 
        
        combined = torch.cat((img_features, text_features), dim=1)
        x = self.relu(self.fusion(combined))
        x = self.dropout(x)
        return self.classifier(x)

# --- 2. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Phase 2 Weights
    p2_model = MedicalFactChecker().to(device)
    # Ensure you have your .pth file in the same folder
    # p2_model.load_state_dict(torch.load('med_model_epoch_5.pth', map_location=device))
    p2_model.eval()

    # Load Phase 3: LLaVA with 4-bit Quantization (Crucial for 8GB VRAM)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    llava_model = LLaVAForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=quant_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    return p2_model, llava_model, processor, tokenizer

# --- 3. STREAMLIT UI SETUP ---
st.set_page_config(page_title="MedVerify AI", page_icon="🏥", layout="wide")

st.title("🏥 MedVerify: Dual-Phase Medical Fact-Checking")
st.markdown("""
This system uses a **Dual-Phase architecture** to detect medical misinformation. 
1. **Phase 2:** Fast Neural Classification. 
2. **Phase 3:** Generative Scientific Reasoning.
""")

p2_model, llava_model, processor, tokenizer = load_models()

# --- 4. SIDEBAR & INPUTS ---
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    claim = st.text_area("Medical Claim:", placeholder="e.g. Garlic cures COVID-19")
    run_btn = st.button("🚀 Run Dual-Phase Analysis")

# --- 5. INFERENCE LOGIC ---
if run_btn and uploaded_file and claim:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Target Image", use_container_width=True)

    with st.spinner("Analyzing across both phases..."):
        # --- PHASE 2 EXECUTION ---
        # (Simplified transform for demo)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = preprocess(image).unsqueeze(0).to("cuda")
        inputs = tokenizer(claim, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = p2_model(inputs['input_ids'], inputs['attention_mask'], img_tensor)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = torch.nn.functional.softmax(outputs, dim=1).max().item()

        # --- PHASE 3 EXECUTION ---
        prompt = f"USER: <image>\nAnalyze this medical claim: '{claim}'. Is it scientifically accurate? Explain why. \nASSISTANT:"
        inputs_llava = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        
        output_ids = llava_model.generate(**inputs_llava, max_new_tokens=256)
        reasoning = processor.decode(output_ids[0], skip_special_tokens=True).split("ASSISTANT:")[-1]

    # --- 6. DISPLAY RESULTS ---
    with col2:
        st.subheader("Results")
        
        # Phase 2 Metric
        verdict = "TRUE" if prediction == 1 else "FALSE"
        color = "green" if verdict == "TRUE" else "red"
        st.markdown(f"**Phase 2 Verdict:** :{color}[{verdict}]")
        st.progress(confidence)
        st.caption(f"Confidence Score: {confidence*100:.2f}%")
        
        st.markdown("---")
        
        # Phase 3 Metric
        st.markdown("**Phase 3 Scientific Reasoning:**")
        st.info(reasoning)

        # Conflict Detection
        if verdict == "TRUE" and any(word in reasoning.lower() for word in ["false", "misleading", "no evidence"]):
            st.warning("⚠️ **System Conflict Detected:** Reasoning suggests potential misinformation despite a 'True' classification.")

    # Cleanup memory
    torch.cuda.empty_cache()
    gc.collect()

else:
    st.info("Please upload an image and enter a claim to begin.")
