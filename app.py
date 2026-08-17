"""
Clinical AI Assistant - Interactive Streamlit Web UI.
Features:
1. Patient Vitals Input Panel (eGFR, HbA1c, BP).
2. One-Click Clinical Case Presets.
3. Side-by-Side Comparison: Base Foundation Model vs. Fine-Tuned Clinical LoRA Model.
4. Real-time Inference Telemetry (Latency & Tokens/sec).
"""

import time
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MERGED_DIR = "./merged_model"
ADAPTER_DIR = "./final_adapter"

st.set_page_config(
    page_title="Clinical AI - Domain Fine-Tuned Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-box { background: #F3F4F6; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_tokenizer():
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)

    # Load Fine-Tuned Model
    if os.path.exists(MERGED_DIR):
        tuned_model = AutoModelForCausalLM.from_pretrained(
            MERGED_DIR,
            torch_dtype=torch.float32
        ).to(device)
    elif os.path.exists(ADAPTER_DIR):
        tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    else:
        tuned_model = base_model

    return tokenizer, base_model, tuned_model, device


def generate_response(model, tokenizer, prompt, device):
    formatted = (
        "<|im_start|>system\n"
        "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency_sec = time.perf_counter() - start_time
    num_tokens = output.shape[1] - inputs.input_ids.shape[1]
    tok_per_sec = (num_tokens / latency_sec) if latency_sec > 0 else 0.0

    text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return text, round(latency_sec, 3), round(tok_per_sec, 1), num_tokens


# App Layout
st.markdown("<div class='main-header'>🩺 Clinical Decision Support AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Fine-Tuned Domain LLM (Qwen2.5-0.5B + PEFT/LoRA on Apple Silicon MPS)</div>", unsafe_allow_html=True)

tokenizer, base_model, tuned_model, device = load_models_and_tokenizer()

# Sidebar: Patient Vitals & Controls
with st.sidebar:
    st.header("📋 Patient Clinical Profile")
    egfr = st.slider("eGFR (mL/min/1.73m²)", 10, 120, 26, help="eGFR < 30 indicates severe CKD (Metformin contraindicated)")
    hba1c = st.slider("HbA1c (%)", 4.0, 15.0, 8.6)
    bp_sys = st.number_input("Systolic BP (mmHg)", value=138)
    bp_dia = st.number_input("Diastolic BP (mmHg)", value=84)

    st.markdown("---")
    st.header("⚡ Quick Clinical Presets")
    preset = st.radio(
        "Choose a Case:",
        [
            "Custom Query",
            "1. CKD + T2D (Metformin Contraindication)",
            "2. Arterial Blood Gas (ABG) Interpretation",
            "3. Drug Interaction: Linezolid + SSRIs",
            "4. Acute Calculous Cholecystitis Management"
        ]
    )

# Prompt Construction
default_prompt = ""
if preset == "1. CKD + T2D (Metformin Contraindication)":
    default_prompt = f"A 62-year-old male with chronic kidney disease (eGFR {egfr} mL/min/1.73m²) and type 2 diabetes presents for glycemic management. His current HbA1c is {hba1c}%. Which first-line agent is contraindicated, and what alternative should be recommended?"
elif preset == "2. Arterial Blood Gas (ABG) Interpretation":
    default_prompt = "Interpret the following arterial blood gas (ABG) result: pH 7.28, PaCO2 58 mmHg, HCO3- 26 mEq/L, PaO2 65 mmHg on room air. Provide the primary acid-base disturbance and next clinical steps."
elif preset == "3. Drug Interaction: Linezolid + SSRIs":
    default_prompt = "Explain the pharmacological mechanism and risk of serotonin syndrome when co-administering Linezolid with SSRIs (e.g., Sertraline). What is the recommended management?"
elif preset == "4. Acute Calculous Cholecystitis Management":
    default_prompt = "Outline the diagnostic confirmation and initial management pathway for acute calculous cholecystitis in a hemodynamically stable patient."

user_query = st.text_area("Clinical Inquiry / Patient Presentation:", value=default_prompt, height=120)

col1, col2 = st.columns(2)

if st.button("🚀 Run Clinical Analysis", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a clinical question.")
    else:
        with col1:
            st.subheader("🤖 Base Foundation Model")
            with st.spinner("Generating base response..."):
                base_text, base_lat, base_tps, base_n = generate_response(base_model, tokenizer, user_query, device)
            st.write(base_text)
            st.caption(f"⚡ Latency: **{base_lat}s** | Speed: **{base_tps} tok/s** | Tokens: **{base_n}**")

        with col2:
            st.subheader("🩺 Fine-Tuned Clinical (LoRA) Model")
            with st.spinner("Generating clinical LoRA response..."):
                tuned_text, tuned_lat, tuned_tps, tuned_n = generate_response(tuned_model, tokenizer, user_query, device)
            st.success(tuned_text)
            st.caption(f"⚡ Latency: **{tuned_lat}s** | Speed: **{tuned_tps} tok/s** | Tokens: **{tuned_n}**")
