"""
Clinical AI Assistant - Interactive Streamlit Web UI.
Production 7B Decision Support Engine.
Features:
1. USMLE Clinical Reasoning & Diagnostic Vignette Presets.
2. Transparent Adapter Detection (shows whether LoRA weights are loaded or base model is active).
3. Side-by-Side Comparison: Base 7B Model vs. Fine-Tuned QLoRA 7B Model.
4. Strict Diagnostic Option Parsing & Real-Time Inference Telemetry.
"""

import time
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from data_utils import extract_predicted_option

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "./final_qlora_7b_adapter"
ADAPTER_WEIGHTS_FILE = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")

st.set_page_config(
    page_title="Clinical AI - 7B QLoRA Diagnostic Assistant",
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
def load_clinical_model():
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_cuda:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)

    # Transparent Adapter Detection
    has_adapter = os.path.exists(ADAPTER_WEIGHTS_FILE) and os.path.getsize(ADAPTER_WEIGHTS_FILE) > 1000
    if has_adapter:
        try:
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            adapter_loaded = True
        except Exception:
            model = base_model
            adapter_loaded = False
    else:
        model = base_model
        adapter_loaded = False

    return tokenizer, model, device, is_cuda, adapter_loaded


def generate_clinical_response(model, tokenizer, prompt, device, is_base_mode=False):
    formatted = (
        "<|im_start|>system\n"
        "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        if is_base_mode and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.2,
                    do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id
                )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
    latency_sec = time.perf_counter() - start_time
    num_tokens = output.shape[1] - inputs.input_ids.shape[1]
    tok_per_sec = (num_tokens / latency_sec) if latency_sec > 0 else 0.0

    raw_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    parsed_option = extract_predicted_option(raw_text)
    return raw_text, parsed_option, round(latency_sec, 3), round(tok_per_sec, 1), num_tokens


# UI Header
st.markdown("<div class='main-header'>🩺 Clinical Decision Support AI (7B QLoRA)</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Domain-Specific Fine-Tuned LLM (Qwen2.5-7B-Instruct + 4-Bit NF4 QLoRA on USMLE Reasoning Cases)</div>", unsafe_allow_html=True)

tokenizer, model, device, is_cuda, adapter_loaded = load_clinical_model()

# Transparent Status Bar
status_col1, status_col2 = st.columns(2)
with status_col1:
    if is_cuda:
        st.success("⚡ **Compute:** NVIDIA CUDA GPU (4-Bit NF4 Quantization Active)")
    else:
        st.info("🍎 **Compute:** Local Apple Silicon (FP16 Mode). For 4-bit GPU acceleration, run on Google Colab.")

with status_col2:
    if adapter_loaded:
        st.success("🟢 **Model Status:** Fine-Tuned QLoRA Clinical Adapter Loaded (`final_qlora_7b_adapter/`)")
    else:
        st.warning("🟡 **Model Status:** Base 7B Model Active (Trained adapter weights will appear here once downloaded from Colab training run).")

# Sidebar: Quick Clinical Presets
with st.sidebar:
    st.header("⚡ USMLE Clinical Cases")
    preset = st.radio(
        "Select a Clinical Vignette:",
        [
            "Custom Clinical Vignette",
            "1. Rust-Colored Sputum Pneumonia",
            "2. Severe CKD & Type 2 Diabetes",
            "3. Drug Interaction: Serotonin Syndrome",
            "4. Acute Calculous Cholecystitis"
        ]
    )

# Prompt Construction
default_prompt = ""
if preset == "1. Rust-Colored Sputum Pneumonia":
    default_prompt = "A 68-year-old male presents with high fever (102.5°F), productive cough with rust-colored sputum, and shortness of breath for 3 days. Exam shows bronchial breath sounds in the right lower lung base.\n\nA: Legionella pneumophila\nB: Streptococcus pneumoniae\nC: Haemophilus influenzae\nD: Staphylococcus aureus\nE: Klebsiella pneumoniae\n\nWhat is the most likely pathogen?"
elif preset == "2. Severe CKD & Type 2 Diabetes":
    default_prompt = "A 62-year-old male with chronic kidney disease (eGFR 24 mL/min/1.73m²) and type 2 diabetes presents for glycemic management. His current HbA1c is 8.6%.\n\nA: Metformin\nB: Empagliflozin\nC: Linagliptin\nD: Glyburide\nE: Pioglitazone\n\nWhich first-line antidiabetic agent is contraindicated?"
elif preset == "3. Drug Interaction: Serotonin Syndrome":
    default_prompt = "A 45-year-old female on Sertraline for depression is prescribed Linezolid for MRSA skin infection. 48 hours later, she develops hyperthermia, tremor, and clonus.\n\nA: Neuroleptic Malignant Syndrome\nB: Serotonin Syndrome via MAO inhibition\nC: Malignant Hyperthermia\nD: Anticholinergic Toxicity\nE: Stevens-Johnson Syndrome\n\nWhat is the underlying pharmacological diagnosis?"
elif preset == "4. Acute Calculous Cholecystitis":
    default_prompt = "A 44-year-old female presents with acute right upper quadrant pain radiating to the right scapula, fever, and positive Murphy's sign.\n\nA: Abdominal CT with IV contrast\nB: Right Upper Quadrant Ultrasound\nC: HIDA scan\nD: ERCP\nE: Magnetic Resonance Cholangiopancreatography\n\nWhat is the initial diagnostic test of choice?"

user_query = st.text_area("Clinical Vignette & Options:", value=default_prompt, height=160)

col1, col2 = st.columns(2)

if st.button("🚀 Run 7B Clinical Diagnostic Inference", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a clinical question.")
    else:
        with col1:
            st.subheader("🤖 Base Qwen2.5-7B Foundation")
            with st.spinner("Generating base 7B response..."):
                base_text, base_opt, base_lat, base_tps, base_n = generate_clinical_response(model, tokenizer, user_query, device=device, is_base_mode=True)
            st.write(base_text)
            if base_opt != "NONE":
                st.info(f"🎯 Parsed Option: **Option {base_opt}**")
            st.caption(f"⚡ Latency: **{base_lat}s** | Speed: **{base_tps} tok/s** | Tokens: **{base_n}**")

        with col2:
            st.subheader("🩺 Fine-Tuned 7B QLoRA Clinical Model")
            with st.spinner("Generating fine-tuned response..."):
                tuned_text, tuned_opt, tuned_lat, tuned_tps, tuned_n = generate_clinical_response(model, tokenizer, user_query, device=device, is_base_mode=False)
            st.success(tuned_text)
            if tuned_opt != "NONE":
                st.success(f"🎯 Parsed Option: **Option {tuned_opt}**")
            st.caption(f"⚡ Latency: **{tuned_lat}s** | Speed: **{tuned_tps} tok/s** | Tokens: **{tuned_n}**")
