import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ==================== Configuration ====================
# Pipeline 1: Department classification (your fine-tuned model)
MODEL_DEPT = "JR-2026/CustomModel_medical"
# Pipeline 2: Urgency assessment (using pre-trained sentiment analysis model)
MODEL_URGENCY = "nlptown/bert-base-multilingual-uncased-sentiment"  # Output 1-5 stars

# Department label mapping (not needed if model directly outputs label string)
# For safety, read id2label from model config
@st.cache_resource
def load_department_pipeline():
    # Use pipeline to automatically load model and tokenizer
    return pipeline("text-classification", model=MODEL_DEPT)

@st.cache_resource
def load_urgency_pipeline():
    # Load sentiment analysis pipeline
    return pipeline("sentiment-analysis", model=MODEL_URGENCY)

def map_urgency(score):
    """Map sentiment model confidence to urgency level"""
    if score < 0.4:
        return "Low (recommend general outpatient)"
    elif score < 0.7:
        return "Medium (recommend prompt consultation)"
    else:
        return "High (recommend immediate medical attention)"

# ==================== UI ====================
st.set_page_config(page_title="MediTriage AI", page_icon="🏥")
st.title("🏥 MediTriage AI - Smart Medical Triage Assistant")
st.markdown("Please describe your symptoms, and the system will recommend a department and assess urgency.")

# User input
user_input = st.text_area("Symptom description", height=150, placeholder="e.g., I've had a headache and fever for two days...")

if st.button("Start Triage", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a symptom description.")
    else:
        with st.spinner("Analyzing..."):
            # Pipeline 1: Department classification
            dept_result = load_department_pipeline()(user_input)
            dept_label = dept_result[0]['label']
            dept_score = dept_result[0]['score']

            # Pipeline 2: Urgency assessment
            urgency_result = load_urgency_pipeline()(user_input)
            urgency_confidence = urgency_result[0]['score']
            urgency_level = map_urgency(urgency_confidence)

        # Display results
        st.success("Analysis complete")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Recommended Department")
            st.metric(value=f"{dept_score:.2%}", label=dept_label)

        with col2:
            st.subheader("⚠️ Urgency Level")
            st.metric(value=f"{urgency_confidence:.2%}", label=urgency_level)

        # Additional note
        st.info("Note: This system is for reference only. Final diagnosis should be made by a qualified physician.")
