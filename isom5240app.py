import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="JR-2026/CustomModel_medical")

st.title("medical classification")
user_input = st.text_input("Describe the condition of the illness：")

if user_input:
    classifier = load_model()
    result = classifier(user_input)
    
    st.write("预测结果：")
    for res in result:
        st.write(f"{res['label']}: {res['score']:.4f}")
