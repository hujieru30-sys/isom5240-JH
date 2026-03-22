import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

st.title("MediAssist AI - 医疗咨询辅助系统")
st.write("输入医学问题，系统将推荐最佳选项")

question = st.text_area("问题")
options = [st.text_input(f"选项 {chr(65+i)}") for i in range(4)]

if st.button("推荐答案"):
    model = DistilBertForSequenceClassification.from_pretrained("./results")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    pred = predict_question(question, options)
    st.write(f"推荐选项：{pred}")
