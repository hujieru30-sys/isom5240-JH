import streamlit as st
from transformers import pipeline

def main():
    sentiment_pipeline = pipeline(model="JR-2026/CustomModel_medical"")

    st.title("Sentiment Analysis with HuggingFace Spaces")
    st.write("Enter a sentence to analyze its sentiment:")

    user_input = st.text_input("")
    if user_input:
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.write(f"Sentiment: ")
        for r in result:
            st.write(f"{r['label']}: {r['score']:.4f}")

if __name__ == "__main__":
    main()
