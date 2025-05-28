import streamlit as st
import os
import requests
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to ask question using Hugging Face API
def ask_question(context, question):
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit app
st.title("PDF Question Answering with Hugging Face")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    context = extract_text_from_pdf(uploaded_file)
    st.success("PDF uploaded and text extracted!")
    st.subheader("Ask a question about the PDF")

    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            answer = ask_question(context, question)
            st.subheader("Answer:")
            st.write(answer.get("answer", "No answer found."))
        else:
            st.warning("Please enter a question.")
