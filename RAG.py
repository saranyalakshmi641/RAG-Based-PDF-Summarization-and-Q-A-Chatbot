import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import fitz
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import os

# Configure Gemini API Key
genai.configure(api_key='AIzaSyDCgKCKWucKNjvph3CF9oeB79dJ-31pLfg')  # Replace with your key

# ---------------------------------------
# PDF TEXT EXTRACTION FUNCTION
# ---------------------------------------
def ext_text(pdf_name):
    text = ""
    with fitz.open(pdf_name) as pdf:
        for i in pdf:
            text += i.get_text()
    return text

# ---------------------------------------
# RETRIEVAL FUNCTION
# ---------------------------------------
def retrive(query, embed_model, index, chunks, top_k=3):
    query_vec = embed_model.encode([query]).astype("float32")
    distance, indices = index.search(query_vec, top_k)
    retrived = [chunks[i] for i in indices[0]]
    return "\n\n".join(retrived)

# ---------------------------------------
# ANSWER GENERATION FUNCTION
# ---------------------------------------
def answer(query, embed_model, index, chunks):
    context = retrive(query, embed_model, index, chunks)
    prompt = f"""
    You are a resume assistant. Based on the context below, answer the user's question.
    Context:
    {context}
    Question: {query}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.title("📄 AI Resume Assistant using RAG")
st.write("Ask questions about your resume!")

uploaded_pdf = st.file_uploader("Upload your resume PDF", type=["pdf"])

if uploaded_pdf:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    st.success("PDF uploaded successfully ✔")

    # Extract text
    data = ext_text("temp.pdf")

    # Split into chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(data)

    # Create embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.info("Embeddings & FAISS Index Created Successfully 🔍")

    # Query box
    question = st.text_input("Ask a question about your resume:")
    if st.button("Search"):
        with st.spinner("Generating answer..."):
            res = answer(question, embed_model, index, chunks)
            st.write("### 🧠 Answer:")
            st.write(res)
