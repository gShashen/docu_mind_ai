import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from sentence_transformers import SentenceTransformer
from ingestion import exctract_text_from_pdf, chunk_text
from embeddings import generate_embeddings, build_faiss_index
from retrieval import retrieve_chunks
from generator import generate_awnswer
import pymupdf

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("Docu Mind")
file = st.file_uploader("Upload", type="pdf")

if file is None and st.session_state.index is not None:
    st.session_state.index = None
    st.session_state.chunks = None

if file is not None:
    with st.spinner("Processing"):
        raw = pymupdf.open(stream=file.read(),filetype="pdf")
        pages = exctract_text_from_pdf(raw)
        chunks = chunk_text(pages,chunk_size = 100, overlap = 20)
        embendings = generate_embeddings(chunks,st.session_state.model)
        st.session_state.index = build_faiss_index(embendings)
        st.session_state.chunks = chunks

        st.success("PDF ready ask question below")


user_input = st.chat_input("Ask a question about your document")



if user_input and st.session_state.index is not None:
    with st.chat_message("user"):
        st.write(user_input)
    with st.spinner("Thinking"):
        retrieved = retrieve_chunks(user_input, st.session_state.index, st.session_state.chunks, st.session_state.model)
        answer = generate_awnswer(user_input, retrieved)
    with st.chat_message("assistant"):
        st.write(answer)

elif user_input and st.session_state.index is None:
    st.warning("Please upload a PDF first.")
