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

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Docu Mind")
file = st.file_uploader("Upload", type="pdf")

if file is None and st.session_state.index is not None:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.messages = []

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


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    if message.get("sources"):
        with st.expander("sources"):
                for chunk in message["sources"]:
                    st.write(f"Page {chunk["page"]}")
                    with st.expander("text"):
                        st.write(f" - {chunk["chunk"]}")

    
if user_input and st.session_state.index is not None:
    st.session_state.messages.append(
        {
            "role": "user", 
            "content": user_input
            }
        )

    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking"):
        retrieved = retrieve_chunks(user_input, st.session_state.index, st.session_state.chunks, st.session_state.model)

        is_relevent = retrieved[0]["score"] <= 1.5

        if not is_relevent:
            answer = "The text does not contain any thing rellevent to the question"
        else:
            answer = generate_awnswer(user_input, retrieved)

    st.session_state.messages.append(
        {
            "role": "assistant", 
            "content": answer,
            "sources": retrieved if is_relevent else None
            }
        )

    with st.chat_message("assistant"):
        st.write(answer)

    if is_relevent:
        with st.expander("sources"):
            for chunk in retrieved:
                st.write(f"Page {chunk["page"]}")
                with st.expander("text"):
                    st.write(f" - {chunk["chunk"]}")

elif user_input and st.session_state.index is None:
    
    st.warning("Please upload a PDF first.")