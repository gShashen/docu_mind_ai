# Docu Mind

Docu mind is RAG based AI chatbot which lets you upload a "PDF" and ask questions about it. The AI will generate you a answer based on the file with page numbers as sources and the piece of text from that page.

## Stack
 - python - core language
 - Claude - LLM
 - pymupdf - Read PDFs
 - sentence transformers - generates Embeddings
 - streamlit - UI
 - faiss - Search and stor embeddings

## How to Run Locally

1. Clone the repo
git clone https://github.com/gShashen/docu_mind_ai.git

2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Add your API key to a `.env` file
ANTHROPIC_API_KEY=your_key_here

5. Run the app
streamlit run app.py

## Architecture

PDF → Ingestion → Chunking → Embedding → FAISS Index → Retrieval → Claude API → Answer

- **Ingestion** - Extracts text from each page using PyMuPDF
- **Chunking** - Breaks extracted text into overlapping word chunks
- **Embedding** - Generates embeddings for each chunk using SentenceTransformer
- **FAISS Index** - Stores and indexes the embeddings for fast similarity search
- **Retrieval** - Embeds the user query and retrieves the most relevant chunks
- **Claude API** - Generates an answer based on the retrieved chunks
- **Answer** - Displays the answer with page sources in the Streamlit UI

## Demo

[Watch Demo] (2026-07-12_documind_demo.mp4)