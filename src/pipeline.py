from ingestion import exctract_text_from_pdf,chunk_text
from embeddings import generate_embeddings,build_faiss_index
from vector_store import save_index,load_index
from retrieval import retrieve_chunks
from generator import generate_awnswer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

pages = exctract_text_from_pdf("data/test.pdf")
chunks = chunk_text(pages)
embeddings = generate_embeddings(chunks,model)
index = build_faiss_index(embeddings)
save_index(index,chunks)

query = input("Input: ")

r_chunks = retrieve_chunks(query,index,chunks,model)

awnswer = generate_awnswer(query,r_chunks,)



print(awnswer)
