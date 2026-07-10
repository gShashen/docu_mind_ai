from sentence_transformers import SentenceTransformer
import faiss

def generate_embeddings(chunks,model):

    text_list = []

    for chunk in chunks:
        text_list.append(chunk["text"])

    embeddings = model.encode(text_list)

    return embeddings

def build_faiss_index(embeddings):
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return index

if __name__ == "__main__":
    from ingestion import exctract_text_from_pdf,chunk_text
    from vector_store import save_index,load_index

    pages = exctract_text_from_pdf("data/test.pdf")
    chunks = chunk_text(pages)
    embedings = generate_embeddings(chunks,SentenceTransformer("all-MiniLM-L6-v2"))
    faiss_index = build_faiss_index(embedings)
    save_index(faiss_index,chunks)

    print(f"Embeddings shape: {embedings.shape}")
    print(f"Total chunks: {len(chunks)}")
    print("Index and chunks saved.")