import faiss

def generate_embeddings(chunks,model):

    """
    Generates embeddings for a list of text chunks using a SentenceTransformer model.

    Args:
        chunks (list): A list of dicts with a 'text' key.
        model (SentenceTransformer): The embedding model to use.

    Returns:
        np.ndarray: A 2D array of embeddings, one row per chunk.
    """

    text_list = []

    for chunk in chunks:
        text_list.append(chunk["text"])

    embeddings = model.encode(text_list)

    return embeddings

def build_faiss_index(embeddings):

    """
    Builds a FAISS index from a set of embeddings.

    Args:
        embeddings (np.ndarray): A 2D float32 array of embeddings.

    Returns:
        faiss.IndexFlatL2: A FAISS index populated with the provided embeddings.
    """
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return index

if __name__ == "__main__":
    from ingestion import exctract_text_from_pdf,chunk_text
    from vector_store import save_index,load_index

    pages = exctract_text_from_pdf("data/test.pdf")
    chunks = chunk_text(pages)
    embedings = generate_embeddings(chunks)
    faiss_index = build_faiss_index(embedings)
    save_index(faiss_index,chunks)

    print(f"Embeddings shape: {embedings.shape}")
    print(f"Total chunks: {len(chunks)}")
    print("Index and chunks saved.")