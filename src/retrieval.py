from sentence_transformers import SentenceTransformer

def retrieve_chunks(query,index,chunks,model,top_k=6):

    """
    Retrieves the most relevant chunks for a given query using FAISS.

    Args:
        query (str): The user's question.
        index (faiss.IndexFlatL2): The FAISS index to search.
        chunks (list): A list of dicts with keys 'chunk_id', 'page', and 'text'.
        model (SentenceTransformer): The embedding model used to encode the query.
        top_k (int): Number of top results to retrieve. Defaults to 6.

    Returns:
        list: A list of dicts with keys 'chunk', 'page', and 'score'.
    """
    top_k = min(top_k, len(chunks))
    embedded_query = model.encode([query]).astype("float32")
    distances,indices = index.search(embedded_query,top_k)

    retrieved_chunks  = []
    for i,idx in enumerate(indices[0]):
        retrieved_chunks.append(
            {
                "chunk": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(distances[0][i])
            }
        )
    return retrieved_chunks
        
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from vector_store import load_index


    model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index,chunks = load_index()
    retrieved_c = retrieve_chunks("What are the recommendations made in the report?", faiss_index, chunks,model)
    
    print(retrieved_c)
    

    print(len(chunks))