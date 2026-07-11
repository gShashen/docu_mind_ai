from sentence_transformers import SentenceTransformer

def retrieve_chunks(query,index,chunks,model,top_k=6):
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