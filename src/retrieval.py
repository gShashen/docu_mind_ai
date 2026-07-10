import faiss

def retrieve_chunks(query,index,chunks,model,top_k=3):
    top_k = min(top_k, len(chunks))
    embedded_query = model.encode([query])
    distance,indices = index.search(embedded_query,top_k)

    retrieved_chunks  = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])


    return retrieved_chunks
        
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from vector_store import load_index


    model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index,chunks = load_index()
    retrieved_c = retrieve_chunks("What is AI?", faiss_index, chunks,model)
    
    print(retrieved_c)

    print(len(chunks))