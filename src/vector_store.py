import faiss
import pickle

def save_index(index, chunks):

    faiss.write_index(index, "data/faiss.index")

    with open("data/chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)

def load_index():

    index = faiss.read_index("data/faiss.index")

    with open("data/chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return index, chunks