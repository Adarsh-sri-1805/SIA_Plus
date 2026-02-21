import faiss
import numpy as np
from db import reviews_collection

class VectorStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def rebuild_index(self):
        data = list(reviews_collection.find({}))

        if not data:
            return

        embeddings = [item["embedding"] for item in data]
        self.texts = [item["text"] for item in data]

        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)

    def search(self, query_emb, top_k=5):
        if self.index.ntotal == 0:
            return []

        query_emb = np.array([query_emb]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)

        return [
            self.texts[idx]
            for idx in indices[0]
            if idx < len(self.texts)
        ]
