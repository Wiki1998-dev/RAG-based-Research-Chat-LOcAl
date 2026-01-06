import chromadb
from chromadb.config import Settings
from .base import VectorStore

class ChromaVectorStore(VectorStore):

    def __init__(self, path: str, collection_name: str):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, embeddings, documents, metadata):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata
        )

    def query(self, embedding, top_k):
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0]  # ← ids are returned automatically
        }

