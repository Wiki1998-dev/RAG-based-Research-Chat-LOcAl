class Retriever:

    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.embed(query)
        results = self.vectorstore.query(query_embedding, top_k)

        # Return EVERYTHING, just like the old script
        return results
