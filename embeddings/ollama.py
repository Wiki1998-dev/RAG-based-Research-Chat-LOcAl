import ollama
from .base import Embedder

class OllamaEmbedder(Embedder):

    def __init__(self, model: str):
        self.model = model

    def embed(self, text: str):
        response = ollama.embed(model=self.model, input=text)
        return response["embeddings"][0]
