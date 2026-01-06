from abc import ABC, abstractmethod
from typing import List

class VectorStore(ABC):

    @abstractmethod
    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadata: List[dict]):
        pass

    @abstractmethod
    def query(self, embedding: List[float], top_k: int):
        pass
