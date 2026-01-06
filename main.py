import os

from config.settings import AppSettings
from chunking.fixed import FixedSizeChunker
from embeddings.ollama import OllamaEmbedder
from vectorstores.chroma import ChromaVectorStore
from ingestion.pipeline import IngestionPipeline
from retrieval.retriever import Retriever
from chat.assistant import ResearchAssistant

def main():
    settings = AppSettings()

    chunker = FixedSizeChunker(settings.chunk_size, settings.chunk_overlap)
    embedder = OllamaEmbedder(settings.embedding_model)
    store = ChromaVectorStore(settings.chroma_path, settings.collection_name)

    ingestion = IngestionPipeline(chunker, embedder, store)

    if not os.path.exists(settings.source_dir):
        os.makedirs(settings.source_dir)

    for file in os.listdir(settings.source_dir):
        if file.endswith(".pdf"):
            ingestion.ingest_pdf(os.path.join(settings.source_dir, file))

    retriever = Retriever(embedder, store)
    assistant = ResearchAssistant(retriever, settings.language_model)

    while True:
        q = input("\nAsk (exit to quit): ")
        if q.lower() == "exit":
            break
        assistant.ask(q)

if __name__ == "__main__":
    main()
