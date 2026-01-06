import fitz
import os

class IngestionPipeline:

    def __init__(self, chunker, embedder, vectorstore):
        self.chunker = chunker
        self.embedder = embedder
        self.vectorstore = vectorstore

    def ingest_pdf(self, file_path: str):
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()

        chunks = self.chunker.split(text)

        ids = []
        embeddings = []
        documents = []
        metadata = []

        for i, chunk in enumerate(chunks):
            ids.append(f"{os.path.basename(file_path)}-{i}")
            embeddings.append(self.embedder.embed(chunk))
            documents.append(chunk)
            metadata.append({"source": os.path.basename(file_path)})

        self.vectorstore.add(ids, embeddings, documents, metadata)
