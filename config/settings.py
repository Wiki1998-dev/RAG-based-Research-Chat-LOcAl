from dataclasses import dataclass

@dataclass(frozen=True)
class AppSettings:
    embedding_model: str = "mxbai-embed-large"
    language_model: str = "llama3"

    source_dir: str = "./research_papers"
    chroma_path: str = "./chroma_db"
    collection_name: str = "research_papers_db"

    chunk_size: int = 1000
    chunk_overlap: int = 200
