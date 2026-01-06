***

# RAG System for Research Papers (Ollama + ChromaDB)

This project implements a modular Retrieval-Augmented Generation (RAG) system for querying research papers stored as PDFs.

Unlike a monolithic script, this system is structured using clear responsibilities and software design patterns. It runs fully locally using **Ollama** for embeddings and language models, **ChromaDB** for persistent vector storage, and **PyMuPDF** for PDF parsing.

## Overview

The system is designed to be easier to understand, debug, extend, and maintain than typical single-file RAG scripts.

**High-Level Flow:**
1.  **Ingest:** PDFs are loaded, text is extracted, and split into chunks.
2.  **Embed:** Chunks are converted into vectors and stored in a persistent database.
3.  **Retrieve:** User queries are embedded to find relevant text chunks.
4.  **Generate:** The language model answers the query using *only* the retrieved context.

## Core Design Principles

*   **Single Responsibility:** Each class and module does exactly one thing.
*   **Explicit Data Flow:** There is no hidden logic or "magic" state management.
*   **No Framework Magic:** The system is built without heavy abstractions like LangChain to ensure full transparency.
*   **Observability:** Retrieved chunks are printed to the console so you can verify exactly what the LLM is reading.

## Architecture

### System Architecture

```text
┌──────────────────────┐
│      User Input      │
│  (Natural Language)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   ResearchAssistant  │
│  (chat/assistant.py) │
│                      │
│ - prints retrieved   │
│   chunks             │
│ - builds prompt      │
│ - calls LLM          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Retriever       │
│ (retrieval/retriever)│
│                      │
│ - embeds query       │
│ - requests top-K     │
│   matches            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Vector Store     │
│ (vectorstores/chroma)│
│                      │
│ - similarity search  │
│ - returns chunks     │
│   + metadata         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Embeddings      │
│ (ollama_embedder.py) │
│                      │
│ - text → vectors     │
│ - shared for ingest  │
│   & query            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     ChromaDB         │
│  (Persistent Store)  │
│                      │
│ - embeddings         │
│ - chunk text         │
│ - metadata           │
└──────────────────────┘
```

### Ingestion Pipeline

```text
┌──────────────┐
│   PDF File   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│   PDF Loader     │
│ (pdf_loader.py)  │
│                  │
│ - extract text   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│    Chunker       │
│ (chunker.py)     │
│                  │
│ - overlapping    │
│   chunks         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│    Embedder      │
│ (Ollama)         │
│                  │
│ - chunk → vector │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Vector Store   │
│ (ChromaDB)       │
│                  │
│ - persist data   │
└──────────────────┘
```

## Project Structure

Each folder corresponds to one specific responsibility in the RAG pipeline.

```text
RAG/
│
├── main.py                  # Entry point: wires components together
├── config.py                # Configuration: constants, paths, model names
│
├── ingestion/
│   ├── pdf_loader.py        # Handles PDF parsing via PyMuPDF
│   └── chunker.py           # Logic for text splitting and overlap
│
├── embeddings/
│   └── ollama_embedder.py   # Wrapper for Ollama embedding models
│
├── vectorstores/
│   └── chroma.py            # Wrapper for ChromaDB persistence
│
├── retrieval/
│   └── retriever.py         # Logic for querying the vector store
│
└── chat/
    └── assistant.py         # Handles user interaction and LLM prompting
```

## Component Details

### Configuration (`config.py`)
Centralizes all constants including model names, directory paths, chunk sizes, and collection names. This prevents "magic values" from being scattered across the codebase.

### PDF Loading (`ingestion/pdf_loader.py`)
*   **Responsibility:** Load PDFs and extract raw text.
*   **Method:** Uses `PyMuPDF` to read pages sequentially and returns a single string per document.
*   **Reasoning:** PDF parsing is fragile; keeping it isolated avoids cross-contamination of logic.

### Chunking (`ingestion/chunker.py`)
*   **Responsibility:** Split raw text into overlapping chunks.
*   **Strategy:** Fixed chunk size with overlapping windows, splitting at whitespace to preserve words.
*   **Reasoning:** Overlap prevents context loss at chunk boundaries, improving retrieval accuracy.

### Embedding Layer (`embeddings/ollama_embedder.py`)
*   **Responsibility:** Convert text into vectors using Ollama.
*   **Abstraction:** Provides a standard `embed(text)` method. This allows you to swap the embedding provider (e.g., to OpenAI or HuggingFace) without breaking the rest of the app.

### Vector Store (`vectorstores/chroma.py`)
*   **Responsibility:** Persist embeddings and perform similarity searches.
*   **Storage:** Uses ChromaDB to store documents, embeddings, and metadata.
*   **Incremental Loading:** Checks for existing IDs before insertion to prevent duplicates.

### Retriever (`retrieval/retriever.py`)
*   **Responsibility:** Perform the actual semantic search.
*   **Behavior:** Embeds the user query, queries the vector store, and returns the top-K results.
*   **Note:** No filtering or thresholding is applied at this stage to ensure total observability of what the database considers "relevant."

### Chat Assistant (`chat/assistant.py`)
*   **Responsibility:** Orchestrate the interaction.
*   **Process:**
    1.  Receives user input.
    2.  Calls the retriever.
    3.  Prints the retrieved chunks to the console (for debugging/preview).
    4.  Constructs a prompt that strictly enforces "answer using only the provided context."
    5.  Streams the LLM response.

## Usage

1.  **Prerequisites:**
    *   Python 3.8+
    *   Ollama installed and running

2.  **Install Dependencies:**
    ```bash
    pip install chromadb pymupdf ollama
    ```

3.  **Pull Models:**
    ```bash
    ollama pull mxbai-embed-large
    ollama pull llama3
    ```

4.  **Add Documents:**
    Place your PDF research papers in the configured `research_papers/` directory.

5.  **Run:**
    ```bash
    python main.py
    ```

## Extensibility

Because of the modular structure, this system is easily extensible. You can add the following features without rewriting the core logic:

*   **Strict RAG:** Add a distance threshold filter in the `Retriever` class.
*   **Re-ranking:** Insert a re-ranking model (like Cross-Encoder) in the `Retriever` before returning results.
*   **Multi-Modal:** Swap `pdf_loader.py` for a loader that handles images or markdown.
*   **Citation:** Modify `assistant.py` to parse metadata and provide citations in the final answer.
