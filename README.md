***

# Research Paper RAG Chatbot
A completely local, transparent Retrieval-Augmented Generation (RAG) system designed for querying research papers. This project allows you to ingest PDF documents, store their semantic meaning, and chat with them using local LLMs.

**The system is intentionally LangChain-free to maintain full transparency, easy debugging, and granular control over every step of the pipeline.**

## Overview

This tool enables you to:
*   **Ingest** PDF research papers.
*   **Split** text into manageable chunks with overlap.
*   **Embed** chunks into vectors using a local embedding model.
*   **Store** vectors in a persistent local database.
*   **Retrieve** the most relevant context based on user queries.
*   **Generate** answers strictly grounded in the provided documents.

## High-Level Architecture

The system consists of five distinct conceptual layers. Each layer is implemented explicitly and can be modified independently:

1.  **Document Ingestion:** Loading PDFs and extracting raw text.
2.  **Text Chunking:** Splitting text into semantic segments.
3.  **Embedding Generation:** Converting text to vector representations.
4.  **Vector Storage & Retrieval:** Persisting data and finding nearest neighbors.
5.  **LLM Generation:** Synthesizing answers from retrieved context.

## Tech Stack & Models

*   **Language:** Python
*   **LLM Host:** [Ollama](https://ollama.com/)
*   **Embedding Model:** `mxbai-embed-large`
*   **Generation Model:** `llama3`
*   **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local, persistent)
*   **PDF Parsing:** `PyMuPDF` (fitz)

## Project Structure

```text
.
├── research_papers/         # Place your PDF files here
├── chroma_db/               # Persistent Vector Database storage
├── main.py                  # Main application script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## How It Works (Core Components)

### 1. PDF Loading & Text Extraction
**Library:** `PyMuPDF (fitz)`

The system reads PDFs sequentially, page by page. The extracted text is concatenated into a single string. This approach preserves reading order and avoids issues often caused by complex PDF layouts.

### 2. Text Chunking
**Strategy:** Sliding Window
*   **Chunk Size:** ~1000 characters
*   **Overlap:** 200 characters
*   **Splitter:** Splits at whitespace to avoid breaking words.

*Why overlap?* It prevents loss of meaning at the boundaries of chunks and improves retrieval accuracy for queries that span multiple sentences.

### 3. Embedding Generation
**Library:** `ollama.embed`

Each text chunk is converted into a high-dimensional vector. We extract the embedding directly from the `response['embeddings'][0]`, ensuring every chunk has a unique vector representation for semantic search.

### 4. Vector Database (ChromaDB)
**Library:** `chromadb`

Chunks are stored with:
*   **ID:** `filename-chunk_index`
*   **Embedding:** Vector data
*   **Document:** Raw text content
*   **Metadata:** Source filename

**Incremental Ingestion:** Before processing, the system checks existing IDs. Only new chunks are embedded and stored, allowing you to add new PDFs without re-processing the entire dataset.

### 5. Retrieval & Context Construction
**Function:** `retrieve(query, top_n)`

1.  The user query is embedded using the same model (`mxbai-embed-large`).
2.  ChromaDB performs a similarity search to find the `top_n` most similar chunks.
3.  **No filtering or distance thresholds are applied.** All retrieved chunks are passed to the LLM to ensure maximum context visibility.

### 6. Interactive Chat & Generation
**Model:** `llama3`

The retrieved text chunks are concatenated into a single context block. The system prompt explicitly instructs the model to:
1.  Use **only** the provided context.
2.  Admit when the context is insufficient.
3.  Never invent information.

Answers are streamed in real-time to reduce perceived latency.

## Design Philosophy

### Why No LangChain?
Many RAG tutorials rely on LangChain, which often abstracts away the critical mechanics of retrieval and prompt construction. This project is built without it to ensure:

*   **Full Visibility:** You see exactly how text is chunked, embedded, and retrieved.
*   **No Hidden Abstractions:** There is no "magic" happening behind the scenes.
*   **Easier Debugging:** If the retrieval is poor, you know exactly where to look (chunk size, overlap, or embedding model).
*   **Performance:** A lighter weight application with fewer dependencies.

### Behavior Guarantees
*   The system always retrieves `top_n` chunks.
*   It explicitly prints retrieved context to the console for transparency.
*   It does not suppress answers based on arbitrary confidence scores.

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install chromadb pymupdf ollama
    ```

2.  **Pull Ollama Models:**
    Ensure Ollama is installed and running, then pull the required models:
    ```bash
    ollama pull mxbai-embed-large
    ollama pull llama3
    ```

3.  **Add Documents:**
    Place your PDF research papers in the `research_papers/` directory.

4.  **Run the Application:**
    ```bash
    python main.py
    ```
    *The first run will take longer as it embeds the PDFs. Subsequent runs will be instant due to the persistent ChromaDB storage.*

5.  **Chat:**
    Enter your query when prompted. Type `exit` to quit.
