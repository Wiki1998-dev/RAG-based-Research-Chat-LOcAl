import ollama
import chromadb
from chromadb.config import Settings
import fitz  # PyMuPDF
import os
import string
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
EMBEDDING_MODEL = 'mxbai-embed-large'
LANGUAGE_MODEL = 'llama3'

SOURCE_DIR = "research_papers"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers_db"

# Global Cache to hold models in memory
GLOBALS = {
    "bm25": None,
    "bm25_corpus": [],
    "bm25_ids": [],
    "cross_encoder": None
}


# ==========================================
# --- 2. INGESTION & CHUNKING ---
# ==========================================

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    if len(text) <= chunk_size: return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        chunk = text[start:end]
        last_space = -1
        for i in range(len(chunk) - 1, -1, -1):
            if chunk[i].isspace():
                last_space = i
                break
        actual_end = start + last_space if last_space != -1 else end
        chunks.append(text[start:actual_end])
        start += (actual_end - start) - chunk_overlap
        if start < 0: start = actual_end
    return chunks


def process_and_embed_pdf(file_path, collection):
    print(f"Processing {file_path}...")
    try:
        doc = fitz.open(file_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()

        chunks = split_text_into_chunks(full_text)

        # Simple deduplication check
        chunk_ids = [f"{os.path.basename(file_path)}-{i}" for i in range(len(chunks))]
        existing = collection.get(ids=chunk_ids)
        if len(existing['ids']) == len(chunks):
            print(f" - All chunks for {os.path.basename(file_path)} already exist.")
            return

        print(f" - Embedding {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
            embedding = response['embeddings'][0]
            collection.add(
                ids=[chunk_ids[i]],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": os.path.basename(file_path)}]
            )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def run_ingestion(collection):
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"Created directory {SOURCE_DIR}. Please add PDFs there and restart.")
        return False

    pdf_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {SOURCE_DIR}.")
        return False

    for pdf_file in pdf_files:
        process_and_embed_pdf(os.path.join(SOURCE_DIR, pdf_file), collection)
    return True


# ==========================================
# --- 3. ADVANCED SEARCH COMPONENTS ---
# ==========================================

def initialize_search_components(collection):
    """Fetches data from Chroma and builds in-memory BM25 index."""
    print("\n--- Initializing Search Components ---")
    data = collection.get()
    GLOBALS["bm25_corpus"] = data['documents']
    GLOBALS["bm25_ids"] = data['ids']

    if GLOBALS["bm25_corpus"]:
        print(f"Building BM25 index for {len(GLOBALS['bm25_corpus'])} chunks...")
        tokenized_corpus = [
            doc.lower().translate(str.maketrans('', '', string.punctuation)).split()
            for doc in GLOBALS["bm25_corpus"]
        ]
        GLOBALS["bm25"] = BM25Okapi(tokenized_corpus)
    else:
        print("Warning: Database empty. BM25 skipped.")


def search_vector(query, collection, top_k=20):
    """Semantic Vector Search"""
    if not query.strip():
        return []

    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)

        # Safety Check: Ensure we actually got an embedding back
        if not response.get('embeddings') or len(response['embeddings']) == 0:
            print(f"Warning: No embedding generated for query: '{query}'")
            return []

        results = collection.query(query_embeddings=[response['embeddings'][0]], n_results=top_k)

        if not results['documents']:
            return []

        # Flatten structure
        hits = []
        for i in range(len(results['ids'][0])):
            hits.append({'id': results['ids'][0][i], 'content': results['documents'][0][i]})
        return hits

    except Exception as e:
        print(f"Error in vector search for query '{query}': {e}")
        return []

def search_keyword(query, top_k=20):
    """BM25 Keyword Search"""
    if not GLOBALS["bm25"]: return []
    tokenized_query = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = GLOBALS["bm25"].get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [{'id': GLOBALS["bm25_ids"][i], 'content': GLOBALS["bm25_corpus"][i]} for i in top_n]


def reciprocal_rank_fusion(vector_res, keyword_res, k=60):
    """Combines results using RRF algorithm"""
    scores = {}
    content_map = {}

    # Process Vector
    for rank, doc in enumerate(vector_res):
        if doc['id'] not in scores: scores[doc['id']] = 0
        scores[doc['id']] += 1 / (k + rank)
        content_map[doc['id']] = doc['content']

    # Process Keyword
    for rank, doc in enumerate(keyword_res):
        if doc['id'] not in scores: scores[doc['id']] = 0
        scores[doc['id']] += 1 / (k + rank)
        content_map[doc['id']] = doc['content']

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [{'content': content_map[did], 'id': did} for did in sorted_ids]


# ==========================================
# --- 4. QUERY TRANSLATION & RERANKING ---
# ==========================================

def generate_sub_queries(query, mode):
    """Generates modified queries based on strategy"""
    if mode == "naive":
        return [query]

    print(f"   -> Running Query Translation: {mode}...")

    queries = [query]  # Always include original

    if mode == "hyde":
        prompt = f"Write a hypothetical academic passage answering this: {query}"
        resp = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])
        # We search using ONLY the fake answer for HyDE
        return [resp['message']['content']]

    elif mode == "multi_query":
        prompt = f"Generate 3 diverse search queries for: {query}. Output only lines of text."
        resp = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])
        generated = resp['message']['content'].strip().split('\n')
        # Clean: remove empty lines and strip whitespace
        queries.extend([q.strip() for q in generated if q.strip()])

    elif mode == "step_back":
        prompt = f"What is the high-level concept or definition behind this question: {query}? Output only the concept."
        resp = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])
        queries.append(resp['message']['content'].strip())

    # Final cleanup to remove duplicates and empty strings
    return list(set([q for q in queries if q]))

def rerank_results(query, docs, method="none", top_n=5):
    """Re-orders results based on relevance"""
    if not docs: return []

    if method == "none":
        return docs[:top_n]

    print(f"   -> Reranking {len(docs)} docs using {method}...")

    if method == "cross_encoder":
        if GLOBALS["cross_encoder"] is None:
            print("      (Loading Cross-Encoder model - this happens once...)")
            GLOBALS["cross_encoder"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [(query, doc['content']) for doc in docs]
        scores = GLOBALS["cross_encoder"].predict(pairs)
        for i, doc in enumerate(docs): doc['score'] = scores[i]
        return sorted(docs, key=lambda x: x['score'], reverse=True)[:top_n]

    if method == "rankgpt":
        # Ask LLM to rank
        passages = "\n".join([f"[{i}] {d['content'][:150]}..." for i, d in enumerate(docs[:10])])
        prompt = f"Rank these passages by relevance to '{query}'. Return ONLY IDs like [0], [1].\n{passages}"
        resp = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])

        # Parse output for numbers
        indices = [int(s) for s in re.findall(r'\d+', resp['message']['content'])]
        reranked = [docs[i] for i in indices if i < len(docs)]
        # Fallback if parsing fails
        return reranked[:top_n] if reranked else docs[:top_n]

    return docs[:top_n]


# ==========================================
# --- 5. PIPELINE & MAIN LOOP ---
# ==========================================

def advanced_retrieve(query, collection, translation_mode, search_mode, rerank_mode):
    # 1. Translate Query
    queries = generate_sub_queries(query, translation_mode)

    all_results = []

    # 2. Search Loop (for each query variation)
    for q in queries:
        if search_mode == "vector":
            all_results.extend(search_vector(q, collection))
        elif search_mode == "hybrid":
            v = search_vector(q, collection)
            k = search_keyword(q)
            all_results.extend(reciprocal_rank_fusion(v, k))

    # Deduplicate results by ID
    unique_docs = {d['id']: d for d in all_results}.values()
    candidates = list(unique_docs)

    if not candidates: return []

    # 3. Rerank
    final_docs = rerank_results(query, candidates, method=rerank_mode, top_n=5)

    return [d['content'] for d in final_docs]


def main():
    # Setup Client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Ingest Data
    has_data = run_ingestion(collection)
    if not has_data: return

    # Init Advanced Components
    initialize_search_components(collection)

    print("\n" + "=" * 50)
    print("--- ADVANCED RAG CONFIGURATION ---")
    print("=" * 50)

    # Menu Selection
    print("\n[1] Query Translation Strategy:")
    print("    1. Naive (None)")
    print("    2. Multi-Query (Expansion)")
    print("    3. HyDE (Hypothetical Answer)")
    print("    4. Step-Back (Abstraction)")
    t_input = input("Select (1-4) [Default 1]: ").strip() or "1"
    t_map = {"1": "naive", "2": "multi_query", "3": "hyde", "4": "step_back"}

    print("\n[2] Search Mode:")
    print("    1. Vector Only (Fast)")
    print("    2. Hybrid (Vector + Keyword + RRF)")
    s_input = input("Select (1-2) [Default 1]: ").strip() or "1"
    s_map = {"1": "vector", "2": "hybrid"}

    print("\n[3] Reranking Mode:")
    print("    1. None")
    print("    2. Cross-Encoder (High Accuracy - slower)")
    print("    3. RankGPT (LLM Based - slowest)")
    r_input = input("Select (1-3) [Default 1]: ").strip() or "1"
    r_map = {"1": "none", "2": "cross_encoder", "3": "rankgpt"}

    config = {
        "translation": t_map.get(t_input, "naive"),
        "search": s_map.get(s_input, "vector"),
        "rerank": r_map.get(r_input, "none")
    }

    print(
        f"\nActive Config: {config['translation'].upper()} -> {config['search'].upper()} -> {config['rerank'].upper()}")

    # Chat Loop
    while True:
        input_query = input('\nAsk a question (or type "exit"): ')
        if input_query.lower() == 'exit': break

        print("\n--- Retrieving Context ---")
        retrieved_knowledge = advanced_retrieve(
            input_query,
            collection,
            config['translation'],
            config['search'],
            config['rerank']
        )

        if not retrieved_knowledge:
            print("No relevant information found.")
            continue

        print(f"Found {len(retrieved_knowledge)} relevant chunks.")

        # LLM Generation
        formatted_knowledge = "\n\n".join([f"--- Chunk ---\n{chunk}" for chunk in retrieved_knowledge])
        instruction_prompt = f"""You are a helpful research assistant.
        Use only the following context to answer the question. If you don't know, say so.

        Context:
        {formatted_knowledge}
        """

        print('\n--- Chatbot Response ---')
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()