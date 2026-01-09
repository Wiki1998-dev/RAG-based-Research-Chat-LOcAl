import ollama
import chromadb
from chromadb.config import Settings
import fitz  # PyMuPDF
import os

# --- 1. Configuration ---
EMBEDDING_MODEL = 'mxbai-embed-large'
LANGUAGE_MODEL = 'llama3'
SOURCE_DIR = "research_papers"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers_db"

# --- 2. Database & Ingestion Setup (Same as before) ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name=COLLECTION_NAME)


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


def process_docs():
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"Directory {SOURCE_DIR} created. Add PDFs there.")
        return

    pdf_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        path = os.path.join(SOURCE_DIR, pdf_file)
        # Check if already processed (simple check based on filename existing in DB)
        # For a more robust check, you'd check specific hashes, but this keeps it simple
        existing = collection.get(where={"source": pdf_file})
        if existing['ids']:
            continue

        print(f"Processing {pdf_file}...")
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        chunks = split_text_into_chunks(text)

        for i, chunk in enumerate(chunks):
            response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
            collection.add(
                ids=[f"{pdf_file}-{i}"],
                embeddings=[response['embeddings'][0]],
                documents=[chunk],
                metadatas=[{"source": pdf_file}]
            )
        print(f"Finished {pdf_file}")


# --- 3. Base Retrieval Helper ---

def query_database(query_text, n_results=5):
    """Encodes a single query string and searches the DB."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=query_text)
    results = collection.query(
        query_embeddings=[response['embeddings'][0]],
        n_results=n_results
    )
    # Flatten list of lists
    return results['documents'][0] if results['documents'] else []


# --- 4. Query Translation Techniques ---

def retrieval_naive(question):
    """Standard RAG: Just embed the user question."""
    print("  -> Using Naive Retrieval...")
    return query_database(question, n_results=5)


def retrieval_multi_query(question):
    """Generates variations of the question and retrieves for all."""
    print("  -> Generating Multi-Query variations...")

    prompt = f"""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}"""

    response = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])

    # Split by newline and clean up
    variations = response['message']['content'].strip().split('\n')
    variations = [v.strip() for v in variations if v.strip()]

    # Add original question to the list
    if question not in variations:
        variations.append(question)

    print(variations)
    print(f"     Generated {len(variations)} variations.")

    unique_docs = set()
    for query in variations:
        docs = query_database(query, n_results=3)  # Get top 3 for each variation
        for doc in docs:
            unique_docs.add(doc)

    return list(unique_docs)


def retrieval_decomposition(question):
    """Breaks complex questions into sub-questions."""
    print("  -> Decomposing question...")

    prompt = f"""You are a helpful assistant that helps retrieves documents.
    Decompose the following complex question into 2-4 simpler sub-questions 
    that, when answered, will help answer the original question.
    Output ONLY the sub-questions separated by newlines.
    Original question: {question}"""

    response = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])
    sub_queries = response['message']['content'].strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]

    print(f"     Decomposed into: {sub_queries}")

    unique_docs = set()
    for q in sub_queries:
        docs = query_database(q, n_results=3)
        for doc in docs:
            unique_docs.add(doc)

    return list(unique_docs)


def retrieval_step_back(question):
    """Generates a higher-level abstract question."""
    print("  -> Generating Step-Back question...")

    prompt = f"""You are an expert at world knowledge. Your task is to step back and paraphrase the user's question to a more generic step-back question, which is easier to answer.
    Output ONLY the step-back question.
    User Question: {question}"""

    response = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])
    step_back_question = response['message']['content'].strip()

    print(f"     Step-back Query: {step_back_question}")

    # Retrieve for original AND step-back
    docs_original = query_database(question, n_results=4)
    docs_stepback = query_database(step_back_question, n_results=3)

    return list(set(docs_original + docs_stepback))


def retrieval_hyde(question):
    """Hypothetical Document Embeddings: Hallucinates an answer, then searches."""
    print("  -> Running HyDE (generating hypothetical answer)...")

    prompt = f"""Please write a short, plausible scientific passage that answers the question. 
    It doesn't have to be factually correct, but it should use the terminology and structure 
    expected in a relevant document.
    Question: {question}"""

    response = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': prompt}])
    hypothetical_answer = response['message']['content']

    print("     Hypothetical Answer generated (first 50 chars):", hypothetical_answer[:50] + "...")

    # Embed the FAKE answer to find REAL documents
    return query_database(hypothetical_answer, n_results=5)


# --- 5. Main Chat Logic ---

def generate_final_answer(query, context_docs):
    formatted_context = "\n\n".join(context_docs)

    system_prompt = """You are a research assistant. Answer the user's question using ONLY the context provided below.
    If the context is insufficient, say you don't know."""

    print("\n--- Generating Response ---")
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Context:\n{formatted_context}\n\nQuestion: {query}"}
        ],
        stream=True
    )

    response = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        response += content
    print("\n" + "=" * 50)


def main():
    process_docs()  # Ensure DB is populated

    strategies = {
        "1": ("Naive RAG", retrieval_naive),
        "2": ("Multi-Query", retrieval_multi_query),
        "3": ("Decomposition", retrieval_decomposition),
        "4": ("Step-Back", retrieval_step_back),
        "5": ("HyDE", retrieval_hyde)
    }

    print("\n--- Research Chatbot with Query Translation ---")

    # User selects strategy ONCE for the session (or you can move this inside the loop)
    print("Select Retrieval Strategy:")
    for key, (name, _) in strategies.items():
        print(f"{key}. {name}")

    choice = input("Enter choice (1-5): ").strip()
    if choice not in strategies:
        print("Invalid choice, defaulting to Naive.")
        choice = "1"

    strategy_name, strategy_func = strategies[choice]
    print(f"Selected Strategy: {strategy_name}")

    while True:
        query = input('\nAsk a question (or "exit", "switch"): ')
        if query.lower() == 'exit':
            break
        if query.lower() == 'switch':
            print("Select Retrieval Strategy:")
            for key, (name, _) in strategies.items():
                print(f"{key}. {name}")
            choice = input("Enter choice (1-5): ").strip()
            if choice in strategies:
                strategy_name, strategy_func = strategies[choice]
                print(f"Switched to: {strategy_name}")
            continue

        # 1. Retrieve Documents using selected strategy
        retrieved_docs = strategy_func(query)

        if not retrieved_docs:
            print("No relevant documents found.")
            continue

        print(f"\nRetrieved {len(retrieved_docs)} unique context chunks.")

        # 2. Generate Answer
        generate_final_answer(query, retrieved_docs)


if __name__ == "__main__":
    main()