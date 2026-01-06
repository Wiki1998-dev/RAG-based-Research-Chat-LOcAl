import ollama

class ResearchAssistant:

    def __init__(self, retriever, model: str):
        self.retriever = retriever
        self.model = model

    def ask(self, question: str):
        retrieval = self.retriever.retrieve(question)

        documents = retrieval["documents"]
        metadatas = retrieval["metadatas"]
        distances = retrieval["distances"]
        ids = retrieval["ids"]

        print("\nRetrieved context:")
        for doc in documents:
            print(f" - {doc[:100].replace(chr(10), ' ')}...")

        formatted_knowledge = "\n".join(f"- {doc}" for doc in documents)

        instruction_prompt = f"""You are a helpful research assistant.
Use only the following context from research papers to answer the question.
Do not make up any new information. If the context does not contain the answer, say
"I don't have enough information from the documents to answer that."

Context:
{formatted_knowledge}
"""

        print("\nChatbot response:")
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": question}
            ],
            stream=True
        )

        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n" + "=" * 50)
