import time
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb

# Start timing the overall execution
start_time = time.time()

# Step 1: Define LLM model and configure ChromaDB
print("Step 1: Configuring LLM and ChromaDB...")

# Define the LLM model to be used
llm_model = "llama3.2"

# Initialize the ChromaDB client with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
print(f"ChromaDB initialized at {os.path.join(os.getcwd(), 'chroma_db')}.")

# Step 2: Define the custom embedding function for ChromaDB
print("Step 2: Defining custom embedding function...")
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)
print(f"Ollama Embeddings initialized with model {llm_model}.")

# Step 3: Create or retrieve the ChromaDB collection
collection_name = "rag_collection_demo_1"
print(f"Step 3: Creating or retrieving ChromaDB collection '{collection_name}'...")

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)
print(f"Collection '{collection_name}' ready for use.")

# Step 4: Add documents to the collection
print("Step 4: Adding sample documents to the collection...")

def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    collection.add(
        documents=documents,
        ids=ids
    )

# Sample documents to add to the collection
documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "ChromaDB is a vector database designed for AI applications."
]
doc_ids = ["doc1", "doc2", "doc3"]

# Add the sample documents to ChromaDB
add_documents_to_collection(documents, doc_ids)
print(f"Documents added to collection: {doc_ids}")

# Step 5: Define function to query ChromaDB
print("Step 5: Defining query function for ChromaDB...")

def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Step 6: Define function to query Ollama LLM
print("Step 6: Defining function to query Ollama LLM...")

def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# Step 7: Define RAG pipeline combining ChromaDB and Ollama
print("Step 7: Defining Retrieval-Augmented Generation (RAG) pipeline...")

def rag_pipeline(query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 7.1: Retrieve relevant documents from ChromaDB
    print("Step 7.1: Retrieving relevant documents from ChromaDB...")
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Step 7.2: Send the query along with the context to Ollama
    print("Step 7.2: Sending augmented prompt to Ollama...")
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response

# Example usage: Test the RAG pipeline
print("Step 8: Testing the RAG pipeline with a sample query...")
query = "What is artificial intelligence?"  # Change the query as needed
response = rag_pipeline(query)

# Print the final response
print("######## Response from LLM ########\n", response)

# End the overall execution time and print it
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution completed in {execution_time:.2f} seconds.")
