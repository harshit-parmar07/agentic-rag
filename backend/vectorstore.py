import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings # Changed to HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Import API keys and configurations from config
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = None
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Define Pinecone index name from config
INDEX_NAME = PINECONE_INDEX_NAME
_INDEX_CHECKED = False

# --- Function to ensure Pinecone index exists ---
def ensure_index_exists():
    global _INDEX_CHECKED
    if _INDEX_CHECKED:
        return
        
    if not pc:
        raise ValueError("Pinecone client is not initialized. Please configure PINECONE_API_KEY in config.env.")
    
    # Check if index exists, and create if it doesn't
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Changed dimension for 'sentence-transformers/all-MiniLM-L6-v2'
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust cloud/region as per your Pinecone setup
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")
        
    _INDEX_CHECKED = True

# --- Retriever (Existing function) ---
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""
    ensure_index_exists()
    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()

# --- Function to add documents to the vector store ---
def add_document_to_vectorstore(text_content: str):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    """
    if not pc:
        raise ValueError("Pinecone client is not initialized. Please configure PINECONE_API_KEY in config.env.")
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    
    # Create Langchain Document objects from the raw text
    documents = text_splitter.create_documents([text_content])
    
    print(f"Splitting document into {len(documents)} chunks for indexing...")
    
    # Ensure index exists before adding documents
    ensure_index_exists()
    
    # Get the vectorstore instance (not the retriever) to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Add documents to the vector store
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")

# --- Function to parse and store PDFs ---
def process_pdf_and_store(file_path: str) -> int:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if documents:
        full_text_content = "\n\n".join([doc.page_content for doc in documents])
        add_document_to_vectorstore(full_text_content)
        return len(documents)
    return 0