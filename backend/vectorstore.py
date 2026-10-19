import io
import os
from typing import List, Union
import requests
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_EMBED_MODEL,
    HF_API_KEY,
    EMBED_MODEL,
    EMBED_DIMENSION,
)

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# --- Global Execution Context Singletons (reused across warm serverless invocations) ---
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
INDEX_NAME = PINECONE_INDEX_NAME
_INDEX_CHECKED = False
_embeddings: Embeddings | None = None
_vectorstore: PineconeVectorStore | None = None
_retriever = None
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


class ServerlessEmbeddings(Embeddings):
    """
    Zero-weight serverless embeddings client (0 MB local model weights).
    Uses HuggingFace Inference API (if HF_API_KEY is set) for BAAI/bge-small-en-v1.5 (384d),
    or falls back to Pinecone Inference API (llama-text-embed-v2 at 384d) using PINECONE_API_KEY.
    """

    def _embed_via_hf(self, texts: List[str]) -> List[List[float]]:
        url = f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}/pipeline/feature-extraction"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data[0], list) and isinstance(data[0][0], list):
            return [item[0] for item in data]
        return data

    def _embed_via_pinecone(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not pc:
            raise ValueError("Pinecone client is not initialized. Configure PINECONE_API_KEY.")
        res = pc.inference.embed(
            model=PINECONE_EMBED_MODEL,
            inputs=texts,
            parameters={
                "input_type": input_type,
                "truncate": "END",
                "dimension": EMBED_DIMENSION,
            },
        )
        return [item["values"] for item in res.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if HF_API_KEY:
            return self._embed_via_hf(texts)
        return self._embed_via_pinecone(texts, input_type="passage")

    def embed_query(self, text: str) -> List[float]:
        if HF_API_KEY:
            return self._embed_via_hf([text])[0]
        return self._embed_via_pinecone([text], input_type="query")[0]


def get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = ServerlessEmbeddings()
    return _embeddings


def ensure_index_exists():
    global _INDEX_CHECKED
    if _INDEX_CHECKED:
        return

    if not pc:
        raise ValueError("Pinecone client is not initialized. Please configure PINECONE_API_KEY.")

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")

    _INDEX_CHECKED = True


def get_vectorstore() -> PineconeVectorStore:
    """Returns a globally cached PineconeVectorStore instance."""
    global _vectorstore
    if _vectorstore is None:
        ensure_index_exists()
        _vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=get_embeddings())
    return _vectorstore


def get_retriever():
    """Returns a globally cached Pinecone retriever instance for warm invocation reuse."""
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever()
    return _retriever


def add_document_to_vectorstore(text_content: str) -> int:
    """Splits text into chunks and upserts them into the cached Pinecone vectorstore."""
    if not text_content or not text_content.strip():
        raise ValueError("Document content cannot be empty.")

    documents = _text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} chunks for indexing...")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")
    return len(documents)


def process_pdf_and_store(pdf_source: Union[bytes, io.BytesIO, str]) -> int:
    """
    Extracts text from a PDF in memory (bytes/BytesIO) or from a /tmp file path
    without requiring persistent local disk storage.
    """
    if isinstance(pdf_source, bytes):
        stream = io.BytesIO(pdf_source)
        reader = PdfReader(stream)
    elif isinstance(pdf_source, io.BytesIO):
        reader = PdfReader(pdf_source)
    else:
        reader = PdfReader(pdf_source)

    pages_text = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted and extracted.strip():
            pages_text.append(extracted)

    if not pages_text:
        raise ValueError("Could not extract any readable text from the uploaded PDF.")

    full_text_content = "\n\n".join(pages_text)
    add_document_to_vectorstore(full_text_content)
    return len(pages_text)