import hashlib
import io
import os
from typing import List, Union
import requests
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from pydantic import Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
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
_pinecone_index = None
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
        all_embeddings: List[List[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = pc.inference.embed(
                model=PINECONE_EMBED_MODEL,
                inputs=batch,
                parameters={
                    "input_type": input_type,
                    "truncate": "END",
                    "dimension": EMBED_DIMENSION,
                },
            )
            all_embeddings.extend(item["values"] for item in res.data)
        return all_embeddings

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


def get_pinecone_index():
    """Returns a globally cached Pinecone Index client (serverless-safe, no multiprocessing)."""
    global _pinecone_index
    if _pinecone_index is None:
        ensure_index_exists()
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


class ServerlessPineconeRetriever(BaseRetriever):
    """Lightweight synchronous Pinecone retriever that avoids multiprocessing.Pool in serverless."""
    namespace: str = Field(default="default")
    k: int = Field(default=5)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        index = get_pinecone_index()
        query_vector = get_embeddings().embed_query(query)
        response = index.query(
            vector=query_vector,
            top_k=self.k,
            namespace=self.namespace or "default",
            include_metadata=True,
        )
        docs: List[Document] = []
        matches = getattr(response, "matches", None) or response.get("matches", [])
        for match in matches:
            metadata = dict(getattr(match, "metadata", None) or match.get("metadata", {}) or {})
            text = metadata.pop("text", "")
            if text:
                docs.append(Document(page_content=text, metadata=metadata))
        return docs


def get_retriever(namespace: str = "default", k: int = 5):
    """Returns a namespace-scoped ServerlessPineconeRetriever instance."""
    return ServerlessPineconeRetriever(namespace=namespace or "default", k=k)


def add_documents_to_vectorstore(
    page_documents: List[Document],
    filename: str = "document.pdf",
    namespace: str = "default",
) -> int:
    """
    Splits page-level Documents into chunks (preserving source & page metadata)
    and synchronously upserts them into a session-isolated Pinecone namespace.
    """
    if not page_documents:
        raise ValueError("Document content cannot be empty.")

    target_namespace = namespace or "default"
    chunks = _text_splitter.split_documents(page_documents)
    if not chunks:
        raise ValueError("No text chunks could be generated from the PDF.")

    texts = [chunk.page_content for chunk in chunks]
    vectors_values = get_embeddings().embed_documents(texts)

    vectors_to_upsert = []
    for idx, (chunk, values) in enumerate(zip(chunks, vectors_values)):
        source = chunk.metadata.get("source", filename)
        page = chunk.metadata.get("page", 1)
        raw_key = f"{target_namespace}:{source}:{page}:{idx}:{chunk.page_content}"
        vec_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        vectors_to_upsert.append(
            {
                "id": vec_id,
                "values": values,
                "metadata": {
                    "text": chunk.page_content,
                    "source": source,
                    "page": page,
                    "chunk_index": idx,
                },
            }
        )

    print(
        f"Upserting {len(vectors_to_upsert)} chunks into Pinecone index '{INDEX_NAME}' "
        f"(namespace='{target_namespace}')..."
    )

    index = get_pinecone_index()
    upsert_batch_size = 100
    for i in range(0, len(vectors_to_upsert), upsert_batch_size):
        index.upsert(
            vectors=vectors_to_upsert[i : i + upsert_batch_size],
            namespace=target_namespace,
        )

    print(
        f"Successfully indexed {len(vectors_to_upsert)} idempotent chunks in namespace '{target_namespace}'."
    )
    return len(vectors_to_upsert)


def process_pdf_and_store(
    pdf_source: Union[bytes, io.BytesIO, str],
    filename: str = "document.pdf",
    namespace: str = "default",
) -> int:
    """
    Extracts text per page in memory with source/page metadata and stores
    idempotent chunks in the session's Pinecone namespace.
    """
    if isinstance(pdf_source, bytes):
        stream = io.BytesIO(pdf_source)
        reader = PdfReader(stream)
    elif isinstance(pdf_source, io.BytesIO):
        reader = PdfReader(pdf_source)
    else:
        reader = PdfReader(pdf_source)

    page_documents: List[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text()
        if extracted and extracted.strip():
            page_documents.append(
                Document(
                    page_content=extracted.strip(),
                    metadata={"source": filename, "page": page_num},
                )
            )

    if not page_documents:
        raise ValueError("Could not extract any readable text from the uploaded PDF.")

    return add_documents_to_vectorstore(
        page_documents=page_documents,
        filename=filename,
        namespace=namespace,
    )