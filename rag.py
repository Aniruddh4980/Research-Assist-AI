import hashlib
import os
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 1000

# The LLM can be shared because it does not store user documents.
llm = None


def initialize_components(collection_name):
    """
    Initialize the LLM and a session-specific ChromaDB collection.
    """

    global llm

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")

    if not GROQ_MODEL:
        raise ValueError("GROQ_MODEL is missing.")

    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL is missing.")

    if llm is None:
        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=MAX_TOKENS
        )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings
    )

    return vector_store


def get_http_status_code(error):
    """
    Extract an HTTP status code from a loading exception.
    """

    response = getattr(error, "response", None)

    if response is not None:
        return getattr(response, "status_code", None)

    # Some loader exceptions include the HTTP status in their message.
    error_message = str(error).lower()

    if "403" in error_message or "forbidden" in error_message:
        return 403

    if "401" in error_message or "unauthorized" in error_message:
        return 401

    return None


def process_urls(urls, vector_store):
    """
    Load, split, and index URLs into the supplied session-specific
    ChromaDB collection.

    Reprocessing a URL replaces its previous chunks instead of
    creating duplicate records.
    """

    if not urls:
        raise ValueError("Please provide at least one URL.")

    url_results = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        )
    }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Avoid processing the same URL multiple times in one request.
    unique_urls = list(dict.fromkeys(
        url.strip() for url in urls if url.strip()
    ))

    for url in unique_urls:

        try:
            loader = WebBaseLoader(
                web_paths=[url],
                header_template=headers,
                requests_kwargs={"timeout": 25},
                raise_for_status=True
            )

            documents = loader.load()

            valid_documents = [
                doc for doc in documents
                if doc.page_content
                and len(doc.page_content.strip()) >= 100
            ]

            if not valid_documents:
                url_results.append({
                    "url": url,
                    "status": "failed",
                    "message": "No meaningful text could be extracted."
                })
                continue

            chunks = text_splitter.split_documents(valid_documents)

            chunks = [
                chunk for chunk in chunks
                if chunk.page_content.strip()
            ]

            if not chunks:
                url_results.append({
                    "url": url,
                    "status": "failed",
                    "message": "No text chunks could be generated."
                })
                continue

            # Generate deterministic IDs using URL, chunk index,
            # and content. Identical chunks receive identical IDs
            # when the same URL is processed again.
            chunk_ids = []

            for index, chunk in enumerate(chunks):

                chunk.metadata["source"] = url

                unique_string = (
                    f"{url}|{index}|{chunk.page_content}"
                )

                chunk_id = hashlib.sha256(
                    unique_string.encode("utf-8")
                ).hexdigest()

                chunk_ids.append(chunk_id)

            # Retrieve previously indexed chunk IDs for this URL.
            existing_records = vector_store.get(
                where={"source": url},
                include=["metadatas"]
            )

            existing_ids = set(existing_records["ids"])
            new_ids = set(chunk_ids)

            # Upsert new chunks. Existing deterministic IDs are
            # updated instead of creating duplicate records.
            vector_store.add_documents(
                documents=chunks,
                ids=chunk_ids
            )

            # Remove outdated chunks if the page content changed
            # or the page now contains fewer chunks.
            obsolete_ids = list(existing_ids - new_ids)

            if obsolete_ids:
                vector_store.delete(ids=obsolete_ids)

            url_results.append({
                "url": url,
                "status": "indexed",
                "message": "Successfully indexed."
            })

        except Exception as error:

            status_code = get_http_status_code(error)

            if status_code in (401, 403):
                url_results.append({
                    "url": url,
                    "status": "blocked",
                    "message": f"Website denied access (HTTP {status_code})."
                })

            else:
                url_results.append({
                    "url": url,
                    "status": "failed",
                    "message": str(error)
                })

    return {
        "url_results": url_results
    }


def generate_answer(query, vector_store):
    """
    Generate an answer using only the current session's collection.
    """

    if llm is None:
        initialize_components(vector_store._collection.name)

    collection_count = vector_store._collection.count()

    if collection_count == 0:
        raise RuntimeError(
            "Your vector database is empty. Please process URLs first."
        )

    retrieved_docs = vector_store.similarity_search(
        query,
        k=4
    )

    if not retrieved_docs:
        raise RuntimeError(
            "No relevant documents retrieved."
        )

    print(f"Retrieved documents: {len(retrieved_docs)}")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nRetrieved document {i + 1}")
        print("Source:", doc.metadata.get("source"))
        print("Content:", doc.page_content[:300])

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4}
        )
    )

    result = chain.invoke(
        {"question": query},
        return_only_outputs=True
    )

    return (
        result.get("answer", "No answer generated."),
        result.get("sources", "")
    )
