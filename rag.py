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
COLLECTION_NAME = os.environ.get(
    "COLLECTION_NAME", "research_assist_384_v1"
)

VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 1000

llm = None
vector_store = None


def initialize_components():

    global llm, vector_store

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

    if vector_store is None:

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
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings
        )

    return vector_store


def process_urls(urls):

    yield "Initializing components..."

    initialize_components()

    if not urls:
        raise ValueError("Please provide at least one URL.")

    all_documents = []
    failed_urls = []

    yield "Loading web pages..."

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

    for url in urls:

        url = url.strip()

        if not url:
            continue

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
                failed_urls.append(
                    f"{url}: No meaningful text extracted."
                )
                continue

            all_documents.extend(valid_documents)

            yield (
                f"Successfully extracted "
                f"{len(valid_documents)} document(s): {url}"
            )

        except Exception as e:

            failed_urls.append(f"{url}: {str(e)}")

    if not all_documents:

        error_details = "\n".join(failed_urls)

        raise RuntimeError(
            "No content could be extracted from the supplied URLs.\n"
            + error_details
        )

    yield f"Successfully loaded {len(all_documents)} documents."

    yield "Splitting documents into chunks..."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = text_splitter.split_documents(all_documents)

    docs = [
        doc for doc in docs
        if doc.page_content.strip()
    ]

    if not docs:

        raise RuntimeError(
            "Documents were loaded, but no text chunks were generated."
        )

    yield f"Generated {len(docs)} chunks."

    yield "Adding documents to ChromaDB..."

    uuids = [str(uuid4()) for _ in docs]

    vector_store.add_documents(
        documents=docs,
        ids=uuids
    )

    collection_count = vector_store._collection.count()

    if collection_count == 0:
        raise RuntimeError("ChromaDB collection is empty.")

    yield (
        f"Successfully stored {len(docs)} chunks. "
        f"Total collection size: {collection_count}."
    )

    if failed_urls:
        yield (
            f"Warning: {len(failed_urls)} URL(s) could not be processed. "
            + " | ".join(failed_urls)
        )


def generate_answer(query):

    initialize_components()

    collection_count = vector_store._collection.count()

    if collection_count == 0:

        raise RuntimeError(
            "Vector database is empty. Please process URLs first."
        )

    retrieved_docs = vector_store.similarity_search(
        query,
        k=4
    )

    if not retrieved_docs:

        raise RuntimeError(
            "No relevant documents retrieved."
        )

    print(f"Collection size: {collection_count}")
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
