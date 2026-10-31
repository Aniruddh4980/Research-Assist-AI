import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
GROQ_MODEL= os.environ.get("GROQ_MODEL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
VECTORSTORE_DIR = Path(__file__).parent / 'resources'/'vectorstore'
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=1000

llm=None
vector_store=None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.5, max_tokens=MAX_TOKENS)

    if vector_store is None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code':True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=ef
            )

def process_urls(urls):
    '''
    This function scraps data from a url and stores it in vector db
    :param urls: input urls
    :return:
    '''
    yield ('Initializing components...✅')
    initialize_components()

    yield ('Loading data...✅')
    loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False)
    data = loader.load()

    if not data:
        raise ValueError("No documents were extracted from the supplied URLs.")

    yield (f"Loaded {len(data)} documents.✅")

    yield ('Splitting text...✅')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '.', ' ']
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        raise ValueError("No text chunks were generated.")

    yield (f"Generated {len(docs)} chunks.✅")

    yield ('Adding docs to vector db...✅')
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    yield ("Documents successfully stored.✅")

def generate_answer(query):
    if not vector_store:
        raise RuntimeError('Vector DB is not Initialized')
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({'question': query}, return_only_outputs=True)
    source = result.get('sources', "")

    return result['answer'], source



if __name__=='__main__':
    urls = [
        "https://edition.cnn.com/2026/09/17/economy/mortgage-rates",
        "https://edition.cnn.com/2024/10/24/economy/why-mortgage-rates-rose-after-fed-cut",
    ]

    process_urls(urls)
    answer, source = generate_answer("Tell me what was the 30 year fixed mortgage rate along with the date")
    print(f'Answer: {answer}')
    print(f'Source: {source}')