# Catalyst AI - Research Assistant using RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) application that transforms public web pages into a searchable knowledge base. Ingest URLs, index their content, and ask questions using retrieved document context, with source references to help verify the answers.

Live Link: https://research-assist-ai.streamlit.app/

![Research Assistant RAG Interface](resources/ui.png)

## ✨ Features

- **Web Content Ingestion:** Load content from multiple public URLs using LangChain's `WebBaseLoader`.
- **Document Processing:** Split extracted content into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Semantic Embeddings:** Generate document embeddings using LangChain's `HuggingFaceEmbeddings` integration.
- **Vector Database:** Store and retrieve document embeddings using Chroma.
- **Session-Based Knowledge Bases:** Create a separate Chroma collection for each Streamlit session, preventing users from retrieving one another's indexed documents.
- **Duplicate-Safe Indexing:** Generate deterministic chunk IDs using SHA-256 hashing. Reprocessing an unchanged URL updates existing records instead of creating duplicate chunks.
- **Content Refresh:** Replace outdated chunks when a successfully fetched webpage changes, while preserving other URLs in the session's knowledge base.
- **URL Processing Feedback:** Display an individual status for each submitted URL: **Indexed**, **Blocked**, or **Failed**.
- **Partial Ingestion:** Continue indexing accessible URLs even when another URL is blocked or fails.
- **Retrieval-Augmented Generation:** Use LangChain's `RetrievalQAWithSourcesChain` to combine relevant document retrieval with LLM-powered answer generation.
- **LLM Integration:** Generate responses using Groq-hosted language models through LangChain's `ChatGroq` integration.
- **Source Attribution:** Display source URLs associated with retrieved information.
- **Persistent Local Storage:** Store session-specific collections in a Chroma database under `resources/vectorstore`.
- **Interactive Chat Interface:** Explore indexed web content through a Streamlit application with chat history and knowledge-base status indicators.

## 🏗️ Architecture

The application uses LangChain to connect document ingestion, embedding, retrieval, and answer generation. Each Streamlit session receives a unique workspace ID, which is used to create its own Chroma collection.

```text
                         Public Web URLs
                                │
                                ▼
                           WebBaseLoader
                                │
                                ▼
                    RecursiveCharacterTextSplitter
                                │
                                ▼
                       HuggingFaceEmbeddings
                                │
                                ▼
                    Deterministic Chunk ID Generation
                                │
                                ▼
                    Session-Specific Chroma Collection
                         (Persistent Storage)
                                │
                                ▼
                             Retriever
                                │
                                ▼
                    RetrievalQAWithSourcesChain
                                │
                                ▼
                             ChatGroq
                                │
                                ▼
                         Generated Answer
                                │
                                ▼
                           Source URLs
```

### How the RAG Pipeline Works

1. **Session Initialization:** Generate a unique workspace ID for the current Streamlit session and initialize its Chroma collection.
2. **Ingestion:** Load content from user-provided URLs using `WebBaseLoader`.
3. **Chunking:** Divide extracted content into smaller text chunks using `RecursiveCharacterTextSplitter`.
4. **Embedding:** Convert document chunks into vector embeddings using a Hugging Face embedding model.
5. **Duplicate-Safe Indexing:** Generate deterministic IDs for chunks and store them in the session's Chroma collection. Reprocessing the same URL does not create duplicate records.
6. **Content Refresh:** When a URL is successfully processed again, update its chunks and remove obsolete chunks from that URL. Other URLs remain unchanged.
7. **Retrieval:** Find relevant document chunks only within the current session's collection.
8. **Generation:** Pass the retrieved context to a Groq-hosted language model through LangChain.
9. **Source Attribution:** Return the generated answer alongside source references.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Session Management | Streamlit Session State |
| RAG Framework | LangChain |
| Retrieval Chain | `RetrievalQAWithSourcesChain` |
| URL Loader | `WebBaseLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Embedding Integration | `HuggingFaceEmbeddings` |
| Embedding Models | Hugging Face |
| Vector Database | Chroma |
| Vector Store Integration | `langchain-chroma` |
| LLM Integration | `ChatGroq` |
| LLM Provider | Groq |
| Configuration Management | `python-dotenv` |
| Chunk ID Generation | Python `hashlib` (SHA-256) |

The project uses modular LangChain integrations for document loading, text splitting, embeddings, vector storage, and LLM interaction.

## 📋 Prerequisites

Before running the application, ensure you have:

- Python 3.10 or newer.
- A [Groq API key](https://console.groq.com/keys).
- Internet access to retrieve web content and download the embedding model when required.
- The dependencies listed in `requirements.txt`.

## 🚀 Installation

### 1. Clone the Repository

Replace the placeholders with your repository URL and directory name.

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip and install the project dependencies.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_available_groq_model
EMBEDDING_MODEL=your_hugging_face_embedding_model
```

Configure each variable according to the models and settings supported by your implementation.

**Security:** Never commit your `.env` file or expose API keys in your repository. Add `.env` to `.gitignore`.

For Streamlit Cloud, configure the required credentials and model settings through the app's deployment secrets/environment configuration.

### 5. Run the Application

Launch the Streamlit application:

```bash
streamlit run main.py
```

Streamlit will display a local URL in the terminal. Open it in your browser, typically at:

```text
http://localhost:8501
```

## 💬 Usage

1. Launch the application.
2. Enter one or more public web URLs in the sidebar.
3. Click **Process URLs** to load, split, embed, and index the content.
4. Check the URL status displayed in the sidebar:
   - **Indexed:** The page was successfully processed and stored.
   - **Blocked:** The website denied the request, for example with HTTP 401 or 403.
   - **Failed:** The page could not be loaded or meaningful text could not be extracted.
5. Ask questions about the successfully indexed documents through the chat interface.
6. Expand **View Cited Sources** below an answer to inspect the associated source URLs.

A blocked or failed URL does not prevent other accessible URLs from being indexed. If no URL can be indexed, the knowledge base remains empty unless it already contained previously indexed documents.

### Reprocessing URLs

You can click **Process URLs** again for a URL that has already been indexed:

- Reprocessing an unchanged page does not increase its stored chunk count.
- If the page content changes, its previous chunks are replaced with the updated chunks.
- If fetching or extracting the page fails, its previously indexed content is retained.
- Processing one URL does not remove the indexed content of other URLs in the same session.

### Session Isolation

Catalyst AI does not require users to log in. Instead, each Streamlit session receives a unique workspace ID and a separate Chroma collection.

This means:

- Each session retrieves information only from its own collection.
- Documents indexed in one independent session are not included in another session's retrieval.
- Chat history and workspace identification are maintained using Streamlit session state.
- **Clear Chat** clears the conversation history; it does not delete the session's indexed documents.

The session UUID provides workspace separation, not user authentication. It should not be treated as an account identity or as a substitute for authorization in a system with sensitive data.

### Example Questions

- What are the main findings in these articles?
- Summarize the key points from the provided sources.
- Compare the recommendations across these documents.
- What evidence supports the main conclusion?
- Identify the major differences between the indexed articles.

The quality of the answers depends on source accessibility, document processing, retrieval relevance, and the selected language model.

## 📁 Project Structure

```text
.
├── main.py                  # Streamlit user interface and session management
├── rag.py                   # RAG pipeline, URL ingestion, and answer generation
├── requirements.txt         # Python dependencies
└── resources/
    ├── ui.png               # Application screenshot
    └── vectorstore/         # Persistent Chroma database and session collections
```

The directory structure above reflects the project layout described in this README. Update it if your repository differs.

## ⚙️ Configuration Reference

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key used to authenticate requests to Groq. |
| `GROQ_MODEL` | Groq-hosted language model used to generate answers. |
| `EMBEDDING_MODEL` | Hugging Face embedding model used for semantic search. |

The application creates a unique Chroma collection name for each Streamlit session using its generated workspace UUID. The collections are stored under `resources/vectorstore`; a shared `COLLECTION_NAME` environment variable is not required by the current session-isolated implementation.

## 🔒 Security and Limitations

- **API key security:** Store credentials in environment variables or Streamlit deployment secrets and never commit them to version control.
- **Session isolation:** Separate Streamlit sessions use separate Chroma collections. This prevents normal cross-session retrieval, but session IDs are not authentication credentials.
- **Collection cleanup:** Session collections are not automatically deleted when a browser session ends. Unused collections can accumulate and may require a cleanup strategy.
- **Deployment storage:** Chroma uses local persistent storage. On hosted platforms, the durability of local files depends on the platform's filesystem and restart/redeployment lifecycle; local persistence should not be assumed to be permanent cloud storage.
- **Website accessibility:** Some websites restrict automated requests or require JavaScript rendering, which may prevent successful ingestion. HTTP 401 and 403 responses are reported as blocked.
- **Source quality:** Incomplete or inaccurate source content can affect the quality of generated answers.
- **Retrieval limitations:** Relevant information may not always be retrieved for every question.
- **LLM limitations:** Generated responses may contain unsupported claims. Review the cited sources when accuracy matters.
- **Internet dependency:** URL ingestion and Groq-hosted inference require network connectivity.

Only process web content that you are authorized to access and use, and respect applicable website terms and copyright requirements.

## 🔮 Potential Future Improvements

The following are potential enhancements rather than claims about existing functionality:

- Add automated cleanup or time-to-live (TTL) handling for abandoned session collections.
- Introduce user authentication and authorization if persistent user workspaces are required.
- Move to a managed vector database for durable multi-user deployment.
- Evaluate retrieval quality using a dedicated RAG evaluation framework.
- Add configurable chunk sizes, overlap settings, and retrieval parameters.
- Introduce more advanced retrieval strategies and metadata filtering.
- Add Docker support for reproducible deployment.
- Implement CI workflows for automated testing and code quality checks.

## 📄 License

This project is distributed under the MIT License, as referenced by the repository's `LICENSE` file.