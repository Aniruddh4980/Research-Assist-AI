# Research Assist AI

A Streamlit-based Retrieval-Augmented Generation (RAG) application that turns public web pages into a searchable knowledge base. Add URLs, index their content, and ask questions answered using retrieved document context—with source URLs for transparency.

![Research Assistant RAG interface](resources/ui.png)

## ✨ Features

- **URL ingestion** — load content from one or more public web pages.
- **Document processing** — split source content into chunks for retrieval.
- **Semantic search** — create embeddings with a Hugging Face embedding model.
- **Persistent vector storage** — store and retrieve chunks with a local Chroma database.
- **LLM-powered answers** — generate responses using a Groq-hosted language model.
- **Source visibility** — inspect the URLs returned as sources for an answer.
- **Interactive UI** — use a Streamlit chat interface with ingestion and query status indicators.

## 🧠 Architecture

```text
Public URLs
    │
    ▼
URL / document loader
    │
    ▼
Text splitting
    │
    ▼
Hugging Face embeddings
    │
    ▼
Chroma vector store (local, persistent)
    │
    ▼
Retriever ───────► Relevant document chunks
    │                         │
    └─────────────┬───────────┘
                  ▼
             Groq LLM
                  │
                  ▼
        Answer + source URLs
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| RAG orchestration / document processing | See `requirements.txt` |
| Embeddings | Hugging Face |
| Vector database | Chroma |
| Answer generation | Groq API |
| Language | Python |

> The exact package versions and additional libraries are defined in `requirements.txt`.

## ✅ Prerequisites

- Python 3.10 or newer
- A [Groq API key](https://console.groq.com/keys)
- Internet access to fetch web pages and download the embedding model on first use

## 🚀 Getting Started

### 1. Clone the repository

Replace the placeholders with your repository URL and directory name.

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_available_groq_model
EMBEDDING_MODEL=your_hugging_face_embedding_model
COLLECTION_NAME=research_store
```

Set each model variable to a model supported by your installed dependencies and available to your account. Keep `.env` private and do not commit API keys to version control.

### 5. Launch the application

```bash
streamlit run main.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

## 💬 How to Use

1. Start the app.
2. Paste one or more public article, documentation, report, or other web-page URLs into the sidebar.
3. Select **Process URLs** to load the pages, split their content, generate embeddings, and index the resulting chunks.
4. Ask a question in the chat interface—for example:
   - *What are the main findings across these sources?*
   - *How do the recommendations in these articles compare?*
   - *What evidence supports the conclusion?*
5. Expand **View Cited Sources** below an answer to inspect the source URLs returned by retrieval.

The local Chroma database is persisted under `resources/vectorstore`, allowing indexed content to remain available between launches.

## 📁 Project Structure

```text
.
├── main.py                  # Streamlit application and chat UI
├── rag.py                   # URL ingestion, retrieval, and answer generation
├── requirements.txt         # Python dependencies
└── resources/
    ├── ui.png               # Application screenshot
    └── vectorstore/         # Persistent Chroma database (generated at runtime)

```

*This structure reflects the files described in this README; update it if the repository layout changes.*

## ⚙️ Configuration

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key used to authenticate with Groq. |
| `GROQ_MODEL` | Groq chat model used to generate answers. |
| `EMBEDDING_MODEL` | Hugging Face embedding model used for semantic search. |
| `COLLECTION_NAME` | Name of the Chroma collection used to store indexed chunks. |

## 🔒 Security, Limitations & Notes

- Do not commit `.env` files, API keys, or other secrets. Add `.env` to `.gitignore`.
- Only ingest pages you are authorized to access and process; respect website terms and applicable copyright rules.
- Some websites block automated requests or render content dynamically, so ingestion may fail for certain URLs.
- Answer quality depends on the accessible source content, chunking, retrieval quality, and the selected language model.
- Source URLs indicate the retrieved references; they do not independently guarantee that every generated statement is fully supported by those sources.
- The app requires internet access for URL ingestion and calls to Groq. The embedding model may need an initial download.

## 🗺️ Potential Improvements

Ideas for future development—not necessarily implemented yet:

- Add automated tests for ingestion, retrieval, and answer generation.
- Add URL validation, duplicate detection, and clearer error handling.
- Add retrieval evaluation and configurable chunking / search parameters.
- Add Docker support and CI checks for repeatable setup.

## 📄 License

This project is described as available under the MIT License. See [`LICENSE`](LICENSE) for the license text.
