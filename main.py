import os
import streamlit as st

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Catalyst AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. Load Streamlit Cloud Secrets
# --------------------------------------------------

for key in [
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "EMBEDDING_MODEL",
    "COLLECTION_NAME"
]:
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

# Import after loading secrets because rag.py reads
# environment variables during module initialization.

from rag import (
    process_urls,
    generate_answer,
    initialize_components
)

# --------------------------------------------------
# 3. Modern Custom CSS & Animations
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.gradient-header {
    background: linear-gradient(
        90deg,
        #3b82f6 0%,
        #8b5cf6 50%,
        #ec4899 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.3rem;
    margin-bottom: 0.2rem;
}

.sub-caption {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 500;
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.18);
    margin-top: 8px;
    width: 100%;
}

.pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.pulse-green {
    background-color: #10b981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-green 2s infinite;
}

.pulse-amber {
    background-color: #f59e0b;
    box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
    animation: pulse-amber 2s infinite;
}

@keyframes pulse-green {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

@keyframes pulse-amber {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 8px rgba(245, 158, 11, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0);
    }
}

.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    background: rgba(128, 128, 128, 0.07);
    border: 1px solid rgba(128, 128, 128, 0.14);
    border-radius: 14px;
    margin: 4px 0;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 50%;
    animation: bounce-pulse 1.4s infinite ease-in-out both;
}

.typing-dot:nth-child(1) {
    animation-delay: -0.32s;
}

.typing-dot:nth-child(2) {
    animation-delay: -0.16s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0s;
}

@keyframes bounce-pulse {
    0%, 80%, 100% {
        transform: scale(0.5);
        opacity: 0.4;
    }
    40% {
        transform: scale(1.15);
        opacity: 1;
    }
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(128, 128, 128, 0.12);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(139, 92, 246, 0.3);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
}

.stTextInput input {
    border-radius: 10px !important;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. Session State Initialization
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! Paste your articles or listing URLs "
                "in the sidebar and click **Process URLs** to begin."
            ),
            "sources": None
        }
    ]

if "urls_processed" not in st.session_state:
    st.session_state.urls_processed = False

if "url_count" not in st.session_state:
    st.session_state.url_count = 3

if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0

if "ingestion_error" not in st.session_state:
    st.session_state.ingestion_error = None


# --------------------------------------------------
# 5. Initialize RAG Components
# --------------------------------------------------

try:
    vector_store = initialize_components()

    current_count = vector_store._collection.count()

    st.session_state.indexed_count = current_count

except Exception as e:

    st.error(f"❌ RAG initialization failed: {e}")
    st.stop()


# --------------------------------------------------
# 6. Sidebar: Knowledge Base
# --------------------------------------------------

with st.sidebar:

    st.markdown("### 📚 Knowledge Base")
    st.caption("Feed any URLs into the RAG vector index.")

    urls = []

    for i in range(st.session_state.url_count):

        url = st.text_input(
            f"URL {i + 1}",
            key=f"url_{i}",
            placeholder="https://example.com..."
        )

        if url.strip():
            urls.append(url.strip())

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "➕ Add URL",
            use_container_width=True
        ):
            st.session_state.url_count += 1
            st.rerun()

    with col2:

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    process_url_button = st.button(
        "🚀 Process URLs",
        type="primary",
        use_container_width=True
    )

    ingest_pulse = st.empty()

    # ----------------------------------------------
    # Process URLs
    # ----------------------------------------------

    if process_url_button:

        if not urls:

            st.warning(
                "⚠️ You must provide at least one valid URL."
            )

        else:

            st.session_state.urls_processed = False
            st.session_state.ingestion_error = None

            try:

                ingest_pulse.markdown("""
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <span style="font-size: 0.82rem; color: #64748b; margin-left: 6px;">
                        Indexing URLs...
                    </span>
                </div>
                """, unsafe_allow_html=True)

                with st.status(
                    "Processing Knowledge Base...",
                    expanded=True
                ) as status:

                    for step_status in process_urls(urls):
                        st.write(f"✔️ {step_status}")

                    # Verify actual database contents
                    actual_count = vector_store._collection.count()

                    if actual_count == 0:
                        raise RuntimeError(
                            "No documents found in ChromaDB."
                        )

                    st.session_state.indexed_count = actual_count
                    st.session_state.urls_processed = True

                    status.update(
                        label=f"✅ Vector Database Ready ({actual_count} chunks)",
                        state="complete",
                        expanded=False
                    )

                ingest_pulse.empty()

                st.toast(
                    f"Successfully indexed {actual_count} chunks!",
                    icon="✨"
                )

            except Exception as e:

                ingest_pulse.empty()

                st.session_state.urls_processed = False
                st.session_state.ingestion_error = str(e)

                st.error(
                    f"❌ URL processing failed: {str(e)}"
                )

    st.divider()

    # ----------------------------------------------
    # Live Database Status
    # ----------------------------------------------

    try:

        actual_count = vector_store._collection.count()

        st.session_state.indexed_count = actual_count

    except Exception:
        actual_count = 0

    if actual_count > 0:

        st.markdown(f"""
        <div class="status-badge">
            <span class="pulse-dot pulse-green"></span>
            <span>Index Available &bull; {actual_count} Chunks</span>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="status-badge">
            <span class="pulse-dot pulse-amber"></span>
            <span>Empty Knowledge Base &bull; Awaiting URLs</span>
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# 7. Main Chat Area
# --------------------------------------------------

st.markdown(
    '<div class="gradient-header">🔬 Catalyst AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-caption">'
    'Ask questions and synthesize insights directly '
    'from your indexed URLs.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 8. Render Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📌 View Cited Sources"):
                st.write(message["sources"])


# --------------------------------------------------
# 9. Chat Input & Answer Generation
# --------------------------------------------------

if query := st.chat_input(
    "Ask a question about the processed URLs..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        pulse_loader = st.empty()

        pulse_loader.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="font-size: 0.85rem; color: #64748b; margin-left: 8px;">
                Searching sources & synthesizing...
            </span>
        </div>
        """, unsafe_allow_html=True)

        try:

            # Always verify the actual database count
            actual_count = vector_store._collection.count()

            if actual_count == 0:

                raise RuntimeError(
                    "Your vector database is empty. "
                    "Please process URLs first."
                )

            answer, sources = generate_answer(query)

            pulse_loader.empty()

            st.markdown(answer)

            if sources:

                with st.expander("📌 View Cited Sources"):
                    st.write(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except RuntimeError as e:

            pulse_loader.empty()

            error_msg = f"⚠️ {str(e)}"

            st.warning(error_msg)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": None
                }
            )

        except Exception as e:

            pulse_loader.empty()

            error_msg = f"❌ Error generating response: {str(e)}"

            st.error(error_msg)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": None
                }
            )
