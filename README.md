# Production-Grade Document RAG System

A production-quality Retrieval-Augmented Generation (RAG) system built from scratch using **FastAPI**, **Streamlit**, **LangChain**, **ChromaDB**, **SentenceTransformers**, and the **Groq API**.

This system supports intelligent document ingestion (PDF, DOCX, TXT, MD), semantic-aware section chunking, multilingual search queries, automatic source citation extraction, and multi-document semantic contradiction checking.

---

## Architecture Diagram

The decoupled architectural layout is described below:

```
                  ┌────────────────────────────────────────┐
                  │              Streamlit UI              │
                  └───────────────────┬────────────────────┘
                                      │ (HTTP REST)
                                      ▼
                  ┌────────────────────────────────────────┐
                  │             FastAPI Backend            │
                  │   (Routing, Middleware, Dependency)    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │           Services Orchestrator        │
                  │              (RAG Pipeline)            │
                  └────────┬──────────┬──────────┬─────────┘
                           │          │          │
         ┌─────────────────┘          │          └─────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Vector Index    │        │ Translation / LLM│        │ Citation Tracker │
│ (ChromaDB + ST)  │        │   (Groq Llama)   │        │ (Regex Mapping)  │
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

---

## Data Flow Diagram

```
[Query Input] ──► [Language Detection] ──► (if not English: Translate to EN)
                                                    │
                                                    ▼
[ChromaDB Search] ◄── [Embed query using S-Transformers] ◄── [EN Query]
       │
       ▼
[Retrieve Top 5 Chunks] ──► [Construct Context Prompt] ──► [LLM Strict Answer Generation]
                                                                    │
                                                                    ▼
[Response Output] ◄── [Translate back to Source Lang] ◄── [Align & Verify Citations [N]]
```

---

## Folder Explanation

```
document-rag-system/
├── README.md                 # Detailed documentation and architectural guide
├── requirements.txt          # Production dependencies configuration
├── .env.example              # Environment variables template
├── .gitignore                # File exclusions configuration
├── app/
│   ├── main.py               # FastAPI application setup, middleware and routes registration
│   ├── config.py             # Configuration loader class utilizing Pydantic Settings
│   ├── dependencies.py       # Global singletons instantiation and DI providers
│   ├── models.py             # Shared domain data models (DocumentChunk, etc.)
│   ├── schemas.py            # API request/response validation schemas
│   ├── routes/               # API Router modules
│   │   ├── ask.py            # /ask (Context QA pipeline endpoint)
│   │   ├── upload.py         # /upload (Document ingestion endpoint)
│   │   ├── contradict.py     # /contradict (Contradiction checker endpoint)
│   │   └── health.py         # /health (System status check endpoint)
│   └── services/             # Core Service Layer
│       ├── document_loader.py # Multi-format document parser (PDF, DOCX, TXT, MD)
│       ├── chunking.py       # Structural, semantic-aware splitters
│       ├── embedding.py      # Locally cached SentenceTransformers embeddings
│       ├── vector_store.py   # Persisted ChromaDB collection manager & duplicate check
│       ├── retriever.py      # Standard Similarity and MMR search controllers
│       ├── llm.py            # Groq API client interface
│       ├── citation.py       # Citation mapping and cleaning
│       ├── translation.py    # Multi-language translation layer
│       ├── contradiction.py  # Retrieve-and-compare document contradiction service
│       └── rag_pipeline.py   # RAG pipeline orchestrator
├── ui/
│   └── streamlit_app.py      # Streamlit browser interface
├── data/
│   ├── documents/            # Ingested local files storage
│   ├── vector_db/            # Persisted ChromaDB database directories
│   └── bootstrap_documents.py# Tool to programmatically populate mock documents
└── tests/                    # Pytest test suite
    ├── test_ingestion.py     # Loader & Chunker tests
    ├── test_retrieval.py     # Vector store queries, duplicate check, and MMR tests
    ├── test_citations.py     # Citation validation tests
    ├── test_contradiction.py # Contradiction model checks
    └── test_translation.py   # Translation pipeline checks
```

---

## Key Features & Design Implementations

### 1. Chunking Strategy
Instead of a naive character-count splitter, the system implements a **section-aware chunking pipeline**:
- Ingested text is analyzed via regular expressions to detect structural headings (Markdown headers like `#`, `##` or short capitalized text blocks).
- Text is divided into logical section segments based on these headers.
- Each section is then split independently using a `RecursiveCharacterTextSplitter` (chunk size = 700 characters, overlap = 120 characters) with separators `["\n\n", "\n", " ", ""]`.
- The section heading is prepended to chunks to retain contextual awareness.
- This prevents paragraph splitting and guarantees that titles are stored directly in the chunk's metadata.

### 2. Embeddings and Caching
- Embeddings are generated using the `sentence-transformers` library with the default model `all-MiniLM-L6-v2`.
- Embeddings are cached in-memory using an MD5-hash dictionary to avoid redundant computations when re-loading or updating collections.

### 3. Vector Database
- Powered by `ChromaDB` persisted locally in `./data/vector_db`.
- Implements duplicate detection using SHA-256 chunk hashes (`chunk_id`). Existing chunks are skipped during incremental ingestion.

### 4. Retrieval and MMR
- Matches queries using cosine similarity (top K = 5).
- Features optional **Maximal Marginal Relevance (MMR)** re-ranking to balance high-similarity matches with semantic diversity, helping avoid redundant context in the LLM prompt.

### 5. Prevent Hallucination
The prompt instructs Llama 3.3 to answer **ONLY** from the retrieved context. If the information is missing, the model returns:
`"I could not find this information in the provided documents."`

### 6. Citation Design
The LLM outputs bracketed markers matching the context index (e.g. `[1]`, `[2]`). The system parses these tags, maps them back to the retrieved chunks, validates the metadata, and returns clean citations with source filenames, pages, chunk IDs, and snippet text.

### 7. Contradiction Detection
Retrieves the top 5 chunks on a topic from two distinct documents and uses Llama 3.3 (JSON Mode) to classify the relationship as `Conflict`, `No Conflict`, or `Partial Conflict`, explaining the discrepancies and providing evidence quotes.

### 8. Translation Workflow
Detects input languages (English, Hindi, French, Spanish, German, Japanese, Marathi). Non-English queries are translated to English for vector search and QA. The final response is translated back, preserving citation markers (e.g. `[1]`) at correct semantic positions.

---

## Installation & Setup

1. **Clone project directories** and navigate to root:
   ```bash
   cd document-rag-system
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\\Scripts\\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**:
   Create a `.env` file in the root directory (based on `.env.example`):
   ```ini
   GROQ_API_KEY=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
   GROQ_MODEL=llama-3.3-70b-versatile
   EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   CHROMA_PERSIST_DIR=./data/vector_db
   UPLOAD_DIR=./data/documents
   ```

5. **Bootstrap mock documents**:
   Populate realistic mock files (including PDF, DOCX, TXT, MD) inside `./data/documents` for testing:
   ```bash
   python data/bootstrap_documents.py
   ```

---

## Running the Application

### 1. Run FastAPI Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open interactive swagger docs at `http://127.0.0.1:8000/docs`.

### 2. Run Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```
This opens the frontend panel in your default browser at `http://localhost:8501`.

---

## Testing

Run the pytests to verify loaders, retrievers, translation pipelines, citations, and contradiction flows:
```bash
pytest tests/
```

---

## API Examples

### Ask Question (`POST /ask`)
**Request:**
```json
{
  "question": "What is the policy on maternity leave?",
  "use_mmr": false
}
```
**Response:**
```json
{
  "question": "What is the policy on maternity leave?",
  "answer": "Maternity leave is provided up to 26 weeks of paid leave [1].",
  "language": "English",
  "citations": [
    {
      "source_file": "company_hr_policy.docx",
      "page": 1,
      "chunk_id": "doc_a1b2c3d4_p1_c0_hash1234",
      "snippet": "Maternity leave is provided up to 26 weeks of paid leave, and paternity leave is provided up to 4 weeks of paid leave.",
      "similarity_score": 0.892
    }
  ],
  "latency_seconds": 0.542
}
```

### Contradiction Audit (`POST /contradict`)
**Request:**
```json
{
  "document1": "company_hr_policy.docx",
  "document2": "cybersecurity_handbook.pdf",
  "topic": "password policy"
}
```
**Response:**
```json
{
  "status": "No Conflict",
  "reasoning": "Document 2 specifies strict password policies. Document 1 does not mention password policies, resulting in no direct contradiction.",
  "document1_evidence": "No relevant text matching password policy found.",
  "document2_evidence": "All corporate passwords must be at least 14 characters in length...",
  "confidence": 0.98
}
```
