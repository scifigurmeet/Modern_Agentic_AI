# Problem Statement: Movie Recommendation RAG System

## Objective
Build a Retrieval-Augmented Generation (RAG) system over a movie dataset that answers natural-language queries such as *"Suggest a sci-fi movie starring Leonardo DiCaprio"* or *"What are the themes in Inception?"* — using **ChromaDB** as the vector store, **without using LangChain**.

## Provided Dataset
A CSV file `movies.csv` with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `title` | Movie name | Inception |
| `year` | Release year | 2010 |
| `genre` | Comma-separated genres | Sci-Fi, Thriller |
| `cast` | Lead actors | Leonardo DiCaprio, Ellen Page |
| `director` | Director name | Christopher Nolan |
| `theme` | Central themes | dreams, memory, reality, heist |
| `plot` | 2–3 sentence summary | A thief who steals corporate secrets... |

## Requirements

1. **Ingest & embed** — Load the CSV, convert each row into a text document, embed it, and store it in a persistent ChromaDB collection along with metadata (`title`, `year`, `genre`, `director`).
2. **Query** — Accept a plain-text question, embed it, retrieve the top-k most relevant rows via semantic search, and generate a natural-language answer grounded in the retrieved context.
3. **No LangChain** — Use the `chromadb` client directly. Embeddings via ChromaDB's built-in embedding function or `sentence-transformers` directly. LLM call via a raw SDK (Groq / Gemini / OpenAI).
4. **Metadata filtering** — Support at least one filtered query (e.g., only movies where `year > 2010`).

## ⚠️ The Chunk-Size Consideration (Core Learning Point)

**Each CSV row is one self-contained semantic unit — do NOT split rows across chunks, and do NOT merge multiple movies into one chunk.**

This is the crux of the problem. Unlike free-flowing prose (articles, PDFs) where you slice text into 500–1000 token windows, structured CSV data has a **natural chunk boundary: the row itself.**

Participants must justify their chunking strategy by measuring actual row sizes:

- **One row = one chunk (recommended).** A typical movie row (~50–150 tokens) fits comfortably in a single embedding. Splitting it would separate `cast` from `theme`, destroying the semantic link the query depends on (e.g., "DiCaprio + dreams" would no longer co-occur in any single vector).
- **If rows are large** (long `plot` + many cast members exceed the embedding model's optimal window, e.g. >256 tokens for `all-MiniLM-L6-v2`), consider embedding a concatenation of only the high-signal fields (`title + genre + cast + theme`) and keeping the long plot in metadata for display only.
- **Never batch multiple movies into one chunk** — retrieval granularity collapses; a query for one movie pulls back a blob referencing five, diluting relevance and confusing the LLM.

**Deliverable requirement:** Participants must print the token/character length distribution of their rows and explicitly state their chunk size decision with reasoning.

## Sample Queries to Pass
1. "Recommend a psychological thriller."
2. "Which movies feature dreams or memory as a theme?"
3. "Find a Christopher Nolan film released after 2010." *(tests metadata filter)*
4. "What is Inception about?"

## Evaluation Criteria
- Correct ChromaDB ingestion with metadata (25%)
- Retrieval quality — relevant rows returned for each query (30%)
- **Justified chunking decision backed by row-size analysis (25%)**
- Working end-to-end query→answer flow without LangChain (20%)

---

Want me to also generate the **starter `movies.csv` dataset** and a **skeleton solution file** (for reference/grading), or keep this as a pure problem statement?
