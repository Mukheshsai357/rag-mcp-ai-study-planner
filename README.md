# RAG + MCP Knowledge & Action Assistant

A learning/portfolio project using FastAPI, PostgreSQL + pgvector, Sentence Transformers, Ollama, and the Python MCP SDK.

## What it does

- Upload text-based PDF documents.
- Extract text, split it into overlapping chunks, embed chunks with `all-MiniLM-L6-v2`, and save vectors in PostgreSQL.
- Search semantically similar chunks using pgvector cosine distance.
- Expose MCP tools to an AI client: `list_documents`, `search_documents_tool`, , `generate_study_plan`, `list_study_tasks`, and `update_study_task_status`.
- Use an Ollama chat model with tool calling to decide when to call those MCP tools.
- Generate and save PDF-grounded study plans; list tasks and track pending/in_progress/completed status.
- Offer API docs at `http://127.0.0.1:8000/docs`.

> Learning-project note: this is a local starter, not a production deployment. It does not include user authentication, multi-user data isolation, OCR for scanned PDFs, robust conversation persistence, or a production-grade authorization/approval system. Do not expose it publicly with real private documents until you add those controls.

## Prerequisites

- Python 3.11 or 3.12
- Docker Desktop (for PostgreSQL with pgvector)
- Ollama installed locally
- Internet access on first run to download Python packages and the embedding model

## 1. Start PostgreSQL

Open a terminal in this project directory:

```powershell
docker compose up -d
```

This starts PostgreSQL on `localhost:5432` with database `rag_mcp`, user `postgres`, and password `postgres`. These credentials are for local development only.

If port 5432 is already in use, change the left-hand port in `docker-compose.yml`, then update `DATABASE_URL` in `.env` to match.

## 2. Create the Python environment (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## 3. Start the local LLM

Install Ollama from its official website, then run:

```powershell
ollama pull llama3.2
```

Make sure the Ollama application/service is running. The default API is `http://localhost:11434`.

The first use of the embedding model downloads it from Hugging Face and can take a while. The LLM and embedding model are separate: Ollama generates answers; Sentence Transformers creates vectors for search.

## 4. Start FastAPI

From the project root, with the virtual environment activated:

```powershell
uvicorn app.main:app --reload
```

On startup, the app enables the PostgreSQL `vector` extension and creates the tables. Open:

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 5. Try it

### Upload a PDF

In `/docs`, open `POST /documents/upload`, click **Try it out**, select a text-based PDF, and execute. Or use PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" -F "file=@C:\path\to\your-document.pdf"
```

### Ask a question

In `/docs`, try `POST /chat` with:

```json
{
  "message": "Summarize the main points in the uploaded document and tell me which file you used."
}
```

The model should call `search_documents_tool` through MCP and use the retrieved chunks to answer. Tool-calling behavior depends on the Ollama model version; if it doesn't call tools reliably, try a different Ollama model with tool support and update `OLLAMA_MODEL` in `.env`.

### Test semantic search directly

Use `POST /search`:

```json
{
  "message": "What does the document say about the main topic?"
}
```

### Create a task

Use `POST /tasks` for direct API creation, or ask `/chat` to create a task. Review the returned result and check `GET /tasks`.

### List uploaded documents

Use `GET /documents`, or ask the chat assistant to list the uploaded documents (which should call the MCP tool).

## How the code is organized

```text
app/
  config.py     # environment configuration
  db.py         # SQLAlchemy engine and DB initialization
  models.py     # documents, chunks, tasks
  rag.py        # PDF extraction, chunking, embeddings, vector retrieval
  llm.py        # Ollama HTTP client
  main.py       # FastAPI endpoints and MCP client/tool-calling loop
mcp_server/
  server.py     # MCP tools exposed to the AI client
```

## Important design notes

- RAG = retrieve relevant document chunks, then give those chunks to the LLM as context.
- MCP = standard interface for exposing tools/resources to an AI application. This project runs the MCP server as a separate subprocess using stdio.
- PostgreSQL stores normal application data; pgvector adds vector similarity search.
- Embeddings have 384 dimensions because `all-MiniLM-L6-v2` outputs 384-dimensional vectors. If you change the embedding model, update the vector column dimension and recreate/migrate the table accordingly.
- PDF text extraction will not read most scanned PDFs. Add OCR (for example, Tesseract) if needed.
- Retrieved text can contain malicious prompt-injection instructions. The system prompt warns the model to treat it as untrusted, but this is not a complete security boundary. Validate tool inputs and add authentication/authorization before real use.

## Troubleshooting

- **Database connection error:** check `docker compose ps`, the port, and `DATABASE_URL` in `.env`.
- **`vector` extension error:** make sure you're using the `pgvector/pgvector` Docker image and that the database user can create extensions.
- **Ollama connection error:** start Ollama and run `ollama list` to confirm the model is present.
- **Model answers without searching:** the model may not support tool calling reliably. Try a different tool-capable Ollama model and update `.env`.
- **PDF has no extracted text:** it may be scanned; OCR is needed.
- **Slow first request:** the embedding model is being downloaded/loaded into memory.

## Stop PostgreSQL

```powershell
docker compose down
```

To delete the local database volume too (this permanently deletes stored project data):

```powershell
docker compose down -v
```


## PDF-grounded study plans (new)

1. Upload a text-based PDF with `POST /documents/upload`. Note its `document_id`.
2. Generate a checklist with `POST /study-plans/generate`:

```json
{"document_id": 1, "task_count": 5}
```

3. See saved items using `GET /study-tasks?document_id=1`.
4. Mark one as completed with `PATCH /study-tasks/1`:

```json
{"status": "completed"}
```

You can also ask the `/chat` endpoint to generate a plan, list tasks, or mark one complete; the model calls the corresponding MCP tools. The tool is only supposed to save a plan when you explicitly request it.

**Notes:** The plan is generated by Ollama from up to 12,000 characters of the uploaded document, so long PDFs may not be fully represented. Check generated tasks for accuracy. This is not a reminder or scheduling system. Generating a plan again saves additional tasks. The older `tasks` table may remain in an existing database, but the new feature uses `study_tasks`. `Base.metadata.create_all()` creates the new table; it does not perform schema migrations.
