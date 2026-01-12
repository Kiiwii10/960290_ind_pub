# TED Talk RAG Assistant

API-only Retrieval-Augmented Generation (RAG) assistant over an English TED dataset (metadata + transcripts), backed by Pinecone and deployed on Vercel.


## Note
```python
return ChatOpenAI(base_url=endpoint, api_key=key, model=model, temperature=1)
```

Sometimes calling the model (as shown above) doesnt run when manually setting tempeture, other times it is needed to set to 1. please be aware of this if the prompt endpoint fails because of it.

## Endpoints

- `GET /api/stats` → returns the chosen RAG hyperparameters (`chunk_size=1024`, `overlap_ratio=0.2`, `top_k=12`)
- `POST /api/prompt` → answers using only retrieved TED context

## How to Run the Project

### Prerequisites

- Python 3.8+
- Node.js (for frontend, optional)
- OpenAI-compatible API key (LLMod.ai)
- Pinecone account and API key

### 1. Backend Setup

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file in the project root by copying `.env.example`:
Then edit `.env` with your actual API keys


#### Ingest Data to Pinecone

Upload the TED talks dataset to Pinecone vector database:

```bash
python ingest.py
```

This processes `csv/ted_talks_en_subset.csv` and uploads embeddings to Pinecone using the hyperparameters defined in `config.py`.

#### Start the Backend Server

```bash
uvicorn index:app --reload
```

The API will be available at `http://localhost:8000`

### 2. Frontend Setup (Optional)

If you want to use the web UI:

#### Install Node Dependencies

```bash
npm install
```

#### Build Frontend

```bash
npm run build:frontend
```

This compiles TypeScript from `frontend/src/main.ts` to `frontend/dist/`.

#### Access the UI

Once the backend is running and the frontend is built, navigate to:

```
http://localhost:8000
```

### 3. Testing

Run the test suite, specify in .env the BASE_URL_TEST_RUN key.

```bash
python tests/run_rag_tests.py
```

## API Usage

### Get RAG Statistics

```bash
curl http://localhost:8000/api/stats
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "What are some TED talks about creativity?"}'
```

## Configuration

Edit `config.py` to adjust RAG hyperparameters:

- `chunk_size`: Token size for text chunks (default: 1024)
- `overlap_ratio`: Overlap between chunks (default: 0.2)
- `top_k`: Number of chunks to retrieve (default: 12)

## Project Structure

```
├── config.py              # RAG configuration and hyperparameters
├── index.py               # FastAPI application (main server)
├── ingest.py              # Data ingestion script for Pinecone
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies (frontend)
├── tsconfig.json          # TypeScript configuration
├── vercel.json            # Vercel deployment config
├── csv/                   # TED talks dataset
├── frontend/              # Web UI
│   └── src/
│       └── main.ts        # Frontend TypeScript code
└── tests/                 # Test suite
    ├── questions.json     # Test questions
    ├── run_rag_tests.py   # RAG test runner
    └── test.py            # Additional tests
```

## Deployment

This project is configured for deployment on Vercel using the `vercel.json` configuration file.
