# FinSolve Technologies - Internal RBAC RAG Chatbot

Production-ready internal chatbot using FastAPI + Streamlit + LangChain + ChromaDB with server-side role-based access control.

## Features
- JWT authentication (`/auth/login`, `/auth/me`)
- Strict server-side RBAC filtering at retrieval time
- Department-isolated Chroma collections (`finance`, `marketing`, `hr`, `engineering`, `general`)
- Groq LLM integration via `langchain_groq.ChatGroq` (`llama-3.1-8b-instant`)
- Streamlit chat UI with login, role badge, chat history, and source display

## Install dependencies
```bash
pip install -r requirements.txt
```

## Environment setup
1. Get a free Groq API key from [console.groq.com](https://console.groq.com)
2. Copy env template:
   ```bash
   cp .env.example .env
   ```
3. Fill your `.env`:
   - `GROQ_API_KEY=<your_key>`
   - optionally change JWT/chroma/resources settings

## Run ingestion
This project uses the existing data files under `resources/data` as-is and maps them to RBAC collections.

```bash
python app/ingest.py
```

## Run backend
```bash
uvicorn app.main:app --reload --port 8000
```

## Run frontend
```bash
streamlit run streamlit_app.py
```

## Test users
- `alice` / `password123` -> `finance`
- `bob` / `password123` -> `marketing`
- `carol` / `password123` -> `hr`
- `dave` / `password123` -> `engineering`
- `eve` / `password123` -> `c_level`
- `frank` / `password123` -> `employee`

## Example queries by role
- finance (`alice`): "What were the total reimbursements last quarter?"
- marketing (`bob`): "How did the Q3 campaign perform?"
- hr (`carol`): "Show me the attendance summary for October"
- engineering (`dave`): "What is our microservices architecture?"
- c_level (`eve`): "Give me a full company performance overview"
- employee (`frank`): "What is the leave policy?"

## API endpoints
- `POST /auth/login`
- `GET /auth/me`
- `POST /chat`
- `GET /health`

## RBAC behavior
- Non-authorized cross-department requests return: `I don't have access to that information`
- `c_level` can retrieve from all collections
- `employee` can access only `general`
