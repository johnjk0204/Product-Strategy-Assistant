# AI Product Strategy Assistant

A multi-agent AI system that transforms raw business data into actionable product strategy reports. Upload sales data, customer reviews, or market research documents and receive a full suite of strategic outputs in minutes.

---

## Features

**9-Agent Analysis Pipeline**

| Agent | Output |
|---|---|
| Customer Feedback Agent | Customer Insights Report |
| Market Research Agent | Market Research Summary |
| Competitor Analysis Agent | Competitor Analysis Report |
| SWOT Analysis Agent | SWOT Analysis |
| Opportunity Assessment Agent | Product Opportunity Assessment |
| Feature Prioritization Agent | Feature Prioritization Recommendations |
| Product Roadmap Agent | Product Roadmap Suggestions |
| Strategy Recommendation Agent | Strategic Action Plan |
| Executive Report Agent | Executive Summary |

**Additional capabilities**
- Downloadable PDF report covering all 9 sections
- Natural language chat — ask questions about your data and analysis results
- Supports CSV, TXT, PDF, and JSON input files

---

## Tech Stack

- **LLM:** GPT-4o Mini (via gateway)
- **Embeddings:** text-embedding-3-small (via gateway)
- **Orchestration:** LangGraph
- **Vector Store:** ChromaDB
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **PDF Generation:** fpdf2

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/johnjk0204/Product-Strategy-Assistant.git
cd Product-Strategy-Assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://keygateway.arshnivlabs.com/v1
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=./chroma_db
REPORTS_DIR=./reports
```

---

## Running the App

**Windows (recommended)**
```
.\start.bat
```

This opens two terminal windows — the FastAPI backend on port 8000 and the Streamlit frontend on port 8501 — and launches the browser automatically.

**Manual start**

Terminal 1 — Backend:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 — Frontend:
```bash
cd frontend
streamlit run app.py --server.port 8501
```

Then open `http://localhost:8501`.

---

## Usage

1. **Upload** — Click "Load Sample Sales Data" or upload your own CSV/TXT/PDF/JSON files
2. **Analyse** — Click "Run Full Analysis" in the sidebar to start the 9-agent pipeline (~2-3 minutes)
3. **Explore** — Browse results across 9 report tabs in the main area
4. **Chat** — Ask natural language questions in the Chat tab
5. **Download** — Click "Download PDF Report" to get a boardroom-ready PDF

---

## Project Structure

```
├── backend/
│   ├── main.py                  # FastAPI app and endpoints
│   ├── config.py                # Environment configuration
│   ├── graph/
│   │   ├── state.py             # LangGraph state definition
│   │   └── workflow.py          # 9-agent pipeline
│   └── utils/
│       ├── document_processor.py  # CSV/PDF/TXT/JSON parsing
│       ├── vector_store.py        # ChromaDB wrapper
│       └── pdf_generator.py       # PDF report generation
├── frontend/
│   └── app.py                   # Streamlit UI
├── data/
│   └── sample/
│       └── sample_sales_data.csv
├── requirements.txt
└── start.bat
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload and index data files |
| POST | `/api/analyze` | Run the 9-agent pipeline |
| GET | `/api/status/{id}` | Poll analysis progress |
| POST | `/api/chat` | Natural language query |
| GET | `/api/download/{id}` | Download PDF report |
| GET | `/health` | Health check |
