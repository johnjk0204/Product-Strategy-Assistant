"""
FastAPI backend for the AI Product Strategy Assistant.

Endpoints:
  POST /api/upload          – upload one or more data files
  POST /api/analyze         – run the 7-agent LangGraph pipeline
  GET  /api/status/{id}     – poll analysis progress
  POST /api/chat            – interactive RAG-based chat
  GET  /api/download/{id}   – download the generated PDF report
  GET  /health              – health check
"""

import os
import uuid
from typing import List, Dict, Any

import httpx
import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, EMBEDDING_MODEL, REPORTS_DIR
from graph.workflow import run_analysis
from utils.document_processor import DocumentProcessor
from utils.pdf_generator import generate_pdf
from utils.vector_store import VectorStore

# ---------------------------------------------------------------------------
app = FastAPI(title="Product Strategy Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_processor = DocumentProcessor()
_vs = VectorStore(
    openai_api_key=OPENAI_API_KEY,
    embedding_model=EMBEDDING_MODEL,
    openai_base_url=OPENAI_BASE_URL,
)
_openai = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    http_client=httpx.Client(verify=False),
)

# In-memory session registry
_sessions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: List[Dict[str, str]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload files first.")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Accept one or more data files, chunk them, and store in ChromaDB."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "status": "ready",
        "files": [],
        "total_chunks": 0,
        "analysis": {},
        "pdf_path": None,
        "error": None,
    }

    total_chunks = 0
    try:
        for f in files:
            raw = await f.read()
            chunks = _processor.process(f.filename, raw)
            n = _vs.add_documents(session_id, chunks, {"filename": f.filename})
            total_chunks += n
            _sessions[session_id]["files"].append({"filename": f.filename, "chunks": n})
    except Exception as exc:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())

    _sessions[session_id]["total_chunks"] = total_chunks

    return {
        "session_id": session_id,
        "files_processed": len(files),
        "total_chunks": total_chunks,
        "status": "ready",
    }


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the full 7-agent LangGraph pipeline and generate a PDF report."""
    sess = _session(req.session_id)
    sess["status"] = "analyzing"

    try:
        report_data = await run_analysis(req.session_id, _vs)
        sess["analysis"] = report_data
        sess["status"] = "complete"

        # Generate PDF
        pdf_path = os.path.join(REPORTS_DIR, req.session_id, "report.pdf")
        generate_pdf(report_data, pdf_path)
        sess["pdf_path"] = pdf_path

        return {"status": "complete", "session_id": req.session_id, "analysis": report_data}

    except Exception as exc:
        sess["status"] = "error"
        sess["error"] = str(exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/status/{session_id}")
def get_status(session_id: str):
    """Return current session status and analysis if complete."""
    sess = _session(session_id)
    return {
        "status": sess["status"],
        "files": sess["files"],
        "total_chunks": sess["total_chunks"],
        "analysis_keys": list(sess["analysis"].keys()) if sess["analysis"] else [],
        "pdf_ready": sess["pdf_path"] is not None,
        "error": sess.get("error"),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """RAG-based chat: retrieve relevant context then answer with the LLM."""
    sess = _session(req.session_id)

    # Retrieve document context
    doc_chunks = _vs.query(req.session_id, req.message, n_results=5)
    context = "\n\n---\n\n".join(doc_chunks) if doc_chunks else "No relevant data found."

    # Include executive summary if analysis is done
    analysis = sess.get("analysis", {})
    exec_summary = analysis.get("executive_summary", "")
    extra = f"\n\nEXECUTIVE SUMMARY:\n{exec_summary[:600]}" if exec_summary else ""

    system_prompt = (
        "You are an AI Product Strategy Assistant. "
        "Answer questions using the business data and analysis results provided. "
        "Be specific, cite numbers where available, and keep answers concise.\n\n"
        f"RELEVANT BUSINESS DATA:\n{context}{extra}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.message})

    resp = _openai.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=messages,
        temperature=0.4,
    )
    return {
        "response": resp.choices[0].message.content,
        "context_chunks_used": len(doc_chunks),
    }


@app.get("/api/download/{session_id}")
def download_report(session_id: str):
    """Serve the generated PDF report."""
    sess = _session(session_id)
    pdf_path = sess.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not yet generated. Run analysis first.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"product_strategy_report_{session_id[:8]}.pdf",
    )
