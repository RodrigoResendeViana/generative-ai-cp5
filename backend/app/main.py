"""
API do tutor de IA Generativa.

    cd backend
    ../.venv/Scripts/uvicorn.exe app.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag
from app import inference

_rag_indice = None
_rag_chunks = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_indice, _rag_chunks
    inference.iniciar()
    if rag.indice_disponivel():
        _rag_indice, _rag_chunks = rag.carregar_indice()
        print(f"Índice RAG carregado: {len(_rag_chunks)} chunks")
    else:
        print("Índice RAG não encontrado — rode scripts/02_prep_corpus.py e scripts/06_rag_index.py para habilitar.")
    yield


app = FastAPI(title="Tutor de IA Generativa", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    use_finetuned: bool = True
    use_rag: bool = False


class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    adapter_available: bool
    rag_available: bool
    sources: list[str] = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "adapter_available": inference.tem_adaptador(),
        "rag_available": _rag_indice is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    inicio = time.perf_counter()
    contexto = None
    fontes: list[str] = []

    if req.use_rag and _rag_indice is not None:
        trechos = rag.buscar(req.message, _rag_indice, _rag_chunks)
        contexto = rag.montar_prompt(req.message, trechos)
        fontes = [f"{t['arquivo']} (p.{t['pagina']})" for t in trechos]

    resposta = inference.gerar(req.message, use_finetuned=req.use_finetuned, contexto_rag=contexto)
    latencia_ms = (time.perf_counter() - inicio) * 1000

    return ChatResponse(
        response=resposta,
        latency_ms=round(latencia_ms, 1),
        adapter_available=inference.tem_adaptador(),
        rag_available=_rag_indice is not None,
        sources=fontes,
    )
