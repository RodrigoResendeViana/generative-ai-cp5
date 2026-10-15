"""
RAG local sobre os PDFs da disciplina: índice FAISS + embeddings multilíngues.
Usado por scripts/06_rag_index.py (construir) e pelo backend (consultar).

Só funciona se data/corpus/ tiver PDFs e o índice já tiver sido construído —
quem clonar o repositório do GitHub não terá os PDFs (propriedade do professor),
então o backend precisa degradar graciosamente quando o índice não existe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config

_encoder = None


def carregar_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(config.MODELO_EMB, device=config.DEVICE)
    return _encoder


def _ler_chunks() -> list[dict]:
    if not config.CHUNKS_JSONL.exists():
        return []
    chunks = []
    with config.CHUNKS_JSONL.open(encoding="utf-8") as f:
        for linha in f:
            chunks.append(json.loads(linha))
    return chunks


def construir_indice() -> None:
    import faiss

    chunks = _ler_chunks()
    if not chunks:
        raise FileNotFoundError(
            f"{config.CHUNKS_JSONL} não existe ou está vazio. Rode scripts/02_prep_corpus.py primeiro."
        )

    enc = carregar_encoder()
    textos = [f"passage: {c['texto']}" for c in chunks]
    vetores = enc.encode(textos, normalize_embeddings=True, show_progress_bar=True)
    vetores = np.asarray(vetores, dtype="float32")

    indice = faiss.IndexFlatIP(vetores.shape[1])
    indice.add(vetores)

    config.RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(config.RAG_INDEX_DIR / "indice.faiss"))
    (config.RAG_INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Índice FAISS com {len(chunks)} chunks salvo em {config.RAG_INDEX_DIR}")


def indice_disponivel() -> bool:
    return (config.RAG_INDEX_DIR / "indice.faiss").exists() and (config.RAG_INDEX_DIR / "chunks.json").exists()


def carregar_indice():
    import faiss

    indice = faiss.read_index(str(config.RAG_INDEX_DIR / "indice.faiss"))
    chunks = json.loads((config.RAG_INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return indice, chunks


def buscar(pergunta: str, indice, chunks: list[dict], k: int = config.TOP_K) -> list[dict]:
    enc = carregar_encoder()
    vetor = enc.encode([f"query: {pergunta}"], normalize_embeddings=True)
    vetor = np.asarray(vetor, dtype="float32")
    scores, idxs = indice.search(vetor, k)
    resultados = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        item = dict(chunks[idx])
        item["score"] = float(score)
        resultados.append(item)
    return resultados


def montar_prompt(pergunta: str, trechos: list[dict]) -> str:
    contexto = "\n\n".join(
        f"[{t['arquivo']}, p.{t['pagina']}]\n{t['texto']}" for t in trechos
    )
    return (
        "Use os trechos abaixo para responder à pergunta. "
        "Se a resposta não estiver nos trechos, diga que não sabe.\n\n"
        f"--- Trechos ---\n{contexto}\n\n--- Pergunta ---\n{pergunta}"
    )
