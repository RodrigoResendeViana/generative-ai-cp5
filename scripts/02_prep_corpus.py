"""
Lê os PDFs de data/corpus/ (local, gitignored — não é distribuído no GitHub) e gera:
  - data/corpus.jsonl  : um registro por página (arquivo, página, texto)
  - data/chunks.jsonl  : chunks de ~500 tokens com 75 de sobreposição, para o RAG

    python scripts/02_prep_corpus.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz
from transformers import AutoTokenizer

import config

MIN_CHARS_PAGINA = 40


def extrair_paginas() -> list[dict]:
    registros = []
    pdfs = sorted(config.CORPUS.glob("*.pdf"))
    if not pdfs:
        print(f"Nenhum PDF encontrado em {config.CORPUS}. Copie os PDFs das aulas para lá e rode de novo.")
        return registros

    paginas_vazias = 0
    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        for i, pagina in enumerate(doc, start=1):
            texto = pagina.get_text("text").strip()
            if len(texto) < MIN_CHARS_PAGINA:
                paginas_vazias += 1
                continue
            registros.append({"arquivo": pdf_path.name, "pagina": i, "texto": texto})
        doc.close()

    print(f"{len(pdfs)} PDFs lidos, {len(registros)} páginas com texto, {paginas_vazias} páginas vazias/curtas.")
    if paginas_vazias > len(registros) * 0.3:
        print("Aviso: muitas páginas vazias — pode haver PDF escaneado (precisaria de OCR).")
    return registros


def montar_chunks(registros: list[dict]) -> list[dict]:
    tok = AutoTokenizer.from_pretrained(config.MODELO_CHAT)
    chunks = []
    excedeu_janela = 0
    janela = 8192

    for reg in registros:
        ids = tok(reg["texto"])["input_ids"]
        if len(ids) > janela:
            excedeu_janela += 1
        passo = config.CHUNK_TOKENS - config.CHUNK_OVERLAP
        for inicio in range(0, max(len(ids), 1), passo):
            fatia = ids[inicio: inicio + config.CHUNK_TOKENS]
            if not fatia:
                continue
            texto_chunk = tok.decode(fatia, skip_special_tokens=True)
            chunks.append({
                "arquivo": reg["arquivo"],
                "pagina": reg["pagina"],
                "texto": texto_chunk,
            })
            if inicio + config.CHUNK_TOKENS >= len(ids):
                break

    print(f"{len(chunks)} chunks gerados. {excedeu_janela} páginas excedem a janela de {janela} tokens sozinhas.")
    return chunks


def main() -> None:
    registros = extrair_paginas()
    if not registros:
        return

    with config.CORPUS_JSONL.open("w", encoding="utf-8") as f:
        for reg in registros:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    print(f"Gravado {config.CORPUS_JSONL}")

    chunks = montar_chunks(registros)
    with config.CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Gravado {config.CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
