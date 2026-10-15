"""
Constrói o índice FAISS local para RAG a partir de data/chunks.jsonl
(gerado por scripts/02_prep_corpus.py).

    python scripts/06_rag_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag

if __name__ == "__main__":
    rag.construir_indice()
