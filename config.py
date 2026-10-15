"""
Configuração central do projeto CP5 — Tutor de IA Generativa.
Todos os scripts importam daqui — não repita caminhos nem hiperparâmetros.
"""
from pathlib import Path
import os
import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------- caminhos
RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "data"
CORPUS = DADOS / "corpus"          # PDFs das aulas (gitignored, local)
RUNS = RAIZ / "runs"               # checkpoints, adaptadores, índice, gráficos (gitignored)
RESULTADOS = RUNS / "resultados"

GABARITO = DADOS / "gabarito.json"
SFT_DATASET = DADOS / "sft_dataset.jsonl"
CORPUS_JSONL = DADOS / "corpus.jsonl"
CHUNKS_JSONL = DADOS / "chunks.jsonl"

ADAPTADOR_DIR = RUNS / "adaptador"       # LoRA treinado com os hiperparâmetros padrão
RAG_INDEX_DIR = RUNS / "rag_index"       # índice FAISS + metadados dos chunks

for _p in (DADOS, CORPUS, RUNS, RESULTADOS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- hardware
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
    USA_BF16 = True
else:
    DTYPE = torch.float32
    USA_BF16 = False

ATTN = "sdpa"

# ---------------------------------------------------------------- modelos
MODELO_CHAT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"   # modelo final do tutor (LoRA)
MODELO_CHAT_360 = "HuggingFaceTB/SmolLM2-360M-Instruct"  # usado p/ iterar rápido no sweep, se necessário
MODELO_EMB = "intfloat/multilingual-e5-small"          # embeddings multilíngues p/ RAG

# ---------------------------------------------------------------- RAG
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 75
TOP_K = 4

# ---------------------------------------------------------------- LoRA / SFT padrão
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_DROPOUT = 0.05

SFT_MAX_LENGTH = 768
SFT_LR = 2e-4
SFT_EPOCHS = 3
SFT_GRAD_ACCUM = 8


def resumo_ambiente() -> str:
    linhas = [
        f"PyTorch      : {torch.__version__}",
        f"CUDA         : {torch.version.cuda}",
        f"Device       : {DEVICE}",
        f"dtype        : {DTYPE}",
        f"attn         : {ATTN}",
    ]
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        livre, total = torch.cuda.mem_get_info()
        linhas += [
            f"GPU          : {p.name} (sm_{p.major}{p.minor})",
            f"VRAM total   : {total / 1024**3:.2f} GB",
            f"VRAM livre   : {livre / 1024**3:.2f} GB",
            f"bf16         : {torch.cuda.is_bf16_supported()}",
        ]
    return "\n".join(linhas)


if __name__ == "__main__":
    print(resumo_ambiente())
