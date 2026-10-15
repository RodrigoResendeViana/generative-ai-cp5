"""
Verificação de ambiente: GPU, bibliotecas e download do modelo.
Roda como notebook no VSCode (células `# %%`, botão Run Cell) ou como script:
    python scripts/00_smoke_test.py
Se terminar em "=== AMBIENTE OK ===", o resto do projeto pode rodar.
"""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import config
import utils

# %% 1) o PyTorch enxerga a GPU?
print(config.resumo_ambiente())
assert torch.cuda.is_available(), "CUDA indisponível — reinstale o torch com --index-url cu128 (ver docs/setup_gpu.md)"

# %% 2) a GPU realmente calcula? (matmul bf16 4096x4096, mede TFLOP/s)
a = torch.randn(4096, 4096, device=config.DEVICE, dtype=config.DTYPE)
b = torch.randn(4096, 4096, device=config.DEVICE, dtype=config.DTYPE)
torch.cuda.synchronize()
inicio = time.perf_counter()
N_ITER = 50
for _ in range(N_ITER):
    c = a @ b
torch.cuda.synchronize()
duracao = time.perf_counter() - inicio
flops = 2 * (4096 ** 3) * N_ITER
print(f"Matmul bf16 4096x4096 x{N_ITER}: {duracao:.3f}s -> {flops / duracao / 1e12:.1f} TFLOP/s")
del a, b, c
torch.cuda.empty_cache()

# %% 3) as bibliotecas estão instaladas?
LIBS = [
    "transformers", "datasets", "accelerate", "peft", "trl",
    "sentence_transformers", "faiss", "pymupdf", "fastapi", "uvicorn",
]
for nome in LIBS:
    mod = importlib.import_module(nome)
    versao = getattr(mod, "__version__", "?")
    print(f"  {nome:<22} {versao}")

# %% 4) o SmolLM2 baixa, carrega e gera?
utils.fixar_seed()
with utils.medir("carregar SmolLM2-1.7B-Instruct"):
    modelo, tok = utils.carregar(config.MODELO_CHAT)

with utils.medir("gerar resposta de teste"):
    resposta = utils.chat(modelo, tok, [
        {"role": "user", "content": "Em uma frase, o que é fine-tuning?"}
    ], max_new_tokens=60)
print("Resposta:", resposta)

utils.liberar(modelo)

print("\n=== AMBIENTE OK ===")
