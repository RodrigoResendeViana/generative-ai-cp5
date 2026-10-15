# Tutor de IA Generativa — CP5

Trabalho da disciplina **Generative AI for Engineering** (FIAP): treinar uma LLM
localmente na GPU, testar hiperparâmetros, avaliar as respostas e servir o modelo por
um frontend de chat.

**Equipe:**

| Nome | RM |
|---|---|
| Rodrigo Viana | 551057 |
| Caique Chagas | 551943 |
| Eduardo Fedeli | 550132 |
| Otavio Vitoriano | 552012 |
| Gabriel Torres | 98600 |

## O problema escolhido

Um **chatbot tutor especializado em IA Generativa/LLMs**, que responde perguntas sobre
transformers, atenção, tokenização, fine-tuning, LoRA/QLoRA, RAG, embeddings e
CUDA/GPU, os conceitos ensinados na própria disciplina. Perguntas fora desse escopo
(cultura geral, dados em tempo real) devem ser recusadas em vez de inventadas, e essa
abstenção é medida na avaliação.

## Requisitos do enunciado

| Requisito | Onde está |
|---|---|
| Selecionar uma LLM | `HuggingFaceTB/SmolLM2-1.7B-Instruct`: cabe numa GPU de 8 GB com LoRA e já vem com *chat template*. Justificativa no [relatório](RELATORIO.md#3-seleção-do-modelo) |
| Problema específico | Tutor de IA Generativa (acima). Dataset: [`docs/dataset.md`](docs/dataset.md) |
| Testar diferentes parâmetros | 4 configurações de LoRA (rank, learning rate, épocas): [`docs/training.md`](docs/training.md) |
| Avaliar o modelo (respostas) | Base vs LoRA vs LoRA+RAG contra um gabarito de 24 perguntas: [`docs/evaluation.md`](docs/evaluation.md) |
| Testes na GPU | Treino e inferência na RTX 4070 Laptop (8 GB): [`docs/setup_gpu.md`](docs/setup_gpu.md) |
| Frontend para o prompt | React (Vite) + API FastAPI: [`frontend/`](frontend/), [`backend/`](backend/) |
| Print da GPU no VSCode | [Comprovação de GPU](#comprovação-de-gpu), logo abaixo |
| Documentação e código no GitHub | Este repositório. Relatório completo: [`RELATORIO.md`](RELATORIO.md) |

## Resultados

### Teste de hiperparâmetros (LoRA)

| Configuração | Rank | Learning rate | Épocas | Eval loss | Pico de VRAM |
|---|---|---|---|---|---|
| r16_lr2e-4_ep3 | 16 | 2e-4 | 3 | 1.801 | 3.68 GB |
| r8_lr2e-4_ep3 | 8 | 2e-4 | 3 | 1.877 | 6.49 GB |
| r16_lr1e-4_ep3 | 16 | 1e-4 | 3 | 1.944 | 6.49 GB |
| **r16_lr2e-4_ep5** | **16** | **2e-4** | **5** | **1.728** | 6.53 GB |

Venceu a configuração com rank maior, learning rate maior e mais épocas. Ela é a que
o backend serve. Gráfico em [`runs/sweep.png`](runs/sweep.png).

### Avaliação das respostas

| Rota | Acerto no gabarito | Abstenção correta (perguntas fora do escopo) |
|---|---|---|
| Modelo base (sem treino) | 33% | 0% |
| + LoRA (fine-tuning local) | 42% | 0% |
| + LoRA + RAG | **54%** | **50%** |

O fine-tuning melhora o vocabulário e o enquadramento das respostas, mas sozinho não
ensina o modelo a se abster. É o RAG (busca nos PDFs das aulas) que faz o modelo
recusar perguntas fora do escopo: **RAG para o fato, LoRA para a forma**. As 24
respostas de cada rota estão em
[`runs/resultados/bakeoff.json`](runs/resultados/bakeoff.json).

## Arquitetura

```
PDFs das aulas → RAG (FAISS)  ─┐
                                ├─→ backend FastAPI (SmolLM2-1.7B + LoRA) → frontend React
data/sft_dataset.jsonl → LoRA ─┘
```

O backend carrega o modelo base uma única vez e liga/desliga o adaptador LoRA na mesma
cópia em memória, o que permite comparar base e fine-tunado dentro de 8 GB de VRAM.
Diagrama completo: [`docs/architecture.md`](docs/architecture.md).

## Comprovação de GPU

![GPU no VSCode](docs/screenshots/gpu_vscode.png)

Saída do `scripts/00_smoke_test.py` no terminal integrado do VSCode:

```
GPU          : NVIDIA GeForce RTX 4070 Laptop GPU (sm_89)
VRAM total   : 8.00 GB
bf16         : True
Matmul bf16 4096x4096 x50: 0.247s -> 27.9 TFLOP/s
[carregar SmolLM2-1.7B-Instruct] tempo=3.60s  pico_VRAM=3.20GB

=== AMBIENTE OK ===
```

## Como rodar

O guia completo, do zero, está em **[`COMO_RODAR.md`](COMO_RODAR.md)**: ambiente,
treino do adaptador, RAG, backend, frontend e problemas comuns. O adaptador LoRA e o
índice RAG não vão para o Git, então quem clona o repositório precisa treinar o
adaptador primeiro (cerca de 4 minutos numa GPU de 8 GB).

Com tudo configurado, subir a aplicação é:

```powershell
# terminal 1 — backend
.venv\Scripts\activate
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm run dev
```

Depois, abra `http://localhost:5173`.

## Estrutura do repositório

```
config.py, utils.py, textos.py,    módulos compartilhados (config.py concentra caminhos
rag.py, avaliacao.py               e hiperparâmetros)
scripts/                           pipeline numerado, de 00_smoke_test a 07_evaluate
data/                              gabarito.json e sft_dataset.jsonl (texto original do grupo)
runs/resultados/, runs/sweep.png   métricas do treino e da avaliação
backend/app/                       API FastAPI (inference.py carrega o modelo, main.py expõe /chat)
frontend/                          React + Vite
docs/                              detalhes de GPU, dataset, treino, avaliação e arquitetura
RELATORIO.md                       relatório acadêmico do trabalho
```

Ficam fora do Git, porque são gerados localmente ou pesados demais: adaptadores e
checkpoints (`runs/adaptador*`, `runs/treino/`), o índice FAISS (`runs/rag_index/`) e
o ambiente `.venv`.

## Dados das aulas

Os PDFs das aulas são material do professor e **não são versionados**. O dataset de
treino e o gabarito foram escritos pelo grupo em texto original; os PDFs só alimentam
o índice RAG local. Detalhes em [`docs/dataset.md`](docs/dataset.md).
