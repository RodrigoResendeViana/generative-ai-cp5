# Arquitetura

## Visão geral

```
                         ┌─────────────────────────┐
                         │   data/corpus/ (PDFs)    │  local, gitignored
                         └────────────┬─────────────┘
                                      │ 02_prep_corpus.py
                                      ▼
                         ┌─────────────────────────┐
                         │  data/chunks.jsonl        │
                         └────────────┬─────────────┘
                                      │ 06_rag_index.py (e5-small + FAISS)
                                      ▼
                         ┌─────────────────────────┐
                         │  runs/rag_index/           │  local, gitignored
                         └────────────┬─────────────┘
                                      │
  data/sft_dataset.jsonl             │
          │                          │
          │ 04_train_lora.py         │
          │ (SFTTrainer + LoRA)      │
          ▼                          │
  runs/adaptador/ (LoRA) ────────────┤
          │                          │
          ▼                          ▼
     ┌────────────────────────────────────┐
     │     backend/app (FastAPI)          │
     │  base model (SmolLM2-1.7B-Instruct)│
     │  + adaptador LoRA (PEFT)           │
     │  + busca RAG opcional              │
     │  POST /chat                        │
     └───────────────┬─────────────────────┘
                      │ fetch JSON
                      ▼
     ┌────────────────────────────────────┐
     │     frontend (React + Vite)        │
     │  chat + toggles base/LoRA, RAG      │
     └────────────────────────────────────┘
```

## Por que LoRA + RAG juntos

Mesma conclusão a que o bake-off do material da disciplina chega: **RAG para o fato,
LoRA para a forma**. LoRA (fine-tuning) ensina o modelo a responder no tom, formato e
vocabulário esperados de um tutor; RAG ancora a resposta em trechos reais dos PDFs das
aulas, reduzindo alucinação sobre conteúdo específico, e é também o que faz o modelo se
abster quando a pergunta foge do escopo (o LoRA sozinho não aprendeu isso). `scripts/07_evaluate.py`
mede as três rotas (base / LoRA / LoRA+RAG) separadamente contra o mesmo gabarito para
comprovar isso — ver `docs/evaluation.md`.

## Uma cópia do modelo em VRAM, não duas

`backend/app/inference.py` carrega o modelo base **uma única vez** e aplica o adaptador
LoRA por cima com PEFT. Para comparar "modelo base" vs "modelo fine-tunado" no
frontend, o backend usa `model.disable_adapter()` para desligar o LoRA temporariamente
na mesma cópia do modelo em memória, em vez de carregar duas cópias completas — essencial
para caber no orçamento de 8 GB de VRAM desta GPU.

## Degradação graciosa

- Se `runs/adaptador/` não existir (LoRA ainda não treinado — caso de quem acabou de
  clonar o repositório), o backend serve só o modelo base e o toggle de fine-tuning
  fica desabilitado no frontend.
- Se `runs/rag_index/` não existir (sem os PDFs locais), o backend serve sem RAG e o
  toggle de RAG fica desabilitado.

## Por que FastAPI + React separados (em vez de uma UI only-Python)

Escolha do grupo: separar backend (serve o modelo, GPU) de frontend (interface do
usuário) deixa claro, na arquitetura, qual parte depende da GPU e qual não — o
frontend pode, inclusive, ser aberto em outra máquina apontando para o backend via
`VITE_API_URL`.
