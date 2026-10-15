# Dataset

## Escopo do problema

O tutor responde sobre **IA Generativa / Modelos de Linguagem (LLMs)**: transformers,
atenção, tokenização, pré-treino vs fine-tuning, LoRA/QLoRA, esquecimento catastrófico,
RAG, embeddings, CUDA/GPU, autoencoders, prompting. Deliberadamente **não** cobre os
temas de ML clássico do 1º semestre da disciplina (regressão, PCA, XGBoost) — manter o
problema específico era um requisito do trabalho.

## Por que o conteúdo é texto original, não extraído dos slides

Os PDFs das aulas são material do professor. Para não versionar conteúdo protegido por
direitos autorais no GitHub:

- **`data/sft_dataset.jsonl`** (dataset de fine-tuning) e **`data/gabarito.json`**
  (avaliação) foram escritos pelo grupo, em texto original, cobrindo os mesmos
  conceitos ensinados na disciplina — não são transcrições dos slides.
- Os PDFs reais (`data/corpus/`) ficam só localmente (gitignored) e alimentam **apenas
  o RAG** (busca e citação de trechos) — essa camada funciona só em quem tiver os PDFs
  localmente; quem clona o repositório do GitHub treina o LoRA (ver `COMO_RODAR.md`,
  seção 2) e roda o tutor sem RAG.

## `data/sft_dataset.jsonl`

Gerado por `scripts/03_build_sft_dataset.py` a partir de uma lista curada de pares
pergunta/resposta definida no próprio script (editável ali mesmo). Formato de
chat (`prompt`/`completion` com roles), para o `SFTTrainer` calcular a loss só na
resposta, não na pergunta.

- **79 exemplos**: 74 pares cobrindo ~45 conceitos centrais da disciplina (1-2
  variações de pergunta por conceito) + 5 exemplos de recusa/abstenção para perguntas
  de cultura geral fora do escopo.
- Dividido em treino/validação (80/20, seed fixa) em `scripts/04_train_lora.py`.

## `data/gabarito.json`

24 perguntas escritas **antes** de rodar qualquer experimento (mesma regra do material
da disciplina: montar o gabarito depois de ver os resultados é marketing, não
engenharia):

- 20 perguntas "normais" (factuais ou de síntese) com uma lista de palavras-chave que
  precisam aparecer na resposta para contar como acerto.
- 4 perguntas "armadilha" — fora do escopo do tutor (ex.: "qual a capital da
  Austrália?") — cuja resposta correta é o modelo se abster, não inventar.

Usado por `scripts/07_evaluate.py` para comparar as rotas (base / LoRA / LoRA+RAG) de
forma consistente — ver `docs/evaluation.md`.

## RAG local (`data/corpus/`)

`scripts/02_prep_corpus.py` lê os PDFs reais das aulas (13 arquivos: Aulas 01-09 de IA
Generativa do 1º semestre + Redes Neurais/Autoencoder, Instalação CUDA/SmolLM2,
SmolLM2 conceitos e IA Generativa final do 2º semestre), extrai o texto por página com
PyMuPDF e divide em chunks de 500 tokens com 75 de sobreposição:

- 271 páginas com texto extraído, 16 páginas vazias/curtas descartadas.
- 275 chunks gerados, nenhum excede a janela de 8.192 tokens do modelo.

`scripts/06_rag_index.py` indexa os chunks com embeddings multilíngues
(`intfloat/multilingual-e5-small`) num índice FAISS (`IndexFlatIP`), salvo em
`runs/rag_index/` (gitignored).

## Limitações

- O dataset de SFT é pequeno (79 exemplos) — suficiente para demonstrar o efeito do
  LoRA (tom, formato, abstenção) dentro do prazo do trabalho, mas não para memorizar
  fatos com precisão; por isso o RAG complementa o LoRA para perguntas que dependem do
  conteúdo real das aulas.
- O RAG só funciona localmente (depende dos PDFs); a partir do repositório público, o
  tutor roda só com LoRA.
