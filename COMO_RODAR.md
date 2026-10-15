# Como rodar a aplicação

Guia prático, passo a passo, para colocar o Tutor de IA Generativa no ar — do zero ou
no dia a dia. Para entender o que o projeto faz e os resultados obtidos, veja o
[`README.md`](README.md); este arquivo é só sobre **rodar**.

Precisa de **dois terminais abertos ao mesmo tempo** (backend e frontend) depois do
setup inicial.

---

## 0. Pré-requisitos

| Item | Versão testada | Obrigatório? |
|---|---|---|
| Windows + PowerShell | Windows 11 | sim |
| GPU NVIDIA + driver atualizado | RTX 4070 8GB, driver 555.97 | recomendado (sem GPU, roda em CPU, bem mais lento) |
| Python | 3.12 (`py -3.12`) | sim — **não** use 3.13/3.14 |
| Node.js | 22.x | sim, só para o frontend |
| Git | qualquer versão recente | sim |

Verifique o que já tem instalado:

```powershell
nvidia-smi
py -0p
node --version
git --version
```

Se `nvidia-smi` não mostrar sua placa, ou `py -0p` não listar o 3.12, resolva isso
antes de continuar — veja [`docs/setup_gpu.md`](docs/setup_gpu.md) para o guia
completo de driver/CUDA.

---

## 1. Primeira vez: configurar o ambiente Python

Na raiz do projeto (pasta que contém `config.py`):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip

# PyTorch com CUDA — o índice NÃO é opcional, confira a versão certa em
# https://pytorch.org/get-started/locally (Windows · Pip · Python · CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# resto das bibliotecas
pip install -r requirements.txt
```

Confirme que o PyTorch enxerga a GPU:

```powershell
pip show torch
# a versão PRECISA terminar em +cu128 (ou outro +cuXXX) — nunca +cpu
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Depois, rode o teste de ambiente completo:

```powershell
python scripts/00_smoke_test.py
```

Precisa terminar em `=== AMBIENTE OK ===`. Se der erro, veja a tabela de problemas
comuns no final deste guia ou em [`docs/setup_gpu.md`](docs/setup_gpu.md).

> A cada nova sessão de trabalho (não só na primeira vez), ative o venv antes de
> rodar qualquer coisa: `.venv\Scripts\activate` (o prompt passa a mostrar `(.venv)`).

---

## 2. Conseguir um modelo treinado (o adaptador LoRA)

O backend precisa de um adaptador LoRA treinado em `runs/adaptador/` para servir o
modelo fine-tunado (sem ele, funciona só com o modelo base — ver seção 5). Essa pasta
é gitignored (não vem do `git clone`), então cada pessoa do grupo tem duas opções:

### Opção A — treinar localmente (recomendado, leva poucos minutos numa GPU)

```powershell
python scripts/01_baseline.py          # mede o modelo antes de treinar (opcional, mas recomendado)
python scripts/03_build_sft_dataset.py # gera data/sft_dataset.jsonl (se ainda não existir)
python scripts/05_hyperparam_sweep.py  # treina 4 configurações de LoRA e salva a melhor em runs/adaptador/
```

Nesta máquina (RTX 4070 8GB) o sweep completo levou **cerca de 4 minutos**. Ao final,
a mensagem `Adaptador da melhor configuração copiado para .../runs/adaptador` confirma
que deu certo.

### Opção B — usar o adaptador de outra pessoa do grupo

A pasta `runs/adaptador/` (tipicamente ~70 MB) pode ser compartilhada por fora do git
(Drive, WeTransfer, pendrive) e colada na mesma posição (`runs/adaptador/`) no seu
checkout do repositório.

---

## 3. (Opcional) Habilitar o RAG — busca nos PDFs reais das aulas

Sem isso o tutor funciona normalmente, só que sem citar trechos dos PDFs. Para
habilitar:

1. Copie os PDFs das aulas de IA Generativa para `data/corpus/` (pasta gitignored —
   não distribuímos o material do professor pelo GitHub, ver
   [`docs/dataset.md`](docs/dataset.md)).
2. Rode:

```powershell
python scripts/02_prep_corpus.py   # extrai texto dos PDFs -> data/chunks.jsonl
python scripts/06_rag_index.py     # monta o índice FAISS -> runs/rag_index/
```

---

## 4. Subir o backend e o frontend

Dois terminais separados, **ambos a partir da raiz do projeto**:

**Terminal 1 — backend (API FastAPI + modelo):**

```powershell
.venv\Scripts\activate
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000
```

Espere aparecer `Application startup complete` e `Uvicorn running on
http://127.0.0.1:8000`. O carregamento do modelo na primeira vez pode levar alguns
segundos (ou minutos, se for baixar o modelo do Hugging Face pela primeira vez).

Teste rápido num terceiro terminal (opcional):

```powershell
curl http://127.0.0.1:8000/health
# esperado: {"status":"ok","adapter_available":true,"rag_available":true}
```

**Terminal 2 — frontend (React/Vite):**

```powershell
cd frontend
npm install     # só na primeira vez (ou quando package.json mudar)
npm run dev
```

Espere aparecer `Local: http://localhost:5173/`.

**Abra `http://localhost:5173` no navegador.** Você verá:

- Badges indicando se o LoRA e o RAG estão disponíveis (vêm do `/health` do backend).
- Dois toggles: comparar modelo base vs fine-tunado, e ligar/desligar o RAG.
- Um campo de chat — digite uma pergunta sobre IA Generativa e envie.

---

## 5. Rodando sem adaptador treinado e/ou sem RAG

A aplicação **degrada graciosamente**:

- Sem `runs/adaptador/`: o backend sobe normalmente servindo só o modelo base; o
  toggle "modelo fine-tunado" fica desabilitado no frontend.
- Sem `runs/rag_index/`: o backend sobe normalmente sem RAG; o toggle "usar RAG" fica
  desabilitado.

Ou seja: dá para ver a aplicação funcionando (mesmo que só com o modelo base) logo
depois do passo 1, sem esperar nenhum treino.

---

## 6. Reproduzir o pipeline completo (treino, sweep, avaliação)

Para refazer todos os experimentos do zero, na ordem:

```powershell
.venv\Scripts\activate

python scripts/00_smoke_test.py        # valida GPU/ambiente
python scripts/01_baseline.py          # mede o modelo antes de treinar
python scripts/03_build_sft_dataset.py # gera data/sft_dataset.jsonl

python scripts/02_prep_corpus.py       # opcional — precisa dos PDFs em data/corpus/
python scripts/06_rag_index.py         # opcional — depende do passo anterior

python scripts/05_hyperparam_sweep.py  # treina as 4 configurações de LoRA, escolhe a melhor
python scripts/07_evaluate.py          # bake-off final: base vs LoRA vs LoRA+RAG
```

Resultados ficam em `runs/resultados/*.json` e `runs/sweep.png`.

---

## Problemas comuns

Problemas de GPU (`CUDA disponível: False`, `CUDA out of memory`, treino lento,
download do Hugging Face) estão em
[`docs/setup_gpu.md`](docs/setup_gpu.md#problemas-comuns). Os da aplicação:

| Sintoma | Causa provável | Correção |
|---|---|---|
| `ModuleNotFoundError: config` (ou `rag`, `utils`) | rodando um script de fora da raiz do projeto, ou VSCode com a pasta errada aberta como workspace | rode os comandos sempre a partir da raiz (`generative-ai-cp5/`); no VSCode, `File → Open Folder` nessa pasta |
| `adapter_available: false` no `/health` | não existe `runs/adaptador/` | treine (seção 2, opção A) ou copie de alguém do grupo (opção B) |
| `rag_available: false` no `/health` | não existe `runs/rag_index/` | rode a seção 3 (precisa dos PDFs em `data/corpus/`) |
| Frontend carrega mas o chat dá erro / "Backend indisponível" | backend não está rodando, ou está em outra porta | confirme que o terminal do backend mostra `Uvicorn running on http://127.0.0.1:8000`; se mudou a porta, ajuste `VITE_API_URL` no frontend |
| `npm install` falha ou `npm run dev` não encontra pacotes | Node desatualizado, ou instalação interrompida | confirme `node --version` ≥ 18, apague `frontend/node_modules` e rode `npm install` de novo |

---

## Checklist rápido (resumo de tudo)

```powershell
# 1. ambiente (uma vez)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/00_smoke_test.py

# 2. modelo treinado (uma vez, ou copiar runs/adaptador/ de alguém)
python scripts/03_build_sft_dataset.py
python scripts/05_hyperparam_sweep.py

# 3. subir a aplicação (toda vez)
# terminal 1
cd backend && ..\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000
# terminal 2
cd frontend && npm install && npm run dev
# navegador
http://localhost:5173
```
