# Setup de GPU — CUDA, PyTorch e VSCode

Testado nesta máquina: **NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)**, driver 555.97 (suporta até CUDA 12.8), Windows 11.

## 1. Confirme o driver

No PowerShell:

```powershell
nvidia-smi
```

Precisa aparecer o nome da sua placa e uma linha `CUDA Version: XX.X`. Esse número é a
versão **máxima** de CUDA que o driver suporta — não é a que está instalada, e você
**não precisa instalar o CUDA Toolkit separado**: o wheel do PyTorch já traz o runtime.

Se o comando falhar, instale/atualize o driver da NVIDIA e reinicie antes de continuar.

## 2. Ambiente virtual (Python 3.12)

Use Python 3.10+ — este projeto foi testado com **Python 3.12**, não com versões mais
novas (3.13/3.14), que ainda não têm suporte garantido no PyTorch/transformers/peft.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

## 3. PyTorch com CUDA — o passo que todo mundo erra

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Confira o índice correto para a sua placa em <https://pytorch.org/get-started/locally/>
(Windows · Pip · Python · CUDA). **Nunca** rode `pip install torch` sem o `--index-url`:
isso instala a build CPU, sem avisar, e o treino roda dezenas de vezes mais lento.

Verifique:

```powershell
pip show torch   # a versão precisa terminar em +cu128 (ou outro +cuXXX)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Nesta máquina, o resultado foi:

```
torch: 2.11.0+cu128
cuda disponivel: True
gpu: NVIDIA GeForce RTX 4070 Laptop GPU
bf16: True
```

## 4. Resto das bibliotecas

```powershell
pip install -r requirements.txt
```

`torch` fica de fora do `requirements.txt` de propósito (ver comentário no arquivo):
se estivesse lá, o pip o baixaria do PyPI comum (build CPU) e desfaria o passo 3.
`bitsandbytes` também não é usado — no Windows nativo seu suporte é instável, e LoRA
em bf16 já cabe confortavelmente no orçamento de VRAM desta GPU (ver abaixo).

## 5. VSCode

1. `File → Open Folder` → selecione a pasta `generative-ai-cp5` (precisa ser esta
   pasta, que contém `config.py` — senão `import config` falha).
2. `Ctrl+Shift+P` → `Python: Select Interpreter` → escolha o `.venv` do projeto.
3. Extensões: **Python**, **Jupyter**, e uma extensão de monitoramento de GPU
   (ex.: **GPU Monitor** ou **GPU Environment**) para acompanhar a VRAM na barra de
   status durante o treino — é dessa extensão que vem o print pedido no enunciado.
   O print está em [`screenshots/gpu_vscode.png`](screenshots/gpu_vscode.png).

## 6. Validar tudo

```powershell
python scripts/00_smoke_test.py
```

Esperado: termina em `=== AMBIENTE OK ===`. Nesta máquina, a saída real foi:

```
PyTorch      : 2.11.0+cu128
CUDA         : 12.8
Device       : cuda
dtype        : torch.bfloat16
GPU          : NVIDIA GeForce RTX 4070 Laptop GPU (sm_89)
VRAM total   : 8.00 GB
VRAM livre   : 6.89 GB
bf16         : True
Matmul bf16 4096x4096 x50: 0.247s -> 27.9 TFLOP/s
[carregar SmolLM2-1.7B-Instruct] tempo=3.60s  pico_VRAM=3.20GB
[gerar resposta de teste] tempo=1.43s  pico_VRAM=3.21GB

=== AMBIENTE OK ===
```

## Orçamento de VRAM (8 GB)

| modo | VRAM |
|---|---|
| Windows + apps em repouso | ~1.1-2.3 GB (feche launchers/overlays de jogos antes de treinar) |
| Inferência bf16 (SmolLM2-1.7B) | ~3.2 GB |
| LoRA bf16 (treino, batch 1 + grad accum) | 3.7-6.5 GB de pico, medido nesta GPU |

Se aparecer `CUDA out of memory`: reduza `per_device_train_batch_size` (já está em 1),
aumente `gradient_accumulation_steps`, reduza `max_length`, e confirme que
`gradient_checkpointing=True` está ativo (já é o padrão em `config.py`).

## Problemas comuns

| Sintoma | Causa | Correção |
|---|---|---|
| `CUDA disponível: False` | PyTorch instalado sem `--index-url` (build CPU) | `pip uninstall -y torch` e refazer o passo 3 |
| `CUDA out of memory` | VRAM ocupada por outros programas, ou batch/sequência grandes demais | feche apps que usam GPU (jogos, overlays); ver "Orçamento de VRAM" acima |
| Download lento / erro 429 do Hugging Face | requisições não autenticadas | `huggingface-cli login` |
| Treino lento com GPU ociosa | threads do DataLoader brigando no Windows | `dataloader_num_workers=0` (já é o padrão) |
