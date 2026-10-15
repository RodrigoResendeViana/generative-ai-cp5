# Treino — LoRA SFT e teste de hiperparâmetros

## Modelo e método

`HuggingFaceTB/SmolLM2-1.7B-Instruct`, fine-tunado com **LoRA** (não fine-tuning
completo) via `SFTTrainer` (TRL) sobre `data/sft_dataset.jsonl` (79 exemplos, split
80/20 treino/validação, seed 42). `target_modules`: `q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj`. `gradient_checkpointing=True`, `bf16=True`, batch 1 +
`gradient_accumulation_steps=8` (batch efetivo 8), `max_length=768`,
`EarlyStoppingCallback(patience=2)`.

## Baseline (antes de qualquer treino)

`scripts/01_baseline.py` mediu o modelo intocado antes de treinar — essencial para
saber se o fine-tuning realmente ajudou:

- Perplexidade held-out, domínio IA Generativa (nunca visto): **46.72**
- Perplexidade held-out, geral (fora do domínio): **21.13**
- Qualitativamente, o modelo base confunde conceitos do domínio — ex.: descreveu LoRA
  como "Lightning Object Detection", um modelo de visão computacional, não a técnica de
  fine-tuning (ver `runs/resultados/baseline.json` para as gerações completas).

## Hiperparâmetros testados

`scripts/05_hyperparam_sweep.py` rodou 4 configurações (grade pequena, um eixo por vez
a partir do default — não o produto cartesiano completo, para caber no tempo do
grupo), todas na RTX 4070 8GB:

| run | r | alpha | lr | épocas | params treináveis | train_loss | eval_loss | tempo | pico VRAM |
|---|---|---|---|---|---|---|---|---|---|
| r16_lr2e-4_ep3 | 16 | 32 | 2e-4 | 3 | 18.1M (1.05%) | 1.804 | 1.801 | 44.9s | 3.68 GB |
| r8_lr2e-4_ep3  | 8  | 16 | 2e-4 | 3 | 9.0M (0.53%)  | 1.910 | 1.877 | 42.8s | 6.49 GB |
| r16_lr1e-4_ep3 | 16 | 32 | 1e-4 | 3 | 18.1M (1.05%) | 1.967 | 1.944 | 43.1s | 6.49 GB |
| **r16_lr2e-4_ep5** | **16** | **32** | **2e-4** | **5** | 18.1M (1.05%) | **1.575** | **1.728** | 71.8s | 6.53 GB |

Gráfico comparativo: `runs/sweep.png`. Resumo bruto: `runs/resultados/sweep_resumo.json`.

### Leitura dos resultados

- **Rank maior (r=16) bateu rank menor (r=8)** com a mesma lr/épocas — mais
  capacidade no adaptador ajudou, e o dataset é pequeno o bastante para não
  overfitar visivelmente nesse rank.
- **lr=2e-4 bateu lr=1e-4** — com um dataset pequeno e poucos passos de treino, uma
  taxa de aprendizado maior converge mais rápido (consistente com a prática de LoRA
  usar lr bem mais alta que fine-tuning completo).
- **5 épocas bateu 3 épocas** sem sinal de overfitting ainda (eval_loss continuou
  caindo) — o `EarlyStoppingCallback` não interrompeu nenhum dos runs antes do fim.
- A configuração vencedora (**r=16, lr=2e-4, 5 épocas**) foi copiada automaticamente
  para `runs/adaptador/` — é essa que o backend serve por padrão.

## Orçamento de VRAM observado

Pico de VRAM durante o treino variou entre 3.68 GB e 6.53 GB nos 4 runs — dentro do
orçamento de ~5-6 GB livres calculado para esta GPU de 8 GB (ver `docs/setup_gpu.md`).
Nenhum run precisou reduzir mais o batch ou a sequência.
