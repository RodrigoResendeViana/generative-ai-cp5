# Avaliação

## Metodologia

`data/gabarito.json` — 24 perguntas (20 normais + 4 armadilhas), escritas **antes** de
rodar qualquer experimento (ver `docs/dataset.md`). `scripts/07_evaluate.py` roda o
gabarito contra três configurações, usando a mesma pontuação (`avaliacao.py`) para
todas:

- **acerto_total**: % de perguntas cuja resposta contém as palavras-chave esperadas
  (ou, nas armadilhas, cuja resposta se absteve).
- **abstenção correta**: nas 4 perguntas armadilha (fora do escopo), % em que o
  modelo respondeu "não sei"/equivalente em vez de inventar.
- **abstenção indevida**: nas 20 perguntas normais, % em que o modelo se absteve
  quando deveria ter respondido.

Pontuação por palavra-chave é deliberadamente simples (mesmo critério usado para
comparar as três rotas entre si, não para publicar como benchmark absoluto) — um
julgamento humano ou um LLM-as-judge calibrado seria o próximo passo natural.

## Resultado — bake-off final

| rota | acerto total | abstenção correta (armadilhas) | abstenção indevida |
|---|---|---|---|
| **base** (SmolLM2-1.7B-Instruct, sem treino) | 33% | 0% | 0% |
| **LoRA** (fine-tunado, sem RAG) | 42% | 0% | 0% |
| **LoRA + RAG** (fine-tunado + busca nos PDFs) | **54%** | **50%** | 0% |

JSON completo (todas as 24 respostas, das 3 rotas): `runs/resultados/bakeoff.json`.

## Leitura dos resultados

- **O fine-tuning (LoRA) sozinho já melhora o acerto geral** (33% → 42%) — o modelo
  aprende o vocabulário e o enquadramento certo dos conceitos (ver exemplo abaixo).
- **Nenhuma das duas rotas sem RAG aprendeu a se abster** (0% de abstenção correta) —
  o modelo base e o LoRA sozinho sempre tentam responder, mesmo a perguntas fora do
  escopo (ex.: respondem "Canberra" para a capital da Austrália em vez de dizer que
  não sabem). Isso bate com a limitação discutida em `docs/dataset.md`: só 5 dos 79
  exemplos de treino eram de recusa, pouco para fixar esse comportamento sozinho.
- **RAG é o que resolve a abstenção**: com RAG, quando a pergunta foge do conteúdo
  recuperado dos PDFs, o modelo tende a dizer que não sabe em vez de inventar — daí o
  salto de 0% para 50% de abstenção correta.
- **Confirma a receita do material da disciplina**: "RAG para o fato, LoRA para a
  forma" — a melhor configuração final é justamente a combinação das duas rotas, não
  uma sozinha.

## Exemplo concreto (pergunta "Qual a diferença entre fine-tuning completo e LoRA?")

- **base**: confunde LoRA com uma técnica de visão computacional ("Layer-wise
  Relevance Training"/detecção de objetos) — resposta incorreta e incoerente.
- **LoRA**: aborda os termos da pergunta (custo de treino, learning rate), mas inventa
  o significado da sigla ("Learning Rate Reducer") e erra a explicação técnica central
  (congelar pesos vs treinar matrizes de baixo posto).
- **LoRA+RAG**: "Fine-tuning completo usa todos os parâmetros do modelo, enquanto
  LoRA usa apenas os pesos mais recentes — reduzindo o custo de treino e o tempo de
  treino." — mais próximo do esperado, embora ainda não cite "baixo posto"/matrizes.

Pelo critério de palavras-chave, nenhuma das três respostas foi contada como acerto
nessa pergunta: a melhora aparece na qualidade, ainda não na pontuação.

## Limitações e próximos passos

- Dataset de treino pequeno (79 exemplos) — mais exemplos, especialmente de recusa,
  tenderiam a melhorar a abstenção mesmo sem RAG.
- Pontuação por palavra-chave é rígida (subestima respostas corretas com sinônimos) —
  um LLM-as-judge calibrado contra uma amostra avaliada por humanos daria uma leitura
  mais fina.
- O gabarito é pequeno (24 perguntas) — resultado indicativo, não um benchmark
  estatisticamente robusto.
