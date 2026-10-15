# Tutor de IA Generativa: Fine-Tuning Local de uma LLM com LoRA e RAG

**Disciplina:** Generative AI for Engineering — CP5  
**Equipe:**
- Rodrigo Viana — RM 551057
- Caique Chagas — RM 551943
- Eduardo Fedeli — RM 550132
- Otavio Vitoriano — RM 552012
- Gabriel Torres — RM 98600

**Data:** 10/10/2026

---

## Resumo

Este trabalho treina localmente, em GPU, um adaptador LoRA para o modelo
`SmolLM2-1.7B-Instruct`, especializando-o como tutor de conceitos de IA
Generativa/LLMs. Quatro configurações de hiperparâmetros foram comparadas, e o
modelo resultante foi avaliado contra um gabarito de 24 perguntas em três
configurações — modelo base, modelo com LoRA, e modelo com LoRA combinado a um
sistema de RAG sobre os PDFs da disciplina. A combinação LoRA+RAG elevou o acerto de
33% (base) para 54%, e foi a única capaz de fazer o modelo se abster corretamente de
perguntas fora de escopo (50%, contra 0% nas outras duas configurações). O modelo é
servido por uma API em FastAPI e consumido por um frontend em React.

---

## 1. Introdução

Modelos de linguagem pré-treinados em larga escala respondem bem a perguntas gerais,
mas não são especializados em um domínio específico nem sabem reconhecer os limites
do próprio conhecimento. O fine-tuning por LoRA (Low-Rank Adaptation) permite
especializar um modelo desses sem re-treinar todos os seus parâmetros, tornando o
processo viável em hardware de GPU doméstico. Este trabalho investiga até que ponto
essa especialização, sozinha, é suficiente — e o que ainda falta para obter respostas
confiáveis — usando como estudo de caso um tutor sobre a própria disciplina de IA
Generativa.

## 2. Objetivos

- Treinar localmente, em GPU, um adaptador LoRA sobre uma LLM de código aberto.
- Definir um problema específico e mensurável para o fine-tuning resolver.
- Comparar diferentes hiperparâmetros de treino e justificar a escolha final.
- Avaliar quantitativamente o ganho do fine-tuning sobre o modelo base.
- Expor o modelo por um frontend utilizável por um usuário final.

## 3. Seleção do Modelo

O modelo escolhido foi o `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Três fatores
motivaram essa escolha: (i) com 1,7 bilhão de parâmetros, cabe com folga em uma GPU
de 8 GB de VRAM quando treinado via LoRA (~6,5 GB de pico observado, contra ~25,5 GB
que um fine-tuning completo exigiria); (ii) já é distribuído com *chat template*,
dispensando trabalho adicional para funcionar como assistente conversacional; (iii)
sua qualidade de resposta é sensivelmente superior às variantes menores da mesma
família (135M e 360M), o que é necessário para um chatbot demonstrável.

**Conclusão:** o tamanho do modelo foi dimensionado pelo hardware disponível
(GPU de 8 GB), não pelo maior modelo possível.

## 4. Definição do Problema

O problema escolhido foi restringir o modelo a um **tutor de IA Generativa/LLMs**:
transformers, atenção, tokenização, pré-treino vs. fine-tuning, LoRA/QLoRA,
esquecimento catastrófico, RAG, embeddings e infraestrutura de GPU/CUDA. Temas de
Machine Learning clássico também presentes na disciplina (regressão, PCA, XGBoost)
foram deliberadamente excluídos do escopo, e perguntas fora do domínio (cultura
geral, dados em tempo real) devem ser recusadas pelo modelo em vez de respondidas.

**Conclusão:** delimitar o escopo negativamente (o que o modelo *não* deve tentar
responder) foi tão importante quanto defini-lo positivamente — é esse limite que
torna a abstenção um critério de avaliação mensurável (seção 8).

## 5. Ambiente de Execução

Todo o treinamento e toda a inferência rodaram em GPU local: NVIDIA GeForce RTX 4070
Laptop (8 GB de VRAM), driver 555.97. A instalação seguiu três cuidados técnicos
necessários nesta configuração:

- PyTorch instalado a partir do índice CUDA do PyTorch (`--index-url
  .../whl/cu128`), nunca do PyPI padrão — o pacote padrão é uma build apenas para
  CPU e não usa a GPU.
- Ambiente isolado em Python 3.12 (não a versão mais recente instalada no sistema),
  por compatibilidade das bibliotecas de ML.
- `bitsandbytes`/QLoRA foram descartados: no Windows nativo seu suporte é instável,
  e o LoRA em bf16 já cabe confortavelmente no orçamento de VRAM da placa.

A verificação de ambiente (`torch.cuda.is_available()`, nome da GPU, suporte a
bf16, e um cálculo real de matriz na GPU) confirmou 27,9 TFLOP/s em bf16 e pico de
3,2 GB de VRAM só para carregar o modelo.

**Conclusão:** não foi necessário instalar o CUDA Toolkit separadamente — o próprio
pacote do PyTorch traz o runtime —, mas a escolha do índice de instalação correto
para a GPU foi o passo onde um erro silencioso (build CPU) seria mais provável.

## 6. Construção do Dataset

Como os materiais da disciplina são propriedade do professor, nenhum texto extraído
deles foi incluído no dataset de treino ou no repositório público. Em vez disso,
foram escritos, em texto original, dois conjuntos:

- **Dataset de treino** (79 exemplos): pares pergunta/resposta cobrindo cerca de 45
  conceitos centrais da disciplina, mais exemplos explícitos de recusa a perguntas
  fora de escopo.
- **Gabarito de avaliação** (24 perguntas): escrito *antes* de qualquer experimento,
  20 perguntas normais com palavras-chave esperadas na resposta e 4 perguntas-
  armadilha (fora de escopo) cuja resposta correta é a abstenção.

Os PDFs reais das aulas foram usados apenas para alimentar um índice de busca (RAG)
local, nunca para treinar o modelo nem para compor o dataset distribuído.

**Conclusão:** separar a fonte protegida (PDFs) do material de treino e de avaliação
obrigou o dataset a ser pequeno — o que, como a seção 8 mostra, também expôs uma
limitação real do fine-tuning isolado.

## 7. Treinamento e Teste de Hiperparâmetros

O treinamento usou LoRA (não fine-tuning completo) via `SFTTrainer`, com
*gradient checkpointing*, precisão bf16, lote efetivo de 8 (via acumulação de
gradiente) e *early stopping*. Antes de qualquer treino, o modelo base foi medido
isoladamente: perplexidade de 46,72 em frases do domínio de IA Generativa (nunca
vistas) contra 21,13 em frases gerais — evidência numérica de que o modelo, de
fábrica, domina menos o vocabulário específico da disciplina do que o português
genérico.

Quatro configurações foram comparadas, variando um eixo de cada vez a partir de um
valor padrão:

| Configuração | Rank | Learning rate | Épocas | Eval loss final | Pico de VRAM |
|---|---|---|---|---|---|
| r16_lr2e-4_ep3 | 16 | 2e-4 | 3 | 1,801 | 3,7 GB |
| r8_lr2e-4_ep3 | 8 | 2e-4 | 3 | 1,877 | 6,5 GB |
| r16_lr1e-4_ep3 | 16 | 1e-4 | 3 | 1,944 | 6,5 GB |
| **r16_lr2e-4_ep5** | **16** | **2e-4** | **5** | **1,728** | 6,5 GB |

**Conclusões:**
- Rank maior (16) superou rank menor (8) nas mesmas condições — mais capacidade no
  adaptador ajudou, sem sinal de overfitting no tamanho de dataset usado.
- Learning rate mais alta (2e-4) convergiu melhor que 1e-4 no número de passos
  disponível — compatível com a prática usual de LoRA usar taxas bem maiores que um
  fine-tuning completo.
- Estender de 3 para 5 épocas continuou reduzindo a perda de validação, sem
  acionar o critério de parada antecipada — a configuração vencedora foi a de maior
  rank, maior learning rate e mais épocas simultaneamente.

## 8. Avaliação do Modelo

O gabarito de 24 perguntas foi aplicado a três configurações do sistema, com a mesma
métrica (acerto por palavra-chave; nas armadilhas, acerto = abstenção):

| Configuração | Acerto total | Abstenção correta (armadilhas) |
|---|---|---|
| Modelo base (sem treino) | 33% | 0% |
| Modelo + LoRA | 42% | 0% |
| Modelo + LoRA + RAG | **54%** | **50%** |

Um exemplo ilustra a progressão: à pergunta "qual a diferença entre fine-tuning
completo e LoRA?", o modelo base responde incoerentemente, tratando LoRA como uma
técnica genérica de aprendizado de máquina ("Layer-wise Relevance Training"); o
modelo com LoRA aborda os termos certos da pergunta (custo de treino, learning rate),
mas inventa o significado da sigla; o modelo com LoRA e RAG chega mais perto, ao
contrastar "todos os parâmetros" com "apenas parte dos pesos", embora ainda não
mencione as matrizes de baixo posto. Pelo critério de palavras-chave, nenhuma das três
respostas foi contada como acerto.

**Conclusões:**
- O fine-tuning isolado melhora o enquadramento e o vocabulário das respostas, mas
  **não ensina o modelo a se abster** — com apenas 5 exemplos de recusa em 79, o
  comportamento não generalizou para perguntas-armadilha inéditas.
- É a camada de RAG — não o fine-tuning — que resolve a abstenção: ao não encontrar
  o conteúdo da pergunta nos trechos recuperados, o modelo passa a reconhecer que
  não sabe a resposta.
- O resultado confirma, empiricamente, a divisão de responsabilidades muitas vezes
  citada na prática de LLMs: **fine-tuning ajusta a forma, e RAG ancora o fato** — e
  a melhor configuração é a combinação das duas técnicas, não uma isolada.

## 9. Aplicação: Backend e Frontend

O modelo é servido por uma API em FastAPI, que carrega a LLM base **uma única vez**
e aplica o adaptador LoRA por cima via PEFT; a comparação entre "modelo base" e
"modelo fine-tunado" é feita ativando/desativando o adaptador na mesma cópia do
modelo em memória (`disable_adapter()`), em vez de carregar duas cópias completas —
decisão necessária para caber no orçamento de VRAM da GPU usada. O frontend, em
React, oferece um campo de prompt, histórico de conversa e dois controles: alternar
entre modelo base e fine-tunado, e ligar/desligar o RAG.

A aplicação também foi projetada para degradar graciosamente: sem um adaptador
treinado, serve apenas o modelo base; sem o índice de RAG local (que depende dos
PDFs, não distribuídos publicamente), serve apenas o LoRA — ambos os casos
sinalizados ao usuário pela própria interface.

**Conclusão:** compartilhar uma única cópia do modelo entre os dois modos de
comparação foi o que tornou viável rodar a aplicação completa (modelo + adaptador +
índice de busca) dentro de 8 GB de VRAM.

## 10. Conclusão

O fine-tuning local via LoRA, mesmo com um dataset pequeno (79 exemplos) e em uma
GPU de consumo, produziu uma melhora mensurável sobre o modelo base (33% → 42% de
acerto). Entretanto, o experimento também demonstrou o limite dessa técnica isolada:
o modelo fine-tunado continuou respondendo com confiança a perguntas fora do seu
escopo. Essa lacuna só foi fechada ao combinar o LoRA com um sistema de RAG sobre o
material real da disciplina, elevando o acerto total a 54% e a taxa de abstenção
correta de 0% para 50%. O resultado reforça que fine-tuning e recuperação de
informação resolvem problemas diferentes e complementares — e que avaliar um modelo
apenas pelo acerto geral, sem medir sua capacidade de reconhecer os próprios limites,
esconde uma falha de confiabilidade relevante para qualquer aplicação real.