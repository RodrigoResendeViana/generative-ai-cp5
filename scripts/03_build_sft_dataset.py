"""
Gera data/sft_dataset.jsonl: pares pergunta/resposta (texto original, escrito pelo grupo)
cobrindo os conceitos centrais de IA Generativa/LLMs ensinados na disciplina.
Formato de chat (prompt/completion com roles) para o SFTTrainer calcular a loss só na resposta.

Nenhum par aqui vaza para o gabarito de avaliação (data/gabarito.json) — datasets
deliberadamente separados, mesma metodologia do material da disciplina.

    python scripts/03_build_sft_dataset.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

SYSTEM_PROMPT = (
    "Você é um tutor especializado em IA Generativa e Modelos de Linguagem (LLMs). "
    "Responda de forma clara e objetiva. Se a pergunta estiver fora desse escopo, "
    "diga que não sabe em vez de inventar uma resposta."
)

# Cada item: (pergunta, resposta). Duas variações de pergunta por conceito quando fizer sentido.
PARES = [
    ("O que é um modelo de linguagem?",
     "É um modelo treinado para prever a probabilidade do próximo token em uma sequência de texto, "
     "aprendendo padrões da linguagem a partir de um grande volume de dados."),
    ("Como um LLM decide qual palavra vem a seguir?",
     "Ele calcula uma distribuição de probabilidade sobre todo o vocabulário e escolhe (ou amostra) "
     "o token mais provável dado o contexto visto até ali."),

    ("O que é a arquitetura Transformer?",
     "É a arquitetura de rede neural, baseada em mecanismos de atenção, que substituiu redes recorrentes "
     "como padrão para processar sequências de texto, permitindo processar todos os tokens em paralelo."),
    ("Por que o Transformer é mais rápido de treinar que uma RNN?",
     "Porque o Transformer processa todos os tokens da sequência em paralelo via atenção, enquanto uma RNN "
     "precisa processar um token de cada vez, em sequência."),

    ("O que é o mecanismo de atenção?",
     "É o mecanismo que pondera a relevância de cada token em relação aos demais na sequência, permitindo "
     "que o modelo combine informação de diferentes posições do contexto ao gerar cada saída."),
    ("Para que serve a atenção multi-cabeça (multi-head attention)?",
     "Permite que o modelo preste atenção a diferentes tipos de relação entre os tokens ao mesmo tempo, "
     "cada 'cabeça' aprendendo um padrão de relação diferente."),

    ("O que é tokenização?",
     "É o processo de quebrar um texto em unidades menores (tokens) — palavras, subpalavras ou caracteres — "
     "que o modelo consegue representar numericamente."),
    ("Por que um texto em português gera mais tokens que o mesmo texto em inglês?",
     "Porque o tokenizador foi treinado majoritariamente sobre texto em inglês, então palavras em português "
     "tendem a ser quebradas em mais subpalavras, custando mais tokens para a mesma frase."),

    ("Qual a diferença entre pré-treino e fine-tuning?",
     "O pré-treino aprende os padrões gerais da linguagem a partir de um volume enorme e diverso de texto, "
     "geralmente do zero. O fine-tuning parte desse modelo já pré-treinado e o especializa em uma tarefa "
     "ou domínio específico, usando um conjunto de dados muito menor."),
    ("Por que não treinamos um LLM do zero para cada tarefa nova?",
     "Porque pré-treinar do zero exige volumes enormes de dados e computação; é muito mais eficiente partir "
     "de um modelo já pré-treinado e especializá-lo via fine-tuning."),

    ("Quais os problemas do fine-tuning completo em uma GPU com pouca memória?",
     "Fine-tuning completo precisa armazenar gradientes e estados do otimizador para todos os parâmetros, "
     "exigindo cerca de 16 bytes por parâmetro — para um modelo de alguns bilhões de parâmetros isso "
     "facilmente excede a VRAM de GPUs domésticas."),
    ("O fine-tuning completo sempre dá um resultado melhor que o LoRA?",
     "Não necessariamente. Fine-tuning completo é mais caro em memória e mais propenso a esquecimento "
     "catastrófico; o LoRA costuma preservar melhor as capacidades originais do modelo mantendo boa "
     "qualidade na nova tarefa."),

    ("O que é um adaptador LoRA?",
     "LoRA (Low-Rank Adaptation) congela os pesos originais do modelo e treina apenas duas matrizes pequenas "
     "de baixo posto que são somadas à saída das camadas originais, reduzindo drasticamente o número de "
     "parâmetros treináveis."),
    ("Quantos dos parâmetros de um modelo o LoRA costuma treinar?",
     "Tipicamente bem menos de 1% dos parâmetros totais do modelo, dependendo do rank escolhido para o "
     "adaptador."),

    ("No LoRA, o que o parâmetro 'r' (rank) controla?",
     "'r' define o posto das matrizes do adaptador: quanto maior o rank, mais capacidade (e mais parâmetros "
     "treináveis) o adaptador tem."),
    ("E o que o 'alpha' do LoRA controla?",
     "'alpha' escala a contribuição do adaptador sobre a saída original da camada — é um fator de "
     "intensidade de quanto o adaptador influencia o comportamento final do modelo."),

    ("O que é QLoRA?",
     "QLoRA combina quantização do modelo base em 4 bits com adaptadores LoRA, permitindo treinar modelos "
     "maiores em GPUs com pouca VRAM, já que o modelo base ocupa uma fração da memória que ocuparia em "
     "precisão completa."),
    ("Por que usar QLoRA em vez de LoRA comum?",
     "Quando a VRAM disponível é muito limitada: o modelo base em 4 bits ocupa bem menos memória que em "
     "bf16, deixando mais espaço para ativações e para o próprio adaptador."),

    ("O que é esquecimento catastrófico?",
     "É quando o modelo, ao ser treinado intensamente em um novo domínio, perde capacidades que tinha antes "
     "do treino, como responder bem em outros idiomas ou assuntos não relacionados ao novo domínio."),
    ("Como medir se um modelo sofreu esquecimento catastrófico?",
     "Comparando a perplexidade (ou a qualidade das respostas) em um conjunto held-out FORA do domínio de "
     "treino, antes e depois do fine-tuning — se piorar bastante, houve esquecimento."),

    ("O que é um conjunto held-out e por que ele importa na avaliação?",
     "É um conjunto de exemplos que não foi usado no treino, usado só para avaliação. Sem ele, não dá para "
     "saber se o modelo generalizou ou só decorou os exemplos de treino."),
    ("Por que avaliar com frases quase idênticas às de treino é um problema?",
     "Porque a melhora medida pode refletir só memorização daquelas frases específicas, não aprendizado "
     "real do domínio — por isso o held-out deve ser claramente diferente do que foi treinado."),

    ("Por que fixar uma seed (semente) ao comparar experimentos de treino?",
     "Porque sem uma seed fixa, duas execuções do mesmo experimento podem dar resultados diferentes por "
     "causa da aleatoriedade na inicialização e no embaralhamento dos dados, tornando as comparações "
     "entre configurações inválidas."),
    ("O que a seed controla, na prática?",
     "Controla as fontes de aleatoriedade do treino: inicialização de pesos novos, ordem de embaralhamento "
     "do dataset e, quando há amostragem na geração, a escolha dos tokens."),

    ("O que é perplexidade?",
     "É uma métrica que mede o quão bem um modelo de linguagem prevê um texto; quanto menor o valor, "
     "melhor o modelo está prevendo aquela sequência."),
    ("Perplexidade baixa garante que o modelo está certo sobre os fatos?",
     "Não. Perplexidade mede só o quão previsível o texto é para o modelo, não se o conteúdo é factualmente "
     "correto — por isso é preciso combinar com avaliação de acerto factual."),

    ("O que é RAG (Retrieval-Augmented Generation)?",
     "É uma técnica que busca os trechos mais relevantes de uma base de documentos e os injeta no prompt "
     "antes da geração, permitindo que o modelo responda com base em informação específica sem precisar "
     "ser retreinado."),
    ("Quando o RAG é mais indicado que o fine-tuning?",
     "Quando a informação muda com frequência ou precisa ser rastreável até a fonte — RAG busca o conteúdo "
     "atualizado em tempo de resposta, enquanto fine-tuning fixaria conhecimento desatualizável nos pesos."),

    ("O que são embeddings?",
     "São vetores numéricos que representam o significado de um texto, permitindo comparar textos por "
     "similaridade (por exemplo, distância de cosseno) em vez de comparação literal de palavras."),
    ("Por que um sistema de RAG costuma usar um modelo de embeddings diferente do modelo de geração?",
     "Porque modelos de geração são otimizados para produzir texto token a token, não para produzir um bom "
     "vetor de sentença; um modelo de embeddings é treinado especificamente para que a similaridade entre "
     "vetores reflita similaridade de significado."),

    ("Por que documentos são divididos em chunks antes de indexar para RAG?",
     "Porque o modelo tem uma janela de contexto limitada e um documento inteiro pode não caber nela; "
     "chunks menores e focados também tendem a gerar buscas mais precisas para uma pergunta específica."),
    ("O que é sobreposição (overlap) entre chunks e por que usá-la?",
     "É uma faixa de texto repetida entre chunks vizinhos, usada para que uma informação que fique na "
     "fronteira entre dois chunks não seja cortada ao meio e perdida na busca."),

    ("O que é o FAISS e para que serve em um sistema de RAG?",
     "FAISS é uma biblioteca para indexar vetores e buscar, de forma eficiente, os mais similares a um "
     "vetor de consulta — é o que permite encontrar rapidamente os chunks mais relevantes para uma pergunta."),
    ("O que significa buscar os 'top-k' chunks mais relevantes?",
     "Significa recuperar os k trechos do índice cuja similaridade com a pergunta é mais alta, para montar "
     "o contexto que será enviado ao modelo gerar a resposta."),

    ("O que é a janela de contexto de um modelo?",
     "É o número máximo de tokens (prompt mais resposta) que o modelo consegue processar de uma vez; "
     "além desse limite, o texto precisa ser cortado ou resumido."),
    ("Por que simplesmente colar um documento inteiro no prompt nem sempre funciona?",
     "Porque o documento pode exceder a janela de contexto do modelo, e mesmo quando cabe, o tempo e o "
     "custo de processar um contexto muito grande crescem rapidamente."),

    ("O que é gradient checkpointing?",
     "É uma técnica que economiza memória não guardando todas as ativações intermediárias durante o "
     "forward, recalculando-as durante o backward — trocando tempo extra de computação por menor uso de VRAM."),
    ("Quando vale a pena ativar gradient checkpointing?",
     "Quando o treino está estourando a memória da GPU (erro de 'CUDA out of memory') e reduzir o batch "
     "não é suficiente."),

    ("O que é gradient accumulation (acúmulo de gradiente)?",
     "É acumular os gradientes de vários mini-batches pequenos antes de atualizar os pesos, simulando um "
     "batch efetivo maior sem precisar caber tudo na memória de uma vez."),
    ("Por que usar batch pequeno com gradient accumulation em vez de um batch grande direto?",
     "Porque um batch grande pode não caber na VRAM disponível; acumular gradientes de vários batches "
     "pequenos permite atingir o mesmo batch efetivo sem estourar a memória."),

    ("O que é a taxa de aprendizado (learning rate) no fine-tuning?",
     "É o tamanho do passo dado na direção do gradiente a cada atualização dos pesos; valores muito altos "
     "desestabilizam o treino, valores muito baixos deixam o aprendizado lento demais."),
    ("Por que o LoRA costuma usar uma learning rate mais alta que o fine-tuning completo?",
     "Porque o adaptador LoRA tem muito menos parâmetros e parte de zero (ou de uma inicialização neutra), "
     "precisando de passos maiores para aprender na mesma quantidade de passos de treino."),

    ("O que é early stopping?",
     "É parar o treino automaticamente quando a métrica de validação para de melhorar por um número de "
     "avaliações seguidas, evitando overfitting e desperdício de tempo de treino."),
    ("Para que serve dividir o dataset em treino e validação?",
     "Para medir, durante o treino, se o modelo está generalizando (melhorando na validação) ou só "
     "decorando os exemplos de treino (overfitting)."),

    ("Qual a diferença entre driver da GPU e CUDA Toolkit?",
     "O driver é o software que permite ao sistema operacional e aos programas se comunicarem com a GPU; "
     "o CUDA Toolkit é o conjunto de bibliotecas de computação sobre CUDA. O wheel do PyTorch já traz o "
     "runtime CUDA necessário, então normalmente não é preciso instalar o Toolkit separado."),
    ("O que o campo 'CUDA Version' no topo do nvidia-smi representa?",
     "É a versão máxima de CUDA que o driver instalado suporta — não é necessariamente a versão de CUDA "
     "realmente usada por um programa específico."),

    ("Por que instalar o PyTorch direto do PyPI, sem especificar o índice CUDA, é um erro comum?",
     "Porque o pacote padrão do PyPI é a build para CPU; sem apontar o índice CUDA correto, o PyTorch é "
     "instalado sem suporte a GPU e o treino roda, muito mais lentamente, no processador."),
    ("Como confirmar que o PyTorch instalado tem suporte a GPU?",
     "Checando se 'pip show torch' mostra uma versão terminada em algo como '+cu128' (não '+cpu'), e se "
     "'torch.cuda.is_available()' retorna True."),

    ("O que significa treinar em bf16 em vez de float32?",
     "bf16 usa metade dos bits de float32 para representar cada número, reduzindo o uso de memória e "
     "acelerando o treino em GPUs modernas, com perda de precisão numérica geralmente aceitável para "
     "treino de redes neurais."),
    ("Por que a quantização em 4 bits economiza tanta memória?",
     "Porque cada peso do modelo passa a ser representado com apenas 4 bits em vez dos 16 ou 32 bits "
     "usuais, reduzindo o tamanho do modelo na memória em vários múltiplos."),

    ("O que é DAPT (pré-treino continuado / continued pretraining)?",
     "É continuar o objetivo original de pré-treino (prever o próximo token) sobre texto cru de um domínio "
     "específico, sem precisar de pares pergunta/resposta, para que o modelo absorva vocabulário e estilo "
     "daquele domínio."),
    ("Quando o DAPT é mais indicado que o SFT (fine-tuning de instrução)?",
     "Quando só existe texto cru do domínio, sem pares pergunta/resposta prontos, e o objetivo é que o "
     "modelo absorva terminologia e padrões de um domínio específico."),

    ("O que é SFT (Supervised Fine-Tuning)?",
     "É o fine-tuning de um modelo usando pares de pergunta (ou instrução) e resposta, calculando a loss "
     "apenas sobre a resposta esperada, ensinando o modelo a seguir instruções em um formato específico."),
    ("Por que a loss do SFT é calculada só na resposta, não na pergunta?",
     "Porque o objetivo é ensinar o modelo a gerar boas respostas dado um contexto, não a prever a "
     "pergunta em si — penalizar erros na pergunta não ajudaria o modelo a responder melhor."),

    ("O que é 'function calling' (uso de ferramentas) em um LLM?",
     "É quando o modelo, em vez de responder diretamente, emite uma chamada estruturada a uma função ou "
     "ferramenta externa (como uma busca ou um cálculo), cujo resultado é devolvido a ele para compor a "
     "resposta final."),
    ("Por que function calling em modelos pequenos costuma ser frágil?",
     "Porque modelos pequenos têm menos capacidade para seguir formatos estruturados de forma consistente, "
     "aumentando a chance de gerar uma chamada de função mal formatada."),

    ("O que é alucinação em um LLM?",
     "É quando o modelo gera uma resposta com aparência confiante e coerente, mas factualmente incorreta "
     "ou inventada, sem ter base real para aquela afirmação."),
    ("Como o RAG ajuda a reduzir alucinações?",
     "Fornecendo ao modelo trechos reais de documentos como contexto, de forma que a resposta possa ser "
     "ancorada nesse conteúdo em vez de depender só do que o modelo 'lembra' dos seus pesos."),

    ("O que é um prompt de sistema (system prompt)?",
     "É uma instrução inicial, geralmente invisível ao usuário final, que define o comportamento, o tom ou "
     "as regras que o modelo deve seguir ao longo da conversa."),
    ("Por que dar instruções claras no prompt melhora a qualidade das respostas?",
     "Porque reduz a ambiguidade sobre o que é esperado, orientando o modelo sobre formato, escopo e nível "
     "de detalhe desejado na resposta."),

    ("O que é a temperatura na geração de texto de um LLM?",
     "É um parâmetro que controla o quanto a geração é aleatória: temperatura 0 tende a escolher sempre o "
     "token mais provável (determinístico), temperaturas mais altas introduzem mais variedade e "
     "imprevisibilidade nas respostas."),
    ("Por que usar temperatura 0 ao comparar resultados de experimentos diferentes?",
     "Porque torna a geração determinística (greedy), eliminando a aleatoriedade da amostragem e tornando "
     "as comparações entre configurações mais justas e reprodutíveis."),

    ("O que é um autoencoder?",
     "É uma rede neural que aprende a comprimir uma entrada em uma representação latente menor "
     "(codificador) e depois reconstruí-la a partir dessa representação (decodificador)."),
    ("Para que serve a representação latente aprendida por um autoencoder?",
     "Pode ser usada para redução de dimensionalidade, detecção de anomalias (quando a reconstrução falha) "
     "ou como base para gerar novas amostras parecidas com os dados de treino."),

    ("O que diferencia um modelo generativo de um modelo discriminativo?",
     "Um modelo generativo aprende a gerar novos dados parecidos com os de treino (texto, imagem, áudio); "
     "um modelo discriminativo aprende a distinguir ou classificar entradas em categorias, sem "
     "necessariamente conseguir gerar novos exemplos."),
    ("Um LLM é um modelo generativo? Por quê?",
     "Sim — ele gera novo texto, token a token, a partir de um contexto, em vez de apenas classificar ou "
     "rotular um texto já existente."),

    ("Qual a diferença entre um modelo 'base' e um modelo 'Instruct'?",
     "O modelo base só continua o texto da forma mais provável, sem necessariamente seguir instruções; "
     "o modelo Instruct passou por um fine-tuning adicional para seguir instruções e manter uma conversa "
     "de forma mais útil e controlada."),
    ("Por que um chatbot geralmente usa a versão 'Instruct' do modelo?",
     "Porque essa versão foi ajustada para seguir instruções e manter o formato de conversa, respondendo "
     "de forma mais direta e alinhada ao que o usuário pediu do que o modelo base faria."),

    ("O que é um chat template?",
     "É um formato padronizado que organiza as mensagens de uma conversa (sistema, usuário, assistente) "
     "em um único texto estruturado que o modelo foi treinado para reconhecer e seguir."),
    ("Por que é importante usar o chat template correto do modelo ao montar o prompt?",
     "Porque o modelo foi treinado esperando aquele formato específico; usar um formato diferente pode "
     "confundir o modelo e piorar a qualidade das respostas."),

    ("Como funciona a avaliação de um modelo com um gabarito baseado em palavras-chave?",
     "Cada pergunta do gabarito tem uma lista de palavras ou termos esperados na resposta; a resposta do "
     "modelo é pontuada como correta se contiver essas palavras-chave, de forma simples e comparável entre "
     "diferentes versões do modelo."),
    ("Por que o gabarito de avaliação deve ser escrito antes de rodar os experimentos?",
     "Para evitar o viés de ajustar os critérios de avaliação depois de ver os resultados — isso garante "
     "que a avaliação meça o modelo de forma justa, e não apenas confirme o que já se esperava ver."),

    ("O que são as perguntas do tipo 'armadilha' em um gabarito de avaliação?",
     "São perguntas fora do escopo do assistente (ou fora do que ele poderia saber), cuja resposta correta "
     "é o modelo se abster e dizer que não sabe, em vez de inventar uma resposta confiante."),
    ("Por que é importante medir abstenção correta e abstenção indevida separadamente?",
     "Porque um modelo que nunca se abstém inventa respostas para perguntas fora do escopo, e um modelo "
     "que se abstém demais deixa de responder perguntas que deveria saber — as duas falhas são diferentes "
     "e precisam ser medidas separadamente."),
]

# Pequeno reforço de abstenção: perguntas de cultura geral fora do escopo do tutor.
RECUSAS = [
    "Qual é a capital da França?",
    "Quem escreveu Dom Casmurro?",
    "Qual o resultado do jogo de ontem?",
    "Me dê uma receita de lasanha.",
    "Qual é a previsão do tempo para amanhã?",
]
RESPOSTA_RECUSA = (
    "Não sei — esse assunto está fora do escopo deste tutor, que é especializado em IA Generativa e "
    "Modelos de Linguagem. Posso ajudar com dúvidas sobre transformers, fine-tuning, LoRA, RAG e temas "
    "relacionados."
)


def main() -> None:
    exemplos = []
    for pergunta, resposta in PARES:
        exemplos.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pergunta},
            ],
            "completion": [{"role": "assistant", "content": resposta}],
        })
    for pergunta in RECUSAS:
        exemplos.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pergunta},
            ],
            "completion": [{"role": "assistant", "content": RESPOSTA_RECUSA}],
        })

    with config.SFT_DATASET.open("w", encoding="utf-8") as f:
        for ex in exemplos:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"{len(exemplos)} exemplos gravados em {config.SFT_DATASET}")
    print(f"  - {len(PARES)} pares de conceito (2 variações cada)")
    print(f"  - {len(RECUSAS)} exemplos de recusa/abstenção")


if __name__ == "__main__":
    main()
