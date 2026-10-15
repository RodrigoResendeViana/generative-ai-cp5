"""
Textos de avaliação, deliberadamente separados do dataset de treino (data/sft_dataset.jsonl).

HELDOUT_DOMINIO: frases sobre IA Generativa/LLMs que o modelo NUNCA viu no treino —
mede o que ele realmente aprendeu sobre o domínio.
HELDOUT_GERAL: frases comuns em português, fora do domínio de IA —
mede o que o fine-tuning fez o modelo esquecer (esquecimento catastrófico).
PROMPTS_TESTE: perguntas usadas para comparação qualitativa (antes/depois do treino).
"""

HELDOUT_DOMINIO = [
    "A atenção multi-cabeça permite que o modelo pondere diferentes partes da entrada simultaneamente.",
    "Um adaptador LoRA congela os pesos originais e treina apenas duas matrizes de baixo posto.",
    "A quantização em 4 bits reduz drasticamente o uso de memória para carregar um modelo grande.",
    "Embeddings multilíngues permitem comparar frases em idiomas diferentes pelo significado.",
    "O FAISS indexa vetores para permitir busca por similaridade em grande escala.",
    "Um autoencoder aprende a comprimir e reconstruir dados através de uma representação latente.",
    "A perplexidade mede o quão bem um modelo de linguagem prevê um texto, quanto menor melhor.",
    "O gradient checkpointing troca tempo de computação por economia de memória durante o treino.",
]

HELDOUT_GERAL = [
    "O Brasil é o maior país da América do Sul em território e população.",
    "A fotossíntese é o processo pelo qual as plantas convertem luz solar em energia química.",
    "Uma receita de pão simples leva farinha, água, fermento e sal.",
    "O sistema solar tem oito planetas orbitando o Sol.",
    "A Copa do Mundo de futebol acontece a cada quatro anos.",
    "Rios formam vales ao longo de milhares de anos de erosão.",
    "A biblioteca da cidade empresta livros gratuitamente para os moradores.",
    "O ciclo da água inclui evaporação, condensação e precipitação.",
]

PROMPTS_TESTE = [
    "O que é fine-tuning de um modelo de linguagem?",
    "Qual a diferença entre LoRA e QLoRA?",
    "Para que serve o RAG (Retrieval-Augmented Generation)?",
    "Por que uma GPU é importante para treinar uma LLM localmente?",
    "O que é esquecimento catastrófico?",
]
