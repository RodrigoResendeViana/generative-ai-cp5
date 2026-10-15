"""
Avalia respostas do modelo contra o gabarito (data/gabarito.json).
Pontuação simples por palavra-chave; nas perguntas-armadilha, acertar é abster-se.
O gabarito foi escrito ANTES de rodar qualquer rota — compara rotas entre si,
não serve como benchmark absoluto.
"""
from __future__ import annotations

import json
import time

import config

MARCADORES_ABSTENCAO = [
    "não sei", "nao sei", "não sabe", "nao sabe", "não tenho essa informação",
    "não tenho informação", "fora do escopo", "não consigo", "não posso ajudar",
    "não é sobre isso", "não está nos trechos", "não consta", "não encontrei",
    "não encontrada", "não há informação", "não foi encontrad",
]


def carregar_gabarito() -> list[dict]:
    return json.loads(config.GABARITO.read_text(encoding="utf-8"))


def absteve(resposta: str) -> bool:
    r = resposta.lower()
    return any(marcador in r for marcador in MARCADORES_ABSTENCAO)


def pontuar_item(item: dict, resposta: str) -> dict:
    r = resposta.lower()
    if item["tipo"] == "armadilha":
        acertou = absteve(resposta)
    else:
        acertou = any(p.lower() in r for p in item["palavras_chave"])
    return {
        "id": item["id"],
        "tipo": item["tipo"],
        "pergunta": item["pergunta"],
        "acertou": acertou,
        "absteve": absteve(resposta),
        "resposta": resposta,
    }


def avaliar_rota(nome_rota: str, funcao_responder) -> dict:
    """funcao_responder(pergunta: str) -> str. Roda o gabarito inteiro e pontua."""
    gabarito = carregar_gabarito()
    itens = []
    inicio = time.perf_counter()
    for item in gabarito:
        resposta = funcao_responder(item["pergunta"])
        itens.append(pontuar_item(item, resposta))
    duracao = time.perf_counter() - inicio

    normais = [i for i in itens if i["tipo"] != "armadilha"]
    armadilhas = [i for i in itens if i["tipo"] == "armadilha"]

    return {
        "rota": nome_rota,
        "acerto_total": sum(i["acertou"] for i in itens) / len(itens),
        "acerto_normais": (sum(i["acertou"] for i in normais) / len(normais)) if normais else None,
        "abstencao_correta": (sum(i["acertou"] for i in armadilhas) / len(armadilhas)) if armadilhas else None,
        "abstencao_indevida": (sum(i["absteve"] for i in normais) / len(normais)) if normais else None,
        "latencia_media_s": duracao / len(itens),
        "itens": itens,
    }
