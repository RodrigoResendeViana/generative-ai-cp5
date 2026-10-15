"""
Roda o gabarito (data/gabarito.json) contra três configurações — modelo base,
LoRA, e LoRA+RAG — e gera a tabela/JSON comparativo final (bake-off).

    python scripts/07_evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import avaliacao
import config
import rag
import utils


def main() -> None:
    utils.fixar_seed()
    resultados = {}

    # --- 1) modelo base, sem adaptador ---
    modelo, tok = utils.carregar(config.MODELO_CHAT)

    def responder_base(pergunta: str) -> str:
        return utils.chat(modelo, tok, [{"role": "user", "content": pergunta}], max_new_tokens=200)

    print("Avaliando rota: base...")
    resultados["base"] = avaliacao.avaliar_rota("base", responder_base)
    utils.liberar(modelo)

    if not config.ADAPTADOR_DIR.exists():
        print(f"Nenhum adaptador em {config.ADAPTADOR_DIR} — rode scripts/05_hyperparam_sweep.py antes.")
    else:
        # --- 2) LoRA ---
        modelo, tok = utils.carregar(config.MODELO_CHAT, adaptador=config.ADAPTADOR_DIR)

        def responder_lora(pergunta: str) -> str:
            return utils.chat(modelo, tok, [{"role": "user", "content": pergunta}], max_new_tokens=200)

        print("Avaliando rota: lora...")
        resultados["lora"] = avaliacao.avaliar_rota("lora", responder_lora)

        # --- 3) LoRA + RAG ---
        if rag.indice_disponivel():
            indice, chunks = rag.carregar_indice()

            def responder_lora_rag(pergunta: str) -> str:
                trechos = rag.buscar(pergunta, indice, chunks)
                prompt_rag = rag.montar_prompt(pergunta, trechos)
                return utils.chat(modelo, tok, [{"role": "user", "content": prompt_rag}], max_new_tokens=200)

            print("Avaliando rota: lora+rag...")
            resultados["lora_rag"] = avaliacao.avaliar_rota("lora+rag", responder_lora_rag)
        else:
            print("Índice RAG não encontrado — pulando a rota lora+rag (rode scripts/02 e 06 antes).")

        utils.liberar(modelo)

    print("\n--- Bake-off ---")
    print(f"{'rota':<10} {'acerto_total':>13} {'abst_correta':>14} {'abst_indevida':>15}")
    for nome, r in resultados.items():
        ac = r["acerto_total"]
        abc = r["abstencao_correta"] or 0
        abi = r["abstencao_indevida"] or 0
        print(f"{nome:<10} {ac:>13.0%} {abc:>14.0%} {abi:>15.0%}")

    utils.salvar_resultado("bakeoff", {"rotas": resultados})


if __name__ == "__main__":
    main()
