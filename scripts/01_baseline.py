"""
Mede o modelo base intocado ANTES de qualquer treino: gerações de referência (greedy,
reprodutível) e perplexidade em held-out dentro/fora do domínio. Sem isso, nenhum
número medido depois do fine-tuning significa nada.

    python scripts/01_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import utils
from textos import HELDOUT_DOMINIO, HELDOUT_GERAL, PROMPTS_TESTE

utils.fixar_seed()

with utils.medir("carregar modelo base"):
    modelo, tok = utils.carregar(config.MODELO_CHAT)

print("\n--- Gerações de referência (modelo base, temperatura=0) ---")
geracoes = {}
for prompt in PROMPTS_TESTE:
    resposta = utils.chat(modelo, tok, [{"role": "user", "content": prompt}], max_new_tokens=150)
    geracoes[prompt] = resposta
    print(f"\n> {prompt}\n{resposta}")

print("\n--- Perplexidade held-out ---")
ppl_dominio = utils.perplexidade(modelo, tok, HELDOUT_DOMINIO)
ppl_geral = utils.perplexidade(modelo, tok, HELDOUT_GERAL)
print(f"Perplexidade (domínio IA Generativa, nunca visto): {ppl_dominio:.2f}")
print(f"Perplexidade (geral, fora do domínio):              {ppl_geral:.2f}")

utils.salvar_resultado("baseline", {
    "modelo": config.MODELO_CHAT,
    "geracoes": geracoes,
    "perplexidade": {"dominio": ppl_dominio, "geral": ppl_geral},
    "parametros": utils.contar_parametros(modelo),
})

utils.liberar(modelo)
