"""
Testa diferentes hiperparâmetros de LoRA (rank, learning rate, épocas) — cobre o
requisito "testar diferentes parâmetros" do trabalho. Grade pequena e documentável
(cada combinação varia UM eixo por vez a partir do default), não o produto cartesiano
completo, para caber no tempo disponível do grupo.

Ao final, copia o adaptador da melhor configuração (menor eval_loss) para
runs/adaptador/ — é esse que o backend serve por padrão.

    python scripts/05_hyperparam_sweep.py
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# "scripts/04_train_lora.py" não pode ser importado como módulo normal (nome começa com dígito)
_spec = importlib.util.spec_from_file_location("treino_lora", Path(__file__).parent / "04_train_lora.py")
_treino_lora = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_treino_lora)
treinar = _treino_lora.treinar

GRADE = [
    {"nome_run": "r16_lr2e-4_ep3", "r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 3},
    {"nome_run": "r8_lr2e-4_ep3", "r": 8, "lora_alpha": 16, "lr": 2e-4, "epochs": 3},
    {"nome_run": "r16_lr1e-4_ep3", "r": 16, "lora_alpha": 32, "lr": 1e-4, "epochs": 3},
    {"nome_run": "r16_lr2e-4_ep5", "r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 5},
]


def main() -> None:
    resultados = []
    for cfg in GRADE:
        print(f"\n=== Rodando {cfg['nome_run']} ===")
        resultados.append(treinar(**cfg))

    nomes = [r["nome_run"] for r in resultados]
    train_loss = [r["train_loss_final"] for r in resultados]
    eval_loss = [r["eval_loss_final"] for r in resultados]

    x = range(len(nomes))
    plt.figure(figsize=(9, 5))
    plt.bar([i - 0.2 for i in x], train_loss, width=0.4, label="train_loss")
    plt.bar([i + 0.2 for i in x], eval_loss, width=0.4, label="eval_loss")
    plt.xticks(list(x), nomes, rotation=20, ha="right")
    plt.ylabel("loss")
    plt.title("Sweep de hiperparâmetros — LoRA SFT (SmolLM2-1.7B-Instruct)")
    plt.legend()
    plt.tight_layout()
    saida_png = config.RUNS / "sweep.png"
    plt.savefig(saida_png, dpi=150)
    print(f"\nGráfico salvo em {saida_png}")

    melhor = min(resultados, key=lambda r: r["eval_loss_final"])
    print(f"Melhor configuração por eval_loss: {melhor['nome_run']} (eval_loss={melhor['eval_loss_final']:.4f})")

    if config.ADAPTADOR_DIR.exists():
        shutil.rmtree(config.ADAPTADOR_DIR)
    shutil.copytree(melhor["adaptador_dir"], config.ADAPTADOR_DIR)
    print(f"Adaptador da melhor configuração copiado para {config.ADAPTADOR_DIR} (usado pelo backend)")

    (config.RESULTADOS / "sweep_resumo.json").write_text(
        json.dumps({
            "resultados": [
                {
                    "nome_run": r["nome_run"],
                    "hiperparametros": r["hiperparametros"],
                    "train_loss_final": r["train_loss_final"],
                    "eval_loss_final": r["eval_loss_final"],
                }
                for r in resultados
            ],
            "melhor": melhor["nome_run"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
