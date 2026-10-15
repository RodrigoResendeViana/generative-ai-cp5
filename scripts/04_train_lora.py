"""
Treino LoRA (SFT) do tutor de IA Generativa sobre data/sft_dataset.jsonl.
Pode ser chamado como script (hiperparâmetros padrão de config.py) ou importado por
scripts/05_hyperparam_sweep.py para testar configurações diferentes.

    python scripts/04_train_lora.py --r 16 --lora_alpha 32 --lr 2e-4 --epochs 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

import config
import utils


def treinar(
    r: int = config.LORA_R,
    lora_alpha: int = config.LORA_ALPHA,
    lr: float = config.SFT_LR,
    epochs: int = config.SFT_EPOCHS,
    grad_accum: int = config.SFT_GRAD_ACCUM,
    nome_run: str = "default",
) -> dict:
    """Treina um adaptador LoRA com os hiperparâmetros dados e devolve métricas do run."""
    utils.fixar_seed()

    ds = load_dataset("json", data_files=str(config.SFT_DATASET), split="train")
    ds = ds.train_test_split(test_size=0.2, seed=config.SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.MODELO_CHAT)
    modelo = AutoModelForCausalLM.from_pretrained(
        config.MODELO_CHAT, dtype=config.DTYPE, attn_implementation=config.ATTN
    ).to(config.DEVICE)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, lora_cfg)
    params = utils.contar_parametros(modelo)
    print(f"[{nome_run}] parâmetros treináveis: {params['treinavel']:,} ({params['pct_treinavel']:.3f}%)")

    saida_dir = config.RUNS / "treino" / nome_run
    args = SFTConfig(
        output_dir=str(saida_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=config.USA_BF16,
        max_length=config.SFT_MAX_LENGTH,
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=config.SEED,
        data_seed=config.SEED,
        logging_steps=5,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=modelo,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    with utils.medir(f"treino LoRA [{nome_run}]"):
        resultado_treino = trainer.train()

    metricas_eval = trainer.evaluate()

    adaptador_dir = config.RUNS / "adaptadores" / nome_run
    trainer.model.save_pretrained(str(adaptador_dir))
    print(f"[{nome_run}] adaptador salvo em {adaptador_dir}")

    resumo = {
        "nome_run": nome_run,
        "hiperparametros": {"r": r, "lora_alpha": lora_alpha, "lr": lr, "epochs": epochs, "grad_accum": grad_accum},
        "parametros": params,
        "train_loss_final": resultado_treino.training_loss,
        "eval_loss_final": metricas_eval.get("eval_loss"),
        "adaptador_dir": str(adaptador_dir),
        "historico": trainer.state.log_history,
    }
    utils.salvar_resultado(f"treino_{nome_run}", resumo)

    utils.liberar(modelo, trainer)
    return resumo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=config.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=config.SFT_LR)
    parser.add_argument("--epochs", type=int, default=config.SFT_EPOCHS)
    parser.add_argument("--grad_accum", type=int, default=config.SFT_GRAD_ACCUM)
    parser.add_argument("--nome_run", type=str, default="default")
    args = parser.parse_args()

    treinar(
        r=args.r, lora_alpha=args.lora_alpha, lr=args.lr,
        epochs=args.epochs, grad_accum=args.grad_accum, nome_run=args.nome_run,
    )
