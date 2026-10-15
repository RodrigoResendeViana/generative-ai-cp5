"""
Funções compartilhadas por todos os scripts: carregar modelo/adaptador, gerar,
medir perplexidade, medir tempo/VRAM, fixar semente, liberar memória, salvar resultados.
"""
from __future__ import annotations

import gc
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def fixar_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def carregar(nome: str, adaptador: str | Path | None = None):
    """Carrega modelo + tokenizer. Se `adaptador` for passado, aplica o LoRA via PEFT."""
    tok = AutoTokenizer.from_pretrained(nome)
    modelo = AutoModelForCausalLM.from_pretrained(
        nome,
        dtype=config.DTYPE,
        attn_implementation=config.ATTN,
    ).to(config.DEVICE)

    if adaptador is not None:
        from peft import PeftModel

        modelo = PeftModel.from_pretrained(modelo, str(adaptador))

    modelo.eval()
    return modelo, tok


def chat(modelo, tok, mensagens: list[dict], max_new_tokens: int = 300, temperature: float = 0.0) -> str:
    """Gera uma resposta de chat. temperature=0 -> greedy, reprodutível."""
    entrada = tok.apply_chat_template(
        mensagens, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(config.DEVICE)

    with torch.no_grad():
        saida = modelo.generate(
            **entrada,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )

    novos_tokens = saida[0][entrada["input_ids"].shape[1]:]
    return tok.decode(novos_tokens, skip_special_tokens=True).strip()


def perplexidade(modelo, tok, textos: list[str]) -> float:
    """Perplexidade média, ponderada por número de tokens, sobre uma lista de textos."""
    total_nll = 0.0
    total_tokens = 0
    for texto in textos:
        enc = tok(texto, return_tensors="pt").to(config.DEVICE)
        n_tokens = enc["input_ids"].shape[1]
        if n_tokens < 2:
            continue
        with torch.no_grad():
            saida = modelo(**enc, labels=enc["input_ids"])
        total_nll += saida.loss.item() * (n_tokens - 1)
        total_tokens += n_tokens - 1
    if total_tokens == 0:
        return float("nan")
    return float(np.exp(total_nll / total_tokens))


@contextmanager
def medir(rotulo: str = ""):
    """Context manager: imprime tempo e pico de VRAM usados pelo bloco."""
    if config.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    inicio = time.perf_counter()
    yield
    if config.DEVICE == "cuda":
        torch.cuda.synchronize()
        pico = torch.cuda.max_memory_allocated() / 1024**3
    else:
        pico = float("nan")
    duracao = time.perf_counter() - inicio
    print(f"[{rotulo}] tempo={duracao:.2f}s  pico_VRAM={pico:.2f}GB")


def liberar(*objetos) -> None:
    """Deleta referências e devolve a VRAM de fato (o `del` sozinho não devolve)."""
    for obj in objetos:
        del obj
    gc.collect()
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def contar_parametros(modelo) -> dict:
    total = sum(p.numel() for p in modelo.parameters())
    treinavel = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return {"total": total, "treinavel": treinavel, "pct_treinavel": 100 * treinavel / total}


def salvar_resultado(nome: str, dados: dict) -> Path:
    config.RESULTADOS.mkdir(parents=True, exist_ok=True)
    caminho = config.RESULTADOS / f"{nome}.json"
    dados = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **dados}
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultado salvo em {caminho}")
    return caminho
