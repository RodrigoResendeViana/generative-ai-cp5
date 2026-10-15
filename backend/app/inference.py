"""
Carrega o modelo base UMA vez e aplica o adaptador LoRA por cima (PEFT).
Comparar "modelo base" vs "modelo fine-tunado" usa o mesmo modelo em VRAM —
`disable_adapter()` liga/desliga o LoRA sem precisar carregar duas cópias do modelo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

SYSTEM_PROMPT = (
    "Você é um tutor especializado em IA Generativa e Modelos de Linguagem (LLMs). "
    "Responda de forma clara e objetiva. Se a pergunta estiver fora desse escopo, "
    "diga que não sabe em vez de inventar uma resposta."
)

_modelo = None
_tok = None
_tem_adaptador = False


def iniciar() -> None:
    global _modelo, _tok, _tem_adaptador

    _tok = AutoTokenizer.from_pretrained(config.MODELO_CHAT)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODELO_CHAT, dtype=config.DTYPE, attn_implementation=config.ATTN
    ).to(config.DEVICE)

    if config.ADAPTADOR_DIR.exists():
        _modelo = PeftModel.from_pretrained(base, str(config.ADAPTADOR_DIR))
        _tem_adaptador = True
        print(f"Adaptador LoRA carregado de {config.ADAPTADOR_DIR}")
    else:
        _modelo = base
        _tem_adaptador = False
        print(f"Nenhum adaptador em {config.ADAPTADOR_DIR} — servindo só o modelo base.")

    _modelo.eval()


def tem_adaptador() -> bool:
    return _tem_adaptador


def gerar(mensagem: str, use_finetuned: bool = True, contexto_rag: str | None = None, max_new_tokens: int = 300) -> str:
    conteudo_usuario = contexto_rag if contexto_rag else mensagem
    mensagens = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": conteudo_usuario},
    ]
    entrada = _tok.apply_chat_template(
        mensagens, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(config.DEVICE)

    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=_tok.eos_token_id)

    usar_base_pura = _tem_adaptador and not use_finetuned
    with torch.no_grad():
        if usar_base_pura:
            with _modelo.disable_adapter():
                saida = _modelo.generate(**entrada, **kwargs)
        else:
            saida = _modelo.generate(**entrada, **kwargs)

    novos_tokens = saida[0][entrada["input_ids"].shape[1]:]
    return _tok.decode(novos_tokens, skip_special_tokens=True).strip()
