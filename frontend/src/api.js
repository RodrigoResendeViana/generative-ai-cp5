const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function enviarMensagem(message, useFinetuned, useRag) {
  const res = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, use_finetuned: useFinetuned, use_rag: useRag }),
  })
  if (!res.ok) throw new Error(`Erro ${res.status} ao consultar a API`)
  return res.json()
}

export async function verificarSaude() {
  const res = await fetch(`${API_URL}/health`)
  if (!res.ok) throw new Error(`Erro ${res.status} ao consultar /health`)
  return res.json()
}
