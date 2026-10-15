import { useEffect, useRef, useState } from 'react'
import { enviarMensagem, verificarSaude } from './api.js'
import './index.css'

export default function App() {
  const [mensagens, setMensagens] = useState([])
  const [texto, setTexto] = useState('')
  const [enviando, setEnviando] = useState(false)
  const [useFinetuned, setUseFinetuned] = useState(true)
  const [useRag, setUseRag] = useState(false)
  const [saude, setSaude] = useState(null)
  const [erro, setErro] = useState(null)
  const fimRef = useRef(null)

  useEffect(() => {
    verificarSaude().then(setSaude).catch(() => setErro('Backend indisponível em http://localhost:8000'))
  }, [])

  useEffect(() => {
    fimRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [mensagens, enviando])

  async function handleEnviar(e) {
    e.preventDefault()
    const pergunta = texto.trim()
    if (!pergunta || enviando) return

    setMensagens((m) => [...m, { autor: 'usuario', texto: pergunta }])
    setTexto('')
    setEnviando(true)
    setErro(null)

    try {
      const resp = await enviarMensagem(pergunta, useFinetuned, useRag)
      setMensagens((m) => [
        ...m,
        {
          autor: 'assistente',
          texto: resp.response,
          latencia: resp.latency_ms,
          fontes: resp.sources,
          modo: useFinetuned ? 'fine-tunado' : 'base',
        },
      ])
    } catch (err) {
      setErro(err.message)
    } finally {
      setEnviando(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Tutor de IA Generativa</h1>
        <p className="subtitulo">CP5 · SmolLM2-1.7B-Instruct + LoRA {saude?.rag_available ? '+ RAG' : ''}</p>
        {saude && (
          <div className="badges">
            <span className={`badge ${saude.adapter_available ? 'ok' : 'off'}`}>
              {saude.adapter_available ? 'LoRA carregado' : 'LoRA não treinado ainda'}
            </span>
            <span className={`badge ${saude.rag_available ? 'ok' : 'off'}`}>
              {saude.rag_available ? 'RAG disponível' : 'RAG indisponível'}
            </span>
          </div>
        )}
      </header>

      <div className="controles">
        <label className={!saude?.adapter_available ? 'desativado' : ''}>
          <input
            type="checkbox"
            checked={useFinetuned}
            disabled={!saude?.adapter_available}
            onChange={(e) => setUseFinetuned(e.target.checked)}
          />
          Usar modelo fine-tunado (desmarque para comparar com o modelo base)
        </label>
        <label className={!saude?.rag_available ? 'desativado' : ''}>
          <input
            type="checkbox"
            checked={useRag}
            disabled={!saude?.rag_available}
            onChange={(e) => setUseRag(e.target.checked)}
          />
          Usar RAG (buscar nos PDFs da disciplina)
        </label>
      </div>

      <main className="chat">
        {mensagens.length === 0 && (
          <p className="vazio">Pergunte algo sobre IA Generativa — ex.: "O que é LoRA?" ou "Para que serve o RAG?"</p>
        )}
        {mensagens.map((m, i) => (
          <div key={i} className={`mensagem ${m.autor}`}>
            <div className="bolha">
              <p>{m.texto}</p>
              {m.autor === 'assistente' && (
                <div className="meta">
                  <span>{m.modo === 'fine-tunado' ? 'fine-tunado' : 'base'}</span>
                  <span>{m.latencia?.toFixed(0)} ms</span>
                  {m.fontes?.length > 0 && <span>fontes: {m.fontes.join(', ')}</span>}
                </div>
              )}
            </div>
          </div>
        ))}
        {enviando && <div className="mensagem assistente"><div className="bolha">digitando…</div></div>}
        <div ref={fimRef} />
      </main>

      {erro && <p className="erro">{erro}</p>}

      <form className="form-entrada" onSubmit={handleEnviar}>
        <input
          type="text"
          placeholder="Digite sua pergunta..."
          value={texto}
          onChange={(e) => setTexto(e.target.value)}
          disabled={enviando}
        />
        <button type="submit" disabled={enviando || !texto.trim()}>Enviar</button>
      </form>
    </div>
  )
}
