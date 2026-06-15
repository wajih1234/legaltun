import { useState } from 'react'
import '../styles/InputBar.css'

function InputBar({ onSend, loading }) {
  const [text, setText] = useState('')

  const handleSubmit = () => {
    if (!text.trim() || loading) return
    onSend(text)
    setText('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="input-area">
      <div className={`input-row ${loading ? 'disabled' : ''}`}>
        <textarea
          className="input-field"
          placeholder="Posez votre question juridique en français…"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={loading}
        />
        <button
          className="send-btn"
          onClick={handleSubmit}
          disabled={loading || !text.trim()}
        >
          {loading ? '...' : '↑'}
        </button>
      </div>
      <p className="input-hint">
        Basé sur les données du Journal Officiel de la République Tunisienne
      </p>
    </div>
  )
}

export default InputBar