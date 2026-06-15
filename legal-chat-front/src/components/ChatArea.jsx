import { useState, useRef, useEffect } from 'react'
import { askQuestion } from '../services/api'
import MessageBubble from './MessageBubble'
import InputBar from './InputBar'

import '../styles/ChatArea.css'

function ChatArea({ conversation, onUpdate, onCreate }) {
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages])

  const handleSend = async (text) => {
    if (!text.trim() || loading) return

    let convId = conversation?.id
    let messages = conversation?.messages || []

    if (!convId) {
      convId = onCreate()
      messages = []
    }

    const userMessage = { role: 'user', text }
    const updated = [...messages, userMessage]
    onUpdate(convId, updated)

    setLoading(true)

    try {
      const data = await askQuestion(text)
      const botMessage = {
        role: 'bot',
        text: data.answer,
        sources: data.results || []
      }
      const final = [...updated, botMessage]
      onUpdate(convId, final)
    } catch (err) {
      const errorMessage = {
        role: 'bot',
        text: 'Une erreur est survenue. Veuillez réessayer.',
        sources: []
      }
      onUpdate(convId, [...updated, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-area">
      <div className="chat-topbar">
        <span className="chat-title">
          {conversation ? conversation.title : 'LexTN'}
        </span>
        <span className="chat-badge">Graphe Neo4j</span>
      </div>

      <div className="messages">
        {!conversation || conversation.messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">⚖️</div>
            <p className="empty-title">Posez votre question juridique</p>
            <p className="empty-sub">Basé sur le Journal Officiel de la République Tunisienne</p>
          </div>
        ) : (
          conversation.messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))
        )}
        {loading && (
          <div className="loading-row">
            <div className="loading-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <InputBar onSend={handleSend} loading={loading} />

    </div>
  )
}

export default ChatArea