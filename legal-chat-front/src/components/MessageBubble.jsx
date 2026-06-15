import '../styles/MessageBubble.css'

function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  const extractSources = (results) => {
    if (!results || results.length === 0) return []
    const sources = new Set()
    results.forEach(r => {
      if (r.source) sources.add(r.source)
    })
    return [...sources]
  }

  const sources = isUser ? [] : extractSources(message.sources)

  return (
    <div className={`msg-row ${isUser ? 'user' : 'bot'}`}>
      {!isUser && (
        <div className="avatar bot">⚖️</div>
      )}
      <div className={`bubble ${isUser ? 'user' : 'bot'}`}>
        <p className="bubble-text">{message.text}</p>
        {sources.length > 0 && (
          <div className="sources">
            {sources.map((src, i) => (
              <span key={i} className="source-tag">
                📄 JORT {src}
              </span>
            ))}
          </div>
        )}
      </div>
      {isUser && (
        <div className="avatar user">👤</div>
      )}
    </div>
  )
}

export default MessageBubble