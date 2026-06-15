import '../styles/Sidebar.css'
function Sidebar({ conversations, activeId, onSelect, onCreate, onDelete, theme, onToggleTheme }) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="logo-row">
          <div className="logo-icon">⚖️</div>
          <div>
            <div className="logo-text">LexTN</div>
            <div className="logo-sub">Droit tunisien</div>
          </div>
        </div>
        <button className="theme-toggle" onClick={onToggleTheme}>
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </div>

      <button className="new-chat-btn" onClick={onCreate}>
        + Nouvelle conversation
      </button>

      <div className="history-section">
        {conversations.length === 0 && (
          <p className="no-history">Aucune conversation</p>
        )}
        {conversations.map(conv => (
          <div
            key={conv.id}
            className={`history-item ${conv.id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(conv.id)}
          >
            <span className="history-title">{conv.title}</span>
            <button
              className="delete-btn"
              onClick={e => {
                e.stopPropagation()
                onDelete(conv.id)
              }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

export default Sidebar