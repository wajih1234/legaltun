import { useConversations } from './hooks/useConversations'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import { useState, useEffect } from 'react'
import './styles/global.css'

function App() {
  const [theme, setTheme] = useState('light')
  const {
    conversations,
    activeId,
    activeConversation,
    setActiveId,
    createConversation,
    updateConversation,
    deleteConversation
  } = useConversations()

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])
  useEffect(() => {
    if (conversations.length === 0) {
      createConversation()
    }
  }, [conversations.length])
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onCreate={createConversation}
        onDelete={deleteConversation}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <ChatArea
        conversation={activeConversation}
        onUpdate={updateConversation}
        onCreate={createConversation}
      />
    </div>
  )
}

export default App