import { useState, useEffect } from 'react'

const STORAGE_KEY = 'legal-chat-conversations'

export function useConversations() {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      setConversations(parsed)
      if (parsed.length > 0) setActiveId(parsed[0].id)
    }
  }, [])

  const save = (updated) => {
    setConversations(updated)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  }

  const createConversation = () => {
    const newConv = {
      id: Date.now().toString(),
      title: 'Nouvelle conversation',
      messages: []
    }
    const updated = [newConv, ...conversations]
    save(updated)
    setActiveId(newConv.id)
    return newConv.id
  }

  const updateConversation = (id, messages) => {
    const updated = conversations.map(c => {
      if (c.id !== id) return c
      const title = messages[0]?.text?.slice(0, 40) || 'Nouvelle conversation'
      return { ...c, messages, title }
    })
    save(updated)
  }

  const deleteConversation = (id) => {
    const updated = conversations.filter(c => c.id !== id)
    save(updated)
    if (activeId === id) {
      setActiveId(updated.length > 0 ? updated[0].id : null)
    }
  }

  const activeConversation = conversations.find(c => c.id === activeId) || null

  return {
    conversations,
    activeId,
    activeConversation,
    setActiveId,
    createConversation,
    updateConversation,
    deleteConversation
  }
}
