const BASE_URL = 'http://localhost:5002'

export async function askQuestion(question) {
  const response = await fetch(`${BASE_URL}/api/naive/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question })
  })

  if (!response.ok) {
    throw new Error(`Erreur serveur: ${response.status}`)
  }

  const data = await response.json()
  return data
}