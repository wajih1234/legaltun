

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


def generate_answer(question, chunks):
    """
    Prend une question et les chunks retournés par le retriever
    et génère une réponse finale en français

    - question : la question de l'utilisateur
    - chunks   : liste de dicts {text, source, score}
    """

    # Etape 1 : construire le contexte depuis les chunks
    if not chunks:
        return "Aucune information trouvée dans les documents."

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source: {chunk['source']} | Score: {chunk['score']}]\n"
        context += chunk['text'] + "\n\n"

    # Etape 2 : construire le prompt
    system_prompt = """Tu es un assistant juridique tunisien.
Tu réponds aux questions en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Réponds toujours en français, de manière concise et précise."""

    user_prompt = f"""Contexte extrait du Journal Officiel de la République Tunisienne:

{context}

Question: {question}

Réponds en te basant uniquement sur le contexte ci-dessus."""

    # Etape 3 : appel Groq
    print(f"[generator] Génération de la réponse...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    print(f"[generator] Réponse générée ")
    return answer


# Test
if __name__ == "__main__":
    from retriever import retrieve

    question = "Qui a été nommé membre représentant le ministère des affaires culturelles au conseil d'administration de l'établissement de la Télévision tunisienne ?"

    print(f"Question: {question}\n")

    # Etape 1: retriever
    chunks = retrieve(question)

    # Etape 2: generator
    answer = generate_answer(question, chunks)

    print(f"\n=== REPONSE FINALE ===")
    print(answer)