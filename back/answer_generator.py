# answer_generator.py

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question, cypher, results):

    # convertir les résultats en JSON lisible
    # ensure_ascii=False garde les caractères arabes et français lisibles
    results_str = json.dumps(results, ensure_ascii=False, indent=2)

    if not results:
        results_str = "Aucun résultat trouvé."

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Tu es un assistant juridique spécialisé dans le droit tunisien.
Tu réponds aux questions en te basant sur les données extraites d'un graphe de connaissances juridiques.
Sois concis (2-3 phrases).
Réponds dans la même langue que l'utilisateur ( français ou anglais)et prends en considération la source dans les results pour donne une réponse correcte.
Si les résultats sont vides, dis qu'aucune information n'a été trouvée."""
            },
            {
                "role": "user",
                "content": f"""Question de l'utilisateur: {question}

Requête Cypher exécutée: {cypher}

Résultats du graphe:
{results_str}

Rédige une réponse claire et directe à la question de l'utilisateur."""
            }
        ],
        max_tokens=600,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # test avec de vraies données
    question = "Qui a été nommé  ingénieur ?"
    cypher = "MATCH (a)-[r]->(b) RETURN a.name AS entity1, type(r) AS relation, b.name AS entity2 LIMIT 3"
    results = [
        {"entity1": "Mohamed ali", "relation": "nommer", "entity2": "Cheffe du Gouvernement"},
        {"entity1": "yassine", "relation": "nommer", "entity2": "directeur général"},
    ]

    answer = generate_answer(question, cypher, results)
    print("Réponse:")
    print(answer)