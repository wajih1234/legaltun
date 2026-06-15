# llm_cypher.py

import os
import re
import time
from dotenv import load_dotenv
from groq import Groq
from schema_fetcher import get_schema, format_schema_for_prompt

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Simple in-memory cache for schema text to avoid
# fetching/re-formatting schema on every request.
SCHEMA_CACHE_TTL_SECONDS = int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "600"))
_schema_cache = {
    "schema_str": None,
    "expires_at": 0.0
}


def _get_cached_schema_str():
    now = time.time()
    if _schema_cache["schema_str"] and now < _schema_cache["expires_at"]:
        return _schema_cache["schema_str"]

    schema = get_schema()
    schema_str = format_schema_for_prompt(schema)
    _schema_cache["schema_str"] = schema_str
    _schema_cache["expires_at"] = now + SCHEMA_CACHE_TTL_SECONDS
    return schema_str


def _sanitize_model_cypher(raw_text):
    """
    Nettoie la sortie brute du LLM pour garder uniquement
    une requête Cypher exécutable.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    #suprrimer le thninking
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


    # Retire les wrappers markdown éventuels.
    text = re.sub(r"^```(?:cypher)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Retire une paire de guillemets entourant toute la requête.
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
    ):
        text = text[1:-1].strip()

    # Cas fréquent: sortie du style ""MATCH ...", on retire les guillemets
    # uniquement en tout début de chaîne (jamais à l'intérieur de la requête).
    text = re.sub(r'^"+(?=[A-Za-z])', "", text)
    text = re.sub(r"^'+(?=[A-Za-z])", "", text)

    return text.strip()


def generate_cypher(question):

    # etape 1: recuperer le schema depuis le cache (ou Neo4j si expiré)
    schema_str = _get_cached_schema_str()

    # etape 3: construire le prompt systeme
    system_prompt = f"""Tu es un expert en génération de requêtes Cypher pour Neo4j appliqué à un graphe de connaissances juridiques tunisiennes.

SCHÉMA DU GRAPHE:
{schema_str}
ÉTAPE 1 — ANALYSE DE LA QUESTION:
Avant de générer la requête, identifie mentalement:
- Le sujet de la question (qui? quoi?)
- L'action demandée (nommer, charger, superviser, fixer, décider...)
- Les filtres présents (date, nom de personne, nom d'entité)
- La relation Cypher correspondante selon la table ci-dessous


RÈGLES STRICTES:
1.Utilise le schéma du graphe (labels et relations) pour comprendre la structure et le contexte des données.

Appuie-toi sur ce schéma pour interpréter la question de l'utilisateur et générer une requête Cypher valide.



2. Retourne UNIQUEMENT la requête Cypher, sans explication, sans markdown, sans backticks
3. Structure de base: MATCH (n:Label)-[:RELATION]->(m) RETURN n.name AS entity1, m.name AS entity2 LIMIT 10
4. Syntaxe correcte: n.name CONTAINS "mot" — JAMAIS CONTAINS(n.name, "mot")
5. Ne filtre QUE sur ce qui est mentionné dans la question — pas de filtres supplémentaires
6. Si impossible à répondre en Cypher, retourne exactement: INVALID
7. GUILLEMETS:
   - Si la valeur contient une apostrophe → guillemets doubles: n.name CONTAINS "l'habitat"
   - Sinon → guillemets simples: n.name CONTAINS 'Zohra'
   - JAMAIS de guillemets échappés: \'  ou \"
- "le 3 février 2026" → n.source = '3_02_2026'
- "en mars 2026" → n.source CONTAINS '_03_2026'
- "en 2026" → n.year = 2026
- Règle: jour sans zéro + underscore + mois sur 2 chiffres + underscore + année
- La propriété source est sur le nœud source (n), pas sur m
8. Toujours inclure n.source AS source dans le RETURN de chaque requête Cypher générée.
EXEMPLES:

Q: "Qu'est-ce que Besma Loukil a été nommée en 13 Mars 2020?"
R: MATCH (n:Personne)-[:nommer]->(m) WHERE n.name CONTAINS "Besma Loukil" AND n.source="13_03_2020" RETURN n.name AS entity1, m.name AS entity2  LIMIT 5

Q: "Qui a été nommé administrateur représentant l'État dans un conseil d'administration bancaire?"
R: MATCH (n:Personne)-[:nommer]->(m) WHERE m.name CONTAINS "conseil d'administration de la Banque" RETURN n.name AS entity1, m.name AS entity2  LIMIT 5

Q: "Qui a été nommé par le ministre de l'équipement "
R: MATCH (n)-[:nommer]->(m) WHERE n.name CONTAINS "équipement et de l'habitat" RETURN n.name AS entity1, m.name AS entity2
Q:"Que décide le jury ?"
R:MATCH (n)-[:décider]->(m)
WHERE n.name CONTAINS 'jury'
RETURN n.name AS entity1, m.name AS entity2

"""

    # etape 4: Groq model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0,
        

    )

    # etape 5: extraire et retourner la requete Cypher
    raw_cypher = response.choices[0].message.content
    cypher = _sanitize_model_cypher(raw_cypher)
    return cypher


if __name__ == "__main__":
    question = "Faicel amiri est eté chargé quoi en 2026?"
    print(f"Question: {question}")
    print()
    cypher = generate_cypher(question)
    print(f"Requête Cypher générée:")
    print(cypher)