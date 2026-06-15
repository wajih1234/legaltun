# entity_extractor.py

import re
from neo4j_client import driver


def guess_entity_labels(question):
    """
    Heuristique simple pour deviner les labels Neo4j
    à partir des mots-clés de la question.
    """
    q = (question or "").lower()
    labels = set()

    person_keywords = ["qui", "personne", "ministre", "président", "nommé", "chargé"]
    startup_keywords = ["startup", "start-up", "entreprise", "société", "compagnie", "company"]
    government_keywords = ["gouvernement", "ministère", "ministere", "etat", "état", "institution"]
    legal_text_keywords = ["loi", "décret", "decret", "arrêté", "arrete", "article"]

    if any(k in q for k in person_keywords):
        labels.add("Personne")
    if any(k in q for k in startup_keywords):
        # Adapte ces labels à ton schéma réel si besoin.
        labels.update(["Startup", "Entreprise"])
    if any(k in q for k in government_keywords):
        # Adapte ces labels à ton schéma réel si besoin.
        labels.update(["Gouvernement", "Institution", "Ministere"])
    if any(k in q for k in legal_text_keywords):
        labels.update(["Loi", "Décret", "Arrêté"])

    return list(labels)


def extract_names_from_question(question):
    """
    Extrait les noms propres de la question en cherchant
    les mots qui commencent par une majuscule (heuristique simple)
    """
    # chercher les mots avec majuscule (noms propres potentiels)
    words = re.findall(r'\b[A-ZÀ-Ü][a-zà-ü]{2,}+(?:\s+[A-ZÀ-Ü][a-zà-ü]+)*\b', question)
    return words


def find_exact_names_in_graph(candidates, labels=None):
    """
    Pour chaque candidat, cherche le nom exact dans Neo4j
    Retourne un dict: {nom_question: nom_exact_graphe}
    """
    if not candidates:
        return {}

    matches = {}
    label_clause = f":{':'.join(labels)}" if labels else ""
    query = f"""
        MATCH (n{label_clause})
        WHERE n.name CONTAINS $candidate
        RETURN n.name AS name
        LIMIT 3
    """

    with driver.session() as session:
        for candidate in candidates:
            # cherche les noms qui contiennent le candidat
            result = session.run(query, candidate=candidate)

            records = result.data()
            if records:
                # prendre le nom le plus court (le plus précis)
                best = min(records, key=lambda x: len(x["name"]))
                matches[candidate] = best["name"]

    return matches


def resolve_entities(question):
    """
    Pipeline complet: question → noms exacts dans le graphe
    """
    candidates = extract_names_from_question(question)
    if not candidates:
        return {}

    labels = guess_entity_labels(question)
    matches = find_exact_names_in_graph(candidates, labels=labels)
    return matches


if __name__ == "__main__":
    question = "Qu'est-ce que Imededdine Kabani a été chargé de faire ?"
    print(f"Question: {question}")
    matches = resolve_entities(question)
    print(f"Entités résolues: {matches}")