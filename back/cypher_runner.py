# cypher_runner.py

from neo4j_client import driver
from neo4j.exceptions import CypherSyntaxError

# Mots-clés interdits pour empêcher les écritures/modifications
# quand on veut exécuter uniquement des requêtes en lecture.
DISALLOWED_KEYWORDS = [
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD CSV",
    "CALL",
    "APOC",
    "DBMS",
    "GRANT",
    "DENY",
    "REVOKE"
]


def validate_cypher_read_only(cypher):
    """
    Vérifie qu'une requête Cypher est en lecture seule.
    Retourne un tuple: (is_valid, reason).
    """
    # 1) Validation basique: requête non vide
    if not cypher or not cypher.strip():
        return False, "empty query"

    normalized = cypher.strip().upper()

    # 2) Allow-list de départ: la requête doit commencer
    # par une clause orientée lecture.
    if not (
        normalized.startswith("MATCH")
        or normalized.startswith("WITH")
        or normalized.startswith("RETURN")
        or normalized.startswith("UNWIND")
    ):
        return False, "query must start with MATCH, WITH, RETURN, or UNWIND"

    # 3) Block-list: rejet des opérations d'écriture/admin
    for keyword in DISALLOWED_KEYWORDS:
        if keyword in normalized:
            return False, f"disallowed keyword detected: {keyword}"

    # 4) Requête valide
    return True, ""


def run_cypher(cypher):
    try:
        # Exécute la requête et convertit le résultat Neo4j en liste de dicts.
        with driver.session() as session:
            result = session.run(cypher)
            records = result.data()
        return records

    # Erreur syntaxique Cypher: on log puis on renvoie une liste vide.
    except CypherSyntaxError as e:
        print(f"Cypher syntax error: {e}")
        return []

    # Toute autre erreur Neo4j: on log puis on renvoie une liste vide.
    except Exception as e:
        print(f"Neo4j error: {e}")
        return []
   
    

if __name__ == "__main__":
    # Test local rapide
    cypher = "MATCH (a)-[r]->(b) WHERE a.year=2025 RETURN a.name AS entity1, type(r) AS relation, b.name AS entity2 LIMIT 7"

    print(f"Running: {cypher}")
    print()

    results = run_cypher(cypher)

    print(f"Results ({len(results)} records):")
    for record in results:
        print(record)