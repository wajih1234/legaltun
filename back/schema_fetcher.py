# schema_fetcher.py

from neo4j_client import driver


def get_schema():
    with driver.session() as session:

        # step 1: get all node labels
        labels_result = session.run("CALL db.labels()")
        labels = [row["label"] for row in labels_result.data()]

        # step 2: get all relationship types
        rels_result = session.run("CALL db.relationshipTypes()")
        relationships = [row["relationshipType"] for row in rels_result.data()]

        # step 3: get example complete triplets
        triplets_result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN a.name AS entity1_name,
                   labels(a)[0] AS entity1_type,
                   type(r) AS relation,
                   b.name AS entity2_name,
                   labels(b)[0] AS entity2_type
            LIMIT 8
        """)
        triplets = triplets_result.data()

    return {
        "labels": labels,
        "relationships": relationships,
        "triplets": triplets
    }


def format_schema_for_prompt(schema):
    labels_str = ", ".join(schema["labels"][:24])
    rels_str = ", ".join(schema["relationships"][:24])

    lines = []
    lines.append(f"Node labels: {labels_str}")
    lines.append(f"Relationship types: {rels_str}")
    lines.append("Node properties: name, source, year")
    lines.append("")
    lines.append("Example triplets:")
    for t in schema["triplets"]:
        e1 = t['entity1_name'][:50]
        e2 = t['entity2_name'][:50]
        lines.append(f"  ({e1}:{t['entity1_type']}) -[{t['relation']}]-> ({e2}:{t['entity2_type']})")

    return "\n".join(lines)


if __name__ == "__main__":
    schema = get_schema()
    prompt_text = format_schema_for_prompt(schema)
    print(prompt_text)
    print()
    print(f"Total characters: {len(prompt_text)}")