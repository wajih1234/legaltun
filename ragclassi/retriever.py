

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Memes parametres que embedder.py
COLLECTION_NAME = "jort_docs"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def get_client():
    """
    Connexion a Qdrant via Docker
    """
    return QdrantClient(host="localhost", port=6333,check_compatibility=False )


def retrieve(question, top_k=7):
    """
    Prend une question en français et retourne
    les chunks les plus pertinents avec leurs scores

    - question : texte de la question
    - top_k    : nombre de chunks a retourner
    """

    # Etape 1 : convertir la question en vecteur
    print(f"[retriever] Question : {question}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    question_vector = model.encode(question).tolist()
    print(f"[retriever] Vecteur créé ({len(question_vector)} dimensions)")

    # Etape 2 : chercher dans Qdrant
    client = get_client()
  
    results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=question_vector,
    limit=top_k,
    with_payload=True,
    score_threshold=0.3
    ).points

    # Etape 3 : formater les résultats
    chunks = []
    for r in results:
        chunks.append({
            "text":     r.payload["text"],
            "source":   r.payload["source"],
            "chunk_id": r.payload["chunk_id"],
            "score":    round(r.score, 4)
        })

    print(f"[retriever] {len(chunks)} chunks trouvés")
    return chunks


# Test
if __name__ == "__main__":
    question = "Je suis fonctionnaire au ministère de l'agriculture — quels cadres ont obtenu la classe exceptionnelle à l'emploi de chef de service dans la direction des affaires administratives ??"
    results = retrieve(question)

    print("\n=== RESULTATS ===")
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Score  : {chunk['score']}")
        print(f"Source : {chunk['source']}")
        print(f"Texte  : {chunk['text'][:200]}...")