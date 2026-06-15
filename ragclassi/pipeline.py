

from retriever import retrieve
from generator import generate_answer


def rag_pipeline(question, top_k=5):
    """
    Pipeline complet RAG :
    1. Retrieve  → cherche les chunks pertinents
    2. Generate  → génère la réponse finale

    - question : question en français
    - top_k    : nombre de chunks à récupérer
    """

    print(f"\n{'='*50}")
    print(f"QUESTION: {question}")
    print(f"{'='*50}")

    # Etape 1 : Retrieval
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "answer": "Aucune information trouvée dans les documents.",
            "chunks": [],
            "sources": []
        }

    # Etape 2 : Generation
    answer = generate_answer(question, chunks)

    # Etape 3 : Formater le résultat
    sources = list(set([chunk["source"] for chunk in chunks]))

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "sources": sources,
        "nb_chunks": len(chunks)
    }


# Test avec plusieurs questions
if __name__ == "__main__":

    questions = [
        "Qui est Maaouia Kaab et quelle fonction lui a été attribuée au ministère de l'industrie ?",
        "Quels gouvernorats ont été déclarés zones sinistrées par la calamité de sécheresse de la campagne agricole 2023-2024 ?",
        "QQuelles sont les conditions requises pour participer au concours interne de promotion au grade d'ingénieur général à l'Institut national de la consommation ?,",
        "Comment le jury du concours interne à l'Institut national de la consommation évalue-t-il les candidats et quel critère départage les ex-aequo ?"
    ]

    for question in questions:
        result = rag_pipeline(question)

        print(f"\n--- REPONSE ---")
        print(result["answer"])
        print(f"\nSources: {result['sources']}")
        print(f"Chunks utilisés: {result['nb_chunks']}")
        print(f"{'='*50}")