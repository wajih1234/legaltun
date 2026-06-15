from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from document_loader import load_and_chunk_pdf
import os

# Dossier local où Qdrant va stocker ses données (sans Docker)
client = QdrantClient(host="localhost", port=6333)

# Nom de la collection dans Qdrant (comme une table dans une DB)
COLLECTION_NAME = "jort_docs"

# Modele d'embedding multilingue (supporte le français)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Taille du vecteur produit par ce modele
VECTOR_SIZE = 384


def get_client():
    """
    Retourne un client Qdrant en mode local (sans Docker)
    """
    return QdrantClient(host="localhost", port=6333,check_compatibility=False )


def create_collection_if_not_exists(client):
    """
    Crée la collection dans Qdrant si elle n'existe pas encore
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE  
            )
        )
        print(f"[embedder] Collection '{COLLECTION_NAME}' créée")
    else:
        print(f"[embedder] Collection '{COLLECTION_NAME}' existe déjà")


def embed_and_store(pdf_path):
    """
    Fonction principale :
    1. Charge le PDF et le découpe en chunks
    2. Convertit chaque chunk en vecteur
    3. Stocke dans Qdrant
    """

    # Etape 1 : charger les chunks
    chunks = load_and_chunk_pdf(pdf_path)

    # Etape 2 : charger le modele d'embedding
    print(f"[embedder] Chargement du modele {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Etape 3 : convertir tous les chunks en vecteurs
    print(f"[embedder] Conversion de {len(chunks)} chunks en vecteurs...")
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=True)
    print(f"[embedder] Vecteurs créés : {vectors.shape}")

    # Etape 4 : connexion Qdrant et création collection
    client = get_client()
    create_collection_if_not_exists(client)

    # Etape 5 : stocker les points dans Qdrant
    # Chaque point = vecteur + payload (texte + métadonnées)
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=abs(hash(chunk["source"] + str(chunk["chunk_id"]))) % (10**9),
            vector=vector.tolist(),
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"]
            }
        ))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"[embedder] {len(points)} chunks stockés dans Qdrant ")
    print(f"[embedder] Source: {os.path.basename(pdf_path)}")


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        embed_and_store(sys.argv[1])
    else:
        print("Usage: python embedder.py docs/13mars2026.pdf")