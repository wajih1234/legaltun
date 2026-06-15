

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    """
    Lit un PDF et retourne le texte brut complet
    """
  
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Texte normal
            page_text = page.extract_text() or ""
            text += page_text + "\n"

            # Tableaux → convertis en texte structuré
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_clean = [cell or "" for cell in row]
                    text += " | ".join(row_clean) + "\n"

    return text


def split_into_chunks(text, source_name, chunk_size=700, overlap=100):
    """
    Découpe le texte en chunks avec overlap
    - chunk_size: taille de chaque chunk en caractères
    - overlap: combien de caractères partagés entre chunks consécutifs
    - source_name: nom du fichier PDF d'origine
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)

    # Chaque chunk garde la référence à son PDF source
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "source": source_name,
            "chunk_id": i
        })

    return result


def load_and_chunk_pdf(file_path):
    """
    Fonction principale : lit le PDF et retourne les chunks
    """
    import os
    source_name = os.path.basename(file_path)  # ex: JORT_3_03_2026.pdf

    print(f"[loader] Lecture de {source_name}...")
    text = load_pdf(file_path)
    print(f"[loader] {len(text)} caractères extraits")

    chunks = split_into_chunks(text, source_name)
    print(f"[loader] {len(chunks)} chunks créés")

    return chunks


# Test rapide
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        chunks = load_and_chunk_pdf(sys.argv[1])
        print(f"\nPremier chunk:")
        print(chunks[0]["text"][:300])
    else:
        print("Usage: python document_loader.py documents/JORT_3_03_2026.pdf")