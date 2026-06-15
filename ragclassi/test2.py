"""
Évaluation RAG avec Precision@K et MRR
Ground truth : réponses textuelles par question
Un chunk est "pertinent" si la réponse attendue apparaît dans son texte (insensible à la casse)
"""
import unicodedata
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
)
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import pdfplumber, uuid, time

# ── À modifier ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "jort_docs1028"
PDF_PATH        = "docs/15mai2026.pdf"
PDF_SOURCE      = "15mai2026.pdf"

CONFIGS = [
    {"name": "Config 4", "chunk_size": 1000, "overlap": 200, "threshold": 0.3, "top_k": 3,  "desc": "Grands chunks, overlap 20%, threshold strict"},
    {"name": "Config 5", "chunk_size": 700,  "overlap": 100, "threshold": 0.5, "top_k": 5,  "desc": "Chunks moyens, overlap amélioré"},
    {"name": "Config 6", "chunk_size": 1000,  "overlap": 200,  "threshold": 0.5, "top_k": 5,  "desc": "Petits chunks, ablation threshold"},
]

# ── Ground truth : réponse attendue pour chaque question ─────────────────────
# La réponse doit apparaître textuellement dans au moins un chunk retourné.
# Tu peux mettre plusieurs fragments alternatifs dans une liste.
GROUND_TRUTH = {
    # Centrales photovoltaïques
    "Quelle société a signé la convention pour la centrale photovoltaïque d'El Khobna ?":
        ["Qair International"],
    "Dans quel gouvernorat se trouve la centrale photovoltaïque El Ksar ?":
        ["Gafsa"],
    "Quelle loi approuve la centrale photovoltaïque de Mezzouna ?":
        ["2026-10"],
    "Quelle société a signé la convention pour la centrale de Segdoud ?":
        ["Voltalia"],
    "Dans quel gouvernorat se trouve la centrale photovoltaïque Menzel El Habib ?":
        ["Gabès", "Gabes"],
    "Quelle société a signé la convention pour la centrale photovoltaïque Menzel El Habib ?":
        ["Voltalia SA", "Voltalia"],
    "À quelle date a été signée la convention de la centrale de Segdoud ?":
        ["8 mai 2024"],
    "Dans quel gouvernorat se trouve la centrale photovoltaïque d'El Khobna ?":
        ["Sidi Bouzid"],
    "Quel est le numéro de la loi approuvant la centrale photovoltaïque El Ksar ?":
        ["2026-11"],
    "Quelle société a signé la convention pour la centrale photovoltaïque El Ksar ?":
        ["Qair International"],
    "Quelle société a signé la convention pour la centrale photovoltaïque de Mezzouna ?":
        ["Scatec ASA", "Scatec"],
    "Quel est le numéro de la loi approuvant la centrale photovoltaïque de Segdoud ?":
        ["2026-12"],

    # Nominations santé
    "Qui remplace Khaled Hachicha au conseil d'établissement de l'Office de Rjim Maâtoug ?":
        ["Amel Hadded"],
    "Qui est le ministre de la santé signataire de l'arrêté du 14 mai 2026 ?":
        ["Mustapha Ferjani"],
    "Qui remplace Maroua Bahri au conseil d'administration de l'hôpital Aziza Othmana ?":
        ["Maha Said"],
    "Qui remplace Hsan Hmida au conseil d'administration de l'Institut de neurologie de Tunis ?":
        ["Ayechi Jdidi"],
    "Qui remplace Adel Mohsni au conseil d'administration de l'hôpital Habib Bougatfa ?":
        ["Mohamed Amine Zoueghi"],
    "Qui remplace Aïda Borji au conseil d'administration de l'hôpital Bechir Hamza ?":
        ["Salem Yahiaoui"],
    "Qui remplace Kamel Yacoubi à la Compagnie des phosphates de Gafsa ?":
        ["Amel Abida"],
    "Qui est nommée administrateur représentant la Banque centrale de Tunisie à la CPG ?":
        ["Amel Abida"],
    "Qui est nommé membre représentant les médecins de libre pratique à l'Instance nationale de l'évaluation en santé ?":
        ["Mohamed Turki"],
    "Qui est nommée membre au conseil d'administration de l'hôpital Abderrahmane Mami de l'Ariana ?":
        ["Racha Ghabara"],
    "Qui est chargé des fonctions d'inspecteur des services médicaux au ministère de la santé ?":
        ["Hedi Ben Sliman"],
    "Qui est nommée chef de service de documents et d'archives au Centre national de formation pédagogique ?":
        ["Samia Mekssi"],

    # Ministères divers
    "Quels sont les deux candidats nommés analystes en chef au ministère de l'agriculture ?":
        ["Nahla Bououni", "Temime Horri"],
    "Qui se voit attribuer la classe exceptionnelle de chef de service au ministère de la jeunesse et des sports ?":
        ["Nadia Ferchichi"],
    "À qui est-il mis fin aux fonctions de chef de bureau à Siliana ?":
        ["Marouen Marzouk"],
    "Qui est chargé des fonctions de directeur de la planification au ministère de l'emploi ?":
        ["Hamouda Gabsi"],
    "Qui est chargé des fonctions de directeur d'appui des activités de formation au ministère de l'emploi ?":
        ["Abderrazek Bouafif"],
    "Qui est la ministre de la famille signataire de l'arrêté du 14 mai 2026 sur la formation des inspecteurs ?":
        ["Asma Jabri"],
    "Qui est la ministre des affaires culturelles signataire des arrêtés du 15 mai 2026 ?":
        ["Amina Srarfi"],
    "Qui est la Cheffe du Gouvernement ayant visé les arrêtés du 15 mai 2026 ?":
        ["Sarra Zaafrani Zenzri"],
    "Qui a promulgué les lois relatives aux centrales photovoltaïques du 15 mai 2026 ?":
        ["Kaïs Saïed", "Kais Saied"],
    "Qui se voit attribuer la classe exceptionnelle de directeur au ministère de la santé ?":
        ["Samia Goumani"],

    # Monuments et patrimoine
    "Dans quelle délégation se trouve l'Escargotière de Henchir el Magtaâ ?":
        ["Gafsa nord"],
    "Quelle est la superficie de l'Escargotière de Henchir el Magtaâ ?":
        ["5000 m²", "5000 m2", "5000"],
    "Quel est le rayon de protection de l'Escargotière de Henchir el Magtaâ ?":
        ["500 m", "500"],
    "Dans quelle délégation se trouve la Maison punique de Gammarth ?":
        ["Marsa", "la Marsa"],
    "Quelle est la superficie de la Maison punique de Gammarth ?":
        ["292 m²", "292 m2", "292"],
    "Quel est le rayon de protection de la Maison punique de Gammarth ?":
        ["200m", "200 m", "200"],

    # Questions moyennes
    "Quelles centrales photovoltaïques approuvées le 15 mai 2026 sont situées dans le gouvernorat de Sidi Bouzid ?":
        ["El Khobna", "Mezzouna"],
    "Quelles centrales photovoltaïques approuvées le 15 mai 2026 sont situées dans le gouvernorat de Gafsa ?":
        ["El Ksar", "Segdoud"],
    "Quelles centrales photovoltaïques ont un contrat de location du terrain en plus de la convention de concession ?":
        ["El Khobna", "Segdoud"],
    "Combien de lois relatives aux centrales photovoltaïques ont été publiées dans le JORT du 15 mai 2026 ?":
        ["cinq", "5"],
    "Quelle est la différence de date de signature entre la centrale de Segdoud et les autres centrales ?":
        ["8 mai 2024", "24 mars 2025"],
    "Quand l'Assemblée des représentants du peuple a-t-elle adopté les lois sur les centrales photovoltaïques ?":
        ["28 avril 2026"],
    "Quand le Conseil national des régions et des districts a-t-il adopté les lois sur les centrales photovoltaïques ?":
        ["13 mai 2026"],
    "Sur quelle loi de 1983 se fonde l'arrêté du ministre de la santé relatif aux substituts du lait maternel ?":
        ["83-24", "n°83-24"],
    "Quel arrêté antérieur est abrogé par l'arrêté du ministre de la santé du 14 mai 2026 ?":
        ["2 septembre 2025"],
    "Quelle commission a émis un avis sur la liste des substituts du lait maternel ?":
        ["commission nationale pour la promotion de l'alimentation du nourrisson"],
    "Quels avantages bénéficie le docteur Hedi Ben Sliman dans ses nouvelles fonctions au ministère de la santé ?":
        ["sous-directeur", "indemnités"],
    "Quelle est la durée de formation des inspecteurs de la jeunesse et de l'enfance selon le nouvel arrêté ?":
        ["deux (2) ans", "2 ans"],
    "Quel pourcentage de l'horaire est consacré aux sciences humaines dans la formation des inspecteurs ?":
        ["35%", "trente-cinq pour cent"],
    "Qui préside la commission pédagogique créée auprès du centre de formation des inspecteurs ?":
        ["directeur général du centre"],
    "Quels articles de l'arrêté de 2017 sont abrogés par le nouvel arrêté du 14 mai 2026 ?":
        ["articles 2, 5", "article 2", "article 5", "article 7", "article 11"],
    "Sur quel décret de 2019 se fonde l'arrêté de la ministre de la famille sur la formation des inspecteurs ?":
        ["2019-920"],
    "Combien de mois de congé annuel bénéficient les participants au cycle de formation des inspecteurs ?":
        ["un (1) mois", "1 mois"],
    "Quel arrêté de 2021 est modifié par le premier arrêté culturel du 15 mai 2026 ?":
        ["21 janvier 2021"],
    "Quel arrêté de 2022 est modifié par le second arrêté culturel du 15 mai 2026 ?":
        ["15 avril 2022"],
    "Quel est le numéro du plan TPD de l'Escargotière de Henchir el Magtaâ et sa date ?":
        ["94943", "13 octobre 2020"],
    "Quel est le numéro du plan TPD de la Maison punique de Gammarth et sa date ?":
        ["113851", "19 mai 2025"],
    "Quel est le numéro du titre foncier de la Maison punique de Gammarth ?":
        ["124041"],
    "Quelle commission a émis un avis sur la protection des monuments historiques en février 2026 ?":
        ["commission nationale du patrimoine"],
    "Quels articles du code du patrimoine sont visés par les arrêtés culturels du 15 mai 2026 ?":
        ["26", "27", "45", "47"],
    "À quelle date prend effet la nomination de Madame Amel Hadded à l'Office de Rjim Maâtoug ?":
        ["1er mai 2026", "1 mai 2026"],
    "À quelle date Docteur Racha Ghabara est-elle nommée au conseil d'administration de l'hôpital Abderrahmane Mami ?":
        ["26 janvier 2026"],
    "À quelle date Docteur Maha Said prend-elle ses fonctions au conseil d'administration de l'hôpital Aziza Othmana ?":
        ["15 janvier 2026"],
    "À quelle date Monsieur Ayechi Jdidi prend-il ses fonctions à l'Institut de neurologie de Tunis ?":
        ["16 avril 2026"],
    "À quelle date Monsieur Mohamed Amine Zoueghi prend-il ses fonctions à l'hôpital Habib Bougatfa ?":
        ["24 février 2026"],
    "Quelle est la qualité de Monsieur Mohamed Amine Zoueghi au conseil d'administration de l'hôpital Habib Bougatfa ?":
        ["partie syndicale", "syndicale"],
        # Questions supplémentaires

"À quelle date la convention de la centrale photovoltaïque d'El Khobna a-t-elle été signée ?":
    ["24 mars 2025"],

"À quelle date la convention de la centrale photovoltaïque de Mezzouna a-t-elle été signée ?":
    ["24 mars 2025"],

"À quelle date la convention de la centrale photovoltaïque El Ksar a-t-elle été signée ?":
    ["24 mars 2025"],

"À quelle date la convention de la centrale photovoltaïque Menzel El Habib a-t-elle été signée ?":
    ["24 mars 2025"],

"Entre quelles parties la convention de la centrale d'El Khobna a-t-elle été signée ?":
    ["Etat Tunisien", "Qair International"],

"Entre quelles parties la convention de la centrale de Mezzouna a-t-elle été signée ?":
    ["Etat Tunisien", "Scatec ASA"],

"Entre quelles parties la convention de la centrale El Ksar a-t-elle été signée ?":
    ["Etat Tunisien", "Qair International"],

"Entre quelles parties la convention de la centrale de Segdoud a-t-elle été signée ?":
    ["Etat tunisien", "Voltalia"],

"Entre quelles parties la convention de la centrale Menzel El Habib a-t-elle été signée ?":
    ["Etat Tunisien", "Voltalia SA"],

"Quelle centrale photovoltaïque est associée à la loi n°2026-9 ?":
    ["El Khobna"],

"Quelle centrale photovoltaïque est associée à la loi n°2026-13 ?":
    ["Menzel El Habib"],

"Quel ministre a signé l'arrêté fixant la liste des substituts du lait maternel ?":
    ["Mustapha Ferjani"],

"Quelle loi du 4 mars 1983 est citée dans l'arrêté sur les substituts du lait maternel ?":
    ["83-24"],

"Quel décret du 3 novembre 1984 est cité dans l'arrêté sur les substituts du lait maternel ?":
    ["84-1314"],

"À quelle date la commission nationale pour la promotion de l'alimentation du nourrisson a-t-elle rendu son avis ?":
    ["29 août 2029"],

"Quel organisme Madame Amel Hadded représente-t-elle à l'Office de Rjim Maâtoug ?":
    ["Office de développement du Sud"],

"Qui représente la commune de Tunis au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie ?":
    ["Ayechi Jdidi"],

"Qui représente les médecins assistants hospitalo-universitaires à l'hôpital Abderrahmane Mami ?":
    ["Racha Ghabara"],

"Qui représente les médecins assistants hospitalo-universitaires à l'hôpital Aziza Othmana ?":
    ["Maha Said"],

"Qui représente la partie syndicale la plus représentative à l'hôpital Habib Bougatfa ?":
    ["Mohamed Amine Zoueghi"],

"Quel est le grade de Hamouda Gabsi ?":
    ["Administrateur en chef"],

"Quel est le grade d'Abderrazek Bouafif ?":
    ["Administrateur général de l'éducation"],

"Quelle direction générale accueille le poste de directeur de la planification occupé par Hamouda Gabsi ?":
    ["direction générale de la planification, la programmation et des projets"],

"Quelle direction générale accueille le poste occupé par Abderrazek Bouafif ?":
    ["direction générale de la formation continue et de développement des compétences"],

"Quel décret de 1994 fixe la composition et le fonctionnement de la commission nationale du patrimoine ?":
    ["94-1475"],

"Quel code est cité dans les arrêtés culturels du 15 mai 2026 ?":
    ["code du patrimoine archéologique, historique et des arts traditionnels"],

"Quelle loi a promulgué le code du patrimoine archéologique, historique et des arts traditionnels ?":
    ["94-35"],

"Quelle loi du 6 décembre 2001 a modifié le code du patrimoine ?":
    ["2001-118"],

"Quel décret-loi du 25 mai 2011 a complété le code du patrimoine ?":
    ["2011-43"],

"Quel monument historique est situé dans la délégation de la Marsa ?":
    ["Maison punique de Gammarth"],
}


# ── Métriques ─────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    # lowercase
    text = text.lower()
    # retire les accents
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # collapse les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_is_relevant(chunk_text: str, question: str) -> bool:
    """
    Un chunk est pertinent si au moins un fragment de la réponse attendue
    apparaît dans le texte du chunk (insensible à la casse).
    """
    expected_fragments = GROUND_TRUTH.get(question)
    if not expected_fragments:
        return False
    chunk_norm = normalize(chunk_text)
    return any(normalize(frag) in chunk_norm for frag in expected_fragments)


def compute_precision_at_k(results, question: str, k: int) -> float:
    """Precision@K"""
    if not results:
        return 0.0
    top_k = results[:k]
    relevant = sum(1 for r in top_k if chunk_is_relevant(r.payload.get("text", ""), question))
    return relevant / len(top_k)


def compute_rr(results, question: str) -> float:
    """Reciprocal Rank : 1/rang du premier chunk pertinent (0 si aucun)."""
    for rank, r in enumerate(results, 1):
        if chunk_is_relevant(r.payload.get("text", ""), question):
            return 1.0 / rank
    return 0.0


def fmt_chunk(text, max_len=220):
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


# ── Init ──────────────────────────────────────────────────────────────────────
print("Chargement du modèle d'embedding...")
model  = SentenceTransformer("OrdalieTech/Solon-embeddings-large-0.1")
client = QdrantClient(host="localhost", port=6333)

QUESTIONS_SIMPLES = [q for q in list(GROUND_TRUTH.keys())[:36]]
QUESTIONS_MOYENNES = [q for q in list(GROUND_TRUTH.keys())[36:]]
ALL_QUESTIONS = [("SIMPLE", q) for q in QUESTIONS_SIMPLES] + \
                [("MOYENNE", q) for q in QUESTIONS_MOYENNES]


# ── Indexation ────────────────────────────────────────────────────────────────
def indexer_pdf(cfg):
    print(f"\n  → Indexation chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']}...")
    with pdfplumber.open(PDF_PATH) as pdf:
        texte = "\n".join(p.extract_text() or "" for p in pdf.pages)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(texte)
    print(f"  → {len(chunks)} chunks créés")

    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": chunk, "source": PDF_SOURCE, "chunk_id": i}
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    for i in range(0, len(points), 64):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+64])
    print(f"  → {len(chunks)} chunks uploadés")
    return len(chunks)


def supprimer_chunks():
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=PDF_SOURCE))]
        )
    )
    remaining = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=PDF_SOURCE))]
        ),
        exact=True
    )
    print(f"  → Chunks supprimés. Restants : {remaining.count}")


def tester_config(cfg):
    source_filter = Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=PDF_SOURCE))]
    )
    resultats = []

    for cat, question in ALL_QUESTIONS:
        vec = model.encode(question).tolist()
        t0  = time.time()
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=source_filter,
            limit=cfg["top_k"],
            score_threshold=cfg["threshold"],
            with_payload=True,
        )
        results = response.points
        elapsed = (time.time() - t0) * 1000

        p_at_1  = compute_precision_at_k(results, question, k=1)
        p_at_k  = compute_precision_at_k(results, question, k=cfg["top_k"])
        rr      = compute_rr(results, question)
        in_gt   = question in GROUND_TRUTH

        resultats.append({
            "cat": cat, "question": question,
            "results": results, "elapsed_ms": elapsed,
            "n": len(results),
            "p_at_1": p_at_1,
            "p_at_k": p_at_k,
            "rr": rr,
            "in_gt": in_gt,
        })

    return resultats


# ── Vérification collection ───────────────────────────────────────────────────
collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in collections:
    print(f"Création de la collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

# ── Boucle principale ─────────────────────────────────────────────────────────
SEP  = "=" * 100
sep2 = "-" * 100
all_lines    = []
global_stats = []

all_lines.append(SEP)
all_lines.append(f"  RAPPORT D'ÉVALUATION RAG — {PDF_SOURCE}")
all_lines.append(f"  Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
all_lines.append(f"  Collection : {COLLECTION_NAME} | Modèle : mpnet")
all_lines.append(f"  Questions  : {len(ALL_QUESTIONS)} total | Avec ground truth : {sum(1 for _, q in ALL_QUESTIONS if q in GROUND_TRUTH)}")
all_lines.append(f"  Métriques  : Precision@1, Precision@K, MRR (Mean Reciprocal Rank)")
all_lines.append(SEP)

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  {cfg['name']} | chunk={cfg['chunk_size']} overlap={cfg['overlap']} "
          f"threshold={cfg['threshold']} top_k={cfg['top_k']}")
    print(f"{'='*60}")

    nb_chunks  = indexer_pdf(cfg)
    print(f"  → Test retrieval sur {len(ALL_QUESTIONS)} questions...")
    resultats  = tester_config(cfg)
    print(f"  → Suppression des chunks...")
    supprimer_chunks()

    # ── Calcul des stats globales ──────────────────────────────────────────
    q_with_gt  = [r for r in resultats if r["in_gt"]]
    total_gt   = len(q_with_gt)

    mean_p1    = sum(r["p_at_1"] for r in q_with_gt) / total_gt if total_gt else 0
    mean_pk    = sum(r["p_at_k"] for r in q_with_gt) / total_gt if total_gt else 0
    mrr        = sum(r["rr"]     for r in q_with_gt) / total_gt if total_gt else 0
    hits       = sum(1 for r in q_with_gt if r["rr"] > 0)
    hit_rate   = hits / total_gt * 100 if total_gt else 0

    # ── Écriture rapport ──────────────────────────────────────────────────
    all_lines.append("")
    all_lines.append(SEP)
    all_lines.append(f"  {cfg['name'].upper()} | chunk_size={cfg['chunk_size']} | overlap={cfg['overlap']} | "
                     f"threshold={cfg['threshold']} | top_k={cfg['top_k']}")
    all_lines.append(f"  {cfg['desc']} | Chunks indexés : {nb_chunks}")
    all_lines.append(SEP)
    all_lines.append(f"  {'Q#':<5} {'Cat':<8} {'P@1':>5} {'P@K':>5} {'RR':>6} {'#chunks':>8}  Question")
    all_lines.append(f"  {'-'*5} {'-'*8} {'-'*5} {'-'*5} {'-'*6} {'-'*8}  {'-'*50}")

    cat_current = None
    for i, r in enumerate(resultats, 1):
        if r["cat"] != cat_current:
            cat_current = r["cat"]
            label = "QUESTIONS SIMPLES" if cat_current == "SIMPLE" else "QUESTIONS MOYENNES"
            all_lines.append(f"\n  ── {label} {'─' * (72 - len(label))}")

        gt_mark = "" if r["in_gt"] else " [NO GT]"
        all_lines.append(
            f"  Q{i:02d}   {r['cat']:<8} {r['p_at_1']:>5.2f} {r['p_at_k']:>5.2f} "
            f"{r['rr']:>6.3f} {r['n']:>8}  {r['question'][:70]}{gt_mark}"
        )

        # Détail des chunks
        for rank, res in enumerate(r["results"], 1):
            relevant_mark = " ✓" if chunk_is_relevant(res.payload.get("text", ""), r["question"]) else "  "
            chunk = fmt_chunk(res.payload.get("text", ""))
            all_lines.append(
                f"         [{rank}]{relevant_mark} score={res.score:.4f}  cid={res.payload.get('chunk_id','?')}"
            )
            all_lines.append(f"              {chunk}")

    all_lines.append("")
    all_lines.append(sep2)
    all_lines.append(f"  RÉSUMÉ {cfg['name']}")
    all_lines.append(f"  Questions avec GT : {total_gt}")
    all_lines.append(f"  Hit Rate          : {hit_rate:.1f}%  ({hits}/{total_gt})")
    all_lines.append(f"  MRR               : {mrr:.4f}")
    all_lines.append(f"  Precision@1       : {mean_p1:.4f}")
    all_lines.append(f"  Precision@{cfg['top_k']}       : {mean_pk:.4f}")
    all_lines.append(sep2)

    global_stats.append({
        "cfg": cfg, "hits": hits, "total_gt": total_gt,
        "hit_rate": hit_rate, "mrr": mrr,
        "mean_p1": mean_p1, "mean_pk": mean_pk,
        "nb_indexed": nb_chunks
    })

    print(f"   Hit Rate={hit_rate:.1f}%  MRR={mrr:.4f}  P@1={mean_p1:.4f}  P@{cfg['top_k']}={mean_pk:.4f}")

# ── Tableau comparatif final ──────────────────────────────────────────────────
all_lines.append("")
all_lines.append(SEP)
all_lines.append("  COMPARATIF FINAL — TOUTES CONFIGS")
all_lines.append(SEP)
all_lines.append(f"  {'Config':<12} {'chunk':>6} {'overlap':>8} {'top_k':>6} "
                 f"{'Indexed':>8} {'Hit Rate':>10} {'MRR':>8} {'P@1':>8} {'P@K':>8}")
all_lines.append(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

best_mrr = max(global_stats, key=lambda x: x["mrr"])
for s in global_stats:
    marker = " ◄ BEST MRR" if s == best_mrr else ""
    c = s["cfg"]
    all_lines.append(
        f"  {c['name']:<12} {c['chunk_size']:>6} {c['overlap']:>8} {c['top_k']:>6} "
        f"{s['nb_indexed']:>8} {s['hit_rate']:>9.1f}% {s['mrr']:>8.4f} "
        f"{s['mean_p1']:>8.4f} {s['mean_pk']:>8.4f}{marker}"
    )

all_lines.append("")
all_lines.append(f"  → Meilleure config (MRR) : {best_mrr['cfg']['name']} — "
                 f"MRR={best_mrr['mrr']:.4f}, P@1={best_mrr['mean_p1']:.4f}, "
                 f"Hit Rate={best_mrr['hit_rate']:.1f}%")
all_lines.append(SEP)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
output = "\n".join(all_lines)
with open("resultats_eval.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("\n" + "="*60)
print("  COMPARATIF FINAL")
print("="*60)
print(f"  {'Config':<12} {'Hit Rate':>10} {'MRR':>8} {'P@1':>8} {'P@K':>8}")
for s in global_stats:
    print(f"  {s['cfg']['name']:<12} {s['hit_rate']:>9.1f}% {s['mrr']:>8.4f} "
          f"{s['mean_p1']:>8.4f} {s['mean_pk']:>8.4f}")
print(f"\n  Meilleure (MRR) : {best_mrr['cfg']['name']} ({best_mrr['mrr']:.4f})")
print(f"\n Résultats sauvegardés dans resultats_eval.txt")