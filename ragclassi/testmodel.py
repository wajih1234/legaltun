from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd

models = {
    "MiniLM-L12 ": "paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet-base":           "paraphrase-multilingual-mpnet-base-v2",
    "sota":"OrdalieTech/Solon-embeddings-large-0.1",
    "allminlm-l6":"sentence-transformers/all-MiniLM-L6-v2"
}

pairs =[
    {
        "query": "Qui a signé la convention pour la centrale photovoltaïque d'El Khobna ?",
        "relevant": "Sont approuvés la convention de concession de production d'électricité, le contrat de location du terrain et leurs annexes de la centrale photovoltaïque d'El Khobna au gouvernorat de Sidi Bouzid signés à Tunis le 24 mars 2025 entre l'Etat Tunisien et la société Qair International.",
        "irrelevant": "Madame Amel Hadded est nommée membre représentant de l'Office de développement du Sud au conseil d'établissement de l'Office de Rjim Maâtoug."
    },
    {
        "query": "Quelle loi approuve la centrale photovoltaïque de Mezzouna ?",
        "relevant": "La loi n°2026-10 du 15 mai 2026 porte approbation de la convention de concession de production d'électricité et ses annexes de la centrale photovoltaïque de Mezzouna au gouvernorat de Sidi Bouzid signée avec la société Scatec ASA.",
        "irrelevant": "La liste des substituts du lait maternel est fixée à l'annexe jointe au présent arrêté du ministre de la santé."
    },
    {
        "query": "Dans quel gouvernorat se trouve la centrale photovoltaïque El Ksar ?",
        "relevant": "Sont approuvées la convention de concession de production d'électricité et ses annexes de la centrale photovoltaïque El Ksar au gouvernorat de Gafsa signées à Tunis le 24 mars 2025 entre l'Etat Tunisien et la société Qair International.",
        "irrelevant": "Sont abrogées toutes les dispositions antérieures contraires au présent arrêté, notamment l'arrêté du 2 septembre 2025."
    },
    {
        "query": "Quelle société a signé la convention pour la centrale de Segdoud et à quelle date ?",
        "relevant": "Sont approuvés la convention de concession de production d'électricité, le contrat de location du terrain et leurs annexes de la centrale photovoltaïque de Segdoud, au gouvernorat de Gafsa, signés à Tunis le 8 mai 2024 entre l'Etat tunisien et la société Voltalia.",
        "irrelevant": "Nahla Bououni et Temime Horri sont nommés dans le grade d'analyste en chef au corps des analystes et des techniciens de l'informatique des administrations publiques."
    },
    {
        "query": "Dans quel gouvernorat se trouve la centrale photovoltaïque Menzel El Habib ?",
        "relevant": "Sont approuvées la convention de concession de production d'électricité et ses annexes de la centrale photovoltaïque Menzel El Habib au gouvernorat de Gabès signées à Tunis le 24 mars 2025 entre l'Etat Tunisien et la société Voltalia SA.",
        "irrelevant": "La formation au cycle de formation des inspecteurs de la jeunesse et de l'enfance dure deux ans."
    },
    {
        "query": "Qui remplace Khaled Hachicha au conseil d'établissement de l'Office de Rjim Maâtoug ?",
        "relevant": "Madame Amel Hadded est nommée membre représentant de l'Office de développement du Sud au conseil d'établissement de l'Office de Rjim Maâtoug pour le développement du Sud et du Sahara, en remplacement du Monsieur Khaled Hachicha, à compter du 1er mai 2026.",
        "irrelevant": "La centrale photovoltaïque de Segdoud au gouvernorat de Gafsa a été signée avec la société Voltalia le 8 mai 2024."
    },
    {
        "query": "Quel arrêté le ministre de la santé a-t-il signé le 14 mai 2026 ?",
        "relevant": "L'arrêté du ministre de la santé du 14 mai 2026 fixe la liste des substituts du lait maternel et abroge toutes les dispositions antérieures contraires, notamment l'arrêté du 2 septembre 2025.",
        "irrelevant": "La centrale photovoltaïque El Ksar est située au gouvernorat de Gafsa signée avec Qair International."
    },
    {
        "query": "Quel est le nom du ministre de la santé signataire de l'arrêté du 14 mai 2026 ?",
        "relevant": "Le ministre de la santé Mustapha Ferjani a signé l'arrêté du 14 mai 2026 fixant la liste des substituts du lait maternel, visé par la Cheffe du Gouvernement Sarra Zaafrani Zenzri.",
        "irrelevant": "Madame Amel Abida est nommée administrateur représentant la Banque centrale de Tunisie au conseil d'administration de la Compagnie des phosphates de Gafsa."
    },
    {
        "query": "Qui est nommée administrateur au conseil d'administration de la Compagnie des phosphates de Gafsa ?",
        "relevant": "Madame Amel Abida est nommée administrateur représentant la Banque centrale de Tunisie au conseil d'administration de la Compagnie des phosphates de Gafsa, en remplacement de Monsieur Kamel Yacoubi.",
        "irrelevant": "L'arrêté du 14 mai 2026 fixe la liste des substituts du lait maternel et abroge l'arrêté du 2 septembre 2025."
    },
    {
        "query": "Qui est nommé membre au conseil d'administration de l'hôpital Bechir Hamza de Tunis ?",
        "relevant": "Professeur Salem Yahiaoui est nommé membre représentant du doyen de la faculté de médecine de Tunis au conseil d'administration de l'hôpital d'enfants Bechir Hamza de Tunis, en remplacement du professeur Aïda Borji, à compter du 30 mars 2026.",
        "irrelevant": "La centrale photovoltaïque d'El Khobna au gouvernorat de Sidi Bouzid a été signée avec la société Qair International le 24 mars 2025."
    },
    {
        "query": "Qui est nommée au conseil d'administration de l'hôpital Abderrahmane Mami de l'Ariana ?",
        "relevant": "Docteur Racha Ghabara est nommée membre représentant des médecins assistants hospitalo-universitaires au conseil d'administration de l'hôpital de pneumo-phtisiologie Abderrahmane Mami de l'Ariana, à compter du 26 janvier 2026.",
        "irrelevant": "Madame Nadia Ferchichi s'est vue attribuer la classe exceptionnelle à l'emploi de chef de service au ministère de la jeunesse et des sports."
    },
    {
        "query": "Qui remplace Maroua Bahri au conseil d'administration de l'hôpital Aziza Othmana ?",
        "relevant": "Docteur Maha Said est nommée membre représentant des médecins assistants hospitalo-universitaires au conseil d'administration de l'hôpital Aziza Othmana de Tunis en remplacement du docteur Maroua Bahri, à compter du 15 janvier 2026.",
        "irrelevant": "La convention de la centrale photovoltaïque de Mezzouna a été signée avec la société Scatec ASA le 24 mars 2025."
    },
    {
        "query": "Qui est nommé membre représentant les médecins de libre pratique à l'Instance nationale de l'évaluation en santé ?",
        "relevant": "Docteur Mohamed Turki est nommé membre représentant les médecins de libre pratique au conseil d'établissement de l'Instance nationale de l'évaluation et de l'accréditation en santé, à compter du 31 mars 2026.",
        "irrelevant": "La Maison punique de Gammarth a une superficie de 292 m² avec un rayon de protection de 200 mètres."
    },
    {
        "query": "Qui remplace Hsan Hmida au conseil d'administration de l'Institut de neurologie de Tunis ?",
        "relevant": "Monsieur Ayechi Jdidi est nommé membre représentant la commune de Tunis au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie de Tunis en remplacement de Monsieur Hsan Hmida, à compter du 16 avril 2026.",
        "irrelevant": "Les deux candidats Nahla Bououni et Temime Horri sont nommés dans le grade d'analyste en chef au ministère de l'agriculture."
    },
    {
        "query": "Qui est nommé membre au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte ?",
        "relevant": "Monsieur Mohamed Amine Zoueghi est nommé membre représentant de la partie syndicale la plus représentative au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte en remplacement de Monsieur Adel Mohsni, à compter du 24 février 2026.",
        "irrelevant": "La centrale photovoltaïque Menzel El Habib au gouvernorat de Gabès a été signée avec la société Voltalia SA."
    },
    {
        "query": "Quelle est la durée de formation des inspecteurs de la jeunesse et de l'enfance ?",
        "relevant": "La formation au cycle de formation des inspecteurs de la jeunesse et de l'enfance dure deux ans, au cours de laquelle les participants bénéficient d'un mois de congé annuel pour chaque année accordée par le centre de formation.",
        "irrelevant": "Madame Amel Abida est nommée administrateur représentant la Banque centrale de Tunisie à la Compagnie des phosphates de Gafsa."
    },
    {
        "query": "Quel pourcentage de l'horaire est consacré aux sciences humaines dans la formation des inspecteurs ?",
        "relevant": "Un domaine théorique portant sur les sciences humaines et sociales, les approches pédagogiques et fondées sur les droits, les méthodologies et les problématiques de l'enfance, auquel est consacré trente-cinq pour cent (35%) de l'horaire global de la formation.",
        "irrelevant": "Docteur Mohamed Turki est nommé membre représentant les médecins de libre pratique à l'Instance nationale de l'évaluation et de l'accréditation en santé."
    },
    {
        "query": "Quelle est la superficie protégée de l'Escargotière de Henchir el Magtaâ ?",
        "relevant": "L'Escargotière de Henchir el Magtaâ, délégation de Gafsa nord, a une superficie de 5000 m² selon les limites indiquées sur les plans TPD n°94943 du 13 octobre 2020, avec un rayon de 500 mètres aux abords.",
        "irrelevant": "Professeur Salem Yahiaoui est nommé membre au conseil d'administration de l'hôpital d'enfants Bechir Hamza de Tunis."
    },
    {
        "query": "Quelle est la superficie de la Maison punique de Gammarth et son rayon de protection ?",
        "relevant": "La Maison punique de Gammarth, délégation de la Marsa, a une superficie de 292 m² du titre foncier n°124041 Tunis, avec un rayon de 200 mètres aux abords, selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026.",
        "irrelevant": "L'arrêté du ministre de la santé du 14 mai 2026 abroge l'arrêté du 2 septembre 2025 fixant la liste des substituts du lait maternel."
    },
    {
        "query": "Quels cadres sont chargés des emplois fonctionnels au ministère de l'emploi le 7 mai 2026 ?",
        "relevant": "Par arrêté du 7 mai 2026, Hamouda Gabsi, administrateur en chef, est chargé des fonctions de directeur de la planification, et Abderrazek Bouafif, administrateur général de l'éducation, est chargé de directeur d'appui des activités de formation au ministère de l'emploi et de la formation professionnelle.",
        "irrelevant": "La centrale photovoltaïque El Ksar au gouvernorat de Gafsa a été approuvée par la loi n°2026-11 du 15 mai 2026."
    },
    {
        "query": "Quelle loi approuve la convention pour la centrale photovoltaïque d'El Khobna ?",
        "relevant": "La loi n°2026-9 du 15 mai 2026 porte approbation de la convention de concession de production d'électricité, du contrat de location du terrain et leurs annexes de la centrale photovoltaïque d'El Khobna au gouvernorat de Sidi Bouzid.",
        "irrelevant": "La loi n°2026-12 du 15 mai 2026 porte approbation de la convention de concession de production d'électricité de la centrale photovoltaïque de Segdoud au gouvernorat de Gafsa signée avec Voltalia."
    },
    {
        "query": "Quel est le numéro de la loi approuvant la centrale photovoltaïque El Ksar ?",
        "relevant": "La loi n°2026-11 du 15 mai 2026 porte approbation de la convention de concession de production d'électricité et ses annexes de la centrale photovoltaïque El Ksar au gouvernorat de Gafsa.",
        "irrelevant": "La loi n°2026-10 du 15 mai 2026 porte approbation de la convention de la centrale photovoltaïque de Mezzouna au gouvernorat de Sidi Bouzid signée avec Scatec ASA."
    },
    {
        "query": "Quel est le numéro de la loi approuvant la centrale photovoltaïque de Segdoud ?",
        "relevant": "La loi n°2026-12 du 15 mai 2026 porte approbation de la convention de concession de production d'électricité, du contrat de location du terrain et leurs annexes de la centrale photovoltaïque de Segdoud au gouvernorat de Gafsa.",
        "irrelevant": "La loi n°2026-13 du 15 mai 2026 porte approbation de la convention de la centrale photovoltaïque Menzel El Habib au gouvernorat de Gabès signée avec Voltalia SA."
    },
    {
        "query": "Quel est le numéro de la loi approuvant la centrale photovoltaïque Menzel El Habib ?",
        "relevant": "La loi n°2026-13 du 15 mai 2026 porte approbation de la convention de concession de production d'électricité et ses annexes de la centrale photovoltaïque Menzel El Habib au gouvernorat de Gabès.",
        "irrelevant": "La loi n°2026-9 du 15 mai 2026 porte approbation de la convention de la centrale photovoltaïque d'El Khobna au gouvernorat de Sidi Bouzid signée avec Qair International."
    },
    {
        "query": "Qui a promulgué les lois relatives aux centrales photovoltaïques du 15 mai 2026 ?",
        "relevant": "Le Président de la République Kaïs Saïed a promulgué les lois n°2026-9 à 2026-13 du 15 mai 2026 relatives aux centrales photovoltaïques, après adoption par l'Assemblée des représentants du peuple et le Conseil national des régions et des districts.",
        "irrelevant": "La Cheffe du Gouvernement Sarra Zaafrani Zenzri a visé l'arrêté du ministre de la santé du 14 mai 2026 fixant la liste des substituts du lait maternel."
    },
    {
        "query": "Quand l'Assemblée des représentants du peuple a-t-elle adopté les lois sur les centrales photovoltaïques ?",
        "relevant": "L'Assemblée des représentants du peuple a discuté et adopté les lois relatives aux centrales photovoltaïques dans sa séance du 28 avril 2026.",
        "irrelevant": "Le Conseil national des régions et des districts a adopté ces lois dans sa séance du 13 mai 2026."
    },
    {
        "query": "Quand le Conseil national des régions et des districts a-t-il adopté les lois sur les centrales photovoltaïques ?",
        "relevant": "Le Conseil national des régions et des districts a discuté et adopté les lois relatives aux centrales photovoltaïques dans sa séance du 13 mai 2026.",
        "irrelevant": "L'Assemblée des représentants du peuple a adopté ces lois dans sa séance du 28 avril 2026."
    },
    {
        "query": "Quelle société a signé la convention pour la centrale photovoltaïque El Ksar et à quelle date ?",
        "relevant": "La convention de la centrale photovoltaïque El Ksar au gouvernorat de Gafsa a été signée à Tunis le 24 mars 2025 entre l'État Tunisien et la société Qair International.",
        "irrelevant": "La convention de la centrale photovoltaïque de Segdoud au gouvernorat de Gafsa a été signée à Tunis le 8 mai 2024 entre l'État tunisien et la société Voltalia."
    },
    {
        "query": "Quelle société a signé la convention pour la centrale photovoltaïque de Mezzouna et à quelle date ?",
        "relevant": "La convention de la centrale photovoltaïque de Mezzouna au gouvernorat de Sidi Bouzid a été signée à Tunis le 24 mars 2025 entre l'État Tunisien et la société Scatec ASA.",
        "irrelevant": "La convention de la centrale photovoltaïque d'El Khobna au gouvernorat de Sidi Bouzid a été signée à Tunis le 24 mars 2025 entre l'État Tunisien et la société Qair International."
    },
    {
        "query": "Quelle société a signé la convention pour la centrale photovoltaïque Menzel El Habib ?",
        "relevant": "La convention de la centrale photovoltaïque Menzel El Habib au gouvernorat de Gabès a été signée à Tunis le 24 mars 2025 entre l'État Tunisien et la société Voltalia SA.",
        "irrelevant": "La convention de la centrale photovoltaïque de Mezzouna au gouvernorat de Sidi Bouzid a été signée avec la société Scatec ASA."
    },
    {
        "query": "Dans quel gouvernorat se trouve la centrale photovoltaïque d'El Khobna ?",
        "relevant": "La centrale photovoltaïque d'El Khobna est située au gouvernorat de Sidi Bouzid, selon la loi n°2026-9 du 15 mai 2026.",
        "irrelevant": "La centrale photovoltaïque El Ksar est située au gouvernorat de Gafsa, selon la loi n°2026-11 du 15 mai 2026."
    },
    {
        "query": "Dans quel gouvernorat se trouve la centrale photovoltaïque de Mezzouna ?",
        "relevant": "La centrale photovoltaïque de Mezzouna est située au gouvernorat de Sidi Bouzid, selon la loi n°2026-10 du 15 mai 2026.",
        "irrelevant": "La centrale photovoltaïque Menzel El Habib est située au gouvernorat de Gabès, selon la loi n°2026-13 du 15 mai 2026."
    },
    {
        "query": "Dans quel gouvernorat se trouve la centrale photovoltaïque de Segdoud ?",
        "relevant": "La centrale photovoltaïque de Segdoud est située au gouvernorat de Gafsa, selon la loi n°2026-12 du 15 mai 2026.",
        "irrelevant": "La centrale photovoltaïque de Mezzouna est située au gouvernorat de Sidi Bouzid, selon la loi n°2026-10 du 15 mai 2026."
    },
    {
        "query": "Dans quel gouvernorat se trouve la centrale photovoltaïque Menzel El Habib ?",
        "relevant": "La centrale photovoltaïque Menzel El Habib est située au gouvernorat de Gabès, selon la loi n°2026-13 du 15 mai 2026.",
        "irrelevant": "La centrale photovoltaïque de Segdoud est située au gouvernorat de Gafsa, selon la loi n°2026-12 du 15 mai 2026."
    },
    {
        "query": "Combien de lois relatives aux centrales photovoltaïques ont été publiées dans le JORT du 15 mai 2026 ?",
        "relevant": "Cinq lois relatives aux centrales photovoltaïques ont été publiées dans le JORT du 15 mai 2026 : les lois n°2026-9, 2026-10, 2026-11, 2026-12 et 2026-13.",
        "irrelevant": "Le JORT du 15 mai 2026 est le numéro 49 de la 169ème année, publié le vendredi 28 dhoulkaâda 1447."
    },
    {
        "query": "Quelles centrales photovoltaïques ont un contrat de location du terrain en plus de la convention de concession ?",
        "relevant": "Les centrales photovoltaïques d'El Khobna (loi 2026-9) et de Segdoud (loi 2026-12) ont à la fois une convention de concession de production d'électricité et un contrat de location du terrain parmi leurs documents approuvés.",
        "irrelevant": "Les centrales photovoltaïques de Mezzouna, El Ksar et Menzel El Habib n'ont qu'une convention de concession de production d'électricité et ses annexes approuvées par les lois correspondantes."
    },
    {
        "query": "Quelle est la date de signature de la convention de la centrale de Segdoud par rapport aux autres centrales ?",
        "relevant": "La convention de la centrale photovoltaïque de Segdoud a été signée le 8 mai 2024, soit près d'un an avant les quatre autres centrales dont les conventions ont été signées le 24 mars 2025.",
        "irrelevant": "La convention de la centrale photovoltaïque El Ksar au gouvernorat de Gafsa a été signée à Tunis le 24 mars 2025 avec la société Qair International."
    },
    {
        "query": "Quelle est la Cheffe du Gouvernement qui a visé les arrêtés publiés dans le JORT du 15 mai 2026 ?",
        "relevant": "La Cheffe du Gouvernement Sarra Zaafrani Zenzri a visé les arrêtés publiés dans le JORT du 15 mai 2026, notamment l'arrêté du ministre de la santé fixant la liste des substituts du lait maternel et les arrêtés de la ministre des affaires culturelles.",
        "irrelevant": "Le Président de la République Kaïs Saïed a promulgué les lois n°2026-9 à 2026-13 relatives aux centrales photovoltaïques."
    },
    {
        "query": "Quel est le numéro du JORT publié le 15 mai 2026 ?",
        "relevant": "Le JORT publié le 15 mai 2026 est le numéro 49 de la 169ème année, correspondant au vendredi 28 dhoulkaâda 1447.",
        "irrelevant": "Le ministre de la santé Mustapha Ferjani a signé l'arrêté du 14 mai 2026 fixant la liste des substituts du lait maternel."
    },
    {
        "query": "Quel arrêté a été abrogé par l'arrêté du ministre de la santé du 14 mai 2026 ?",
        "relevant": "L'arrêté du 14 mai 2026 du ministre de la santé abroge l'arrêté du 2 septembre 2025 fixant la liste des substituts du lait maternel, ainsi que toutes dispositions antérieures contraires.",
        "irrelevant": "L'arrêté de la ministre de la famille du 14 mai 2026 abroge des dispositions de l'arrêté du 22 mars 2017 fixant le régime de formation des inspecteurs de la jeunesse et de l'enfance."
    },
    {
        "query": "Sur quelle loi se base l'arrêté du ministre de la santé relatif aux substituts du lait maternel ?",
        "relevant": "L'arrêté du ministre de la santé du 14 mai 2026 se fonde notamment sur la loi n°83-24 du 4 mars 1983 relative au contrôle de la qualité, à la commercialisation et à l'information sur l'utilisation des substituts du lait maternel et produits apparentés.",
        "irrelevant": "L'arrêté de la ministre de la famille du 14 mai 2026 se fonde sur le décret gouvernemental n°2019-920 du 26 septembre 2019 portant statut particulier des membres du corps de l'inspection pédagogique."
    },
    {
        "query": "Quel décret régit la commission nationale pour la promotion de l'alimentation du nourrisson ?",
        "relevant": "Le décret n°84-1314 du 3 novembre 1984 fixe les attributions, la composition et le mode de fonctionnement de la commission nationale pour la promotion de l'alimentation du nourrisson et de l'enfant.",
        "irrelevant": "Le décret n°2003-2020 du 22 septembre 2003 fixe les attributions du ministère des affaires de la femme, de la famille et de l'enfance."
    },
    {
        "query": "Quelle commission a émis un avis sur la liste des substituts du lait maternel ?",
        "relevant": "La commission nationale pour la promotion de l'alimentation du nourrisson et de l'enfant a émis un avis lors de sa réunion du 29 août 2029, sur lequel se fonde l'arrêté du ministre de la santé du 14 mai 2026.",
        "irrelevant": "La commission nationale du patrimoine a émis un avis lors de sa réunion du 13 février 2026 concernant la protection des monuments historiques et archéologiques."
    },
    {
        "query": "Qui est nommée à la classe exceptionnelle de directeur d'administration centrale au ministère de la santé ?",
        "relevant": "Madame Samia Goumani épouse El Badri, administrateur général de la santé publique et directeur du centre national de greffe de la moelle osseuse, s'est vu attribuer la classe exceptionnelle à l'emploi de directeur d'administration centrale au ministère de la santé, par arrêté du 23 avril 2026.",
        "irrelevant": "Madame Nadia Ferchichi s'est vu attribuer la classe exceptionnelle à l'emploi de chef de service d'administration centrale au ministère de la jeunesse et des sports."
    },
    {
        "query": "Qui est nommé inspecteur des services médicaux au ministère de la santé ?",
        "relevant": "Le docteur Hedi Ben Sliman, inspecteur régional de la santé publique, est chargé des fonctions d'inspecteur des services médicaux à l'inspection médicale au ministère de la santé, par arrêté du 15 mai 2026.",
        "irrelevant": "Monsieur Ayechi Jdidi est nommé membre représentant la commune de Tunis au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie."
    },
    {
        "query": "Quels avantages bénéficie le docteur Hedi Ben Sliman dans ses nouvelles fonctions ?",
        "relevant": "En application de l'article 13 du décret n°81-793 du 9 juin 1981, le docteur Hedi Ben Sliman bénéficie des indemnités et des avantages attribués à l'emploi de sous-directeur d'administration centrale.",
        "irrelevant": "Madame Samia Goumani épouse El Badri bénéficie de la classe exceptionnelle à l'emploi de directeur d'administration centrale au ministère de la santé."
    },
    {
        "query": "Qui est nommée chef de service de documents et d'archives au Centre national de formation pédagogique des cadres de la santé ?",
        "relevant": "Madame Samia Mekssi, professeur principal émérite de l'enseignement paramédical, est chargée des fonctions de chef de service de documents et d'archives au Centre national de formation pédagogique des cadres de la santé, par arrêté du 23 avril 2026.",
        "irrelevant": "Docteur Maha Said est nommée membre représentant des médecins assistants hospitalo-universitaires au conseil d'administration de l'hôpital Aziza Othmana de Tunis."
    },
    {
        "query": "Qui est remplacé par le professeur Salem Yahiaoui au conseil d'administration de l'hôpital Bechir Hamza ?",
        "relevant": "Le professeur Salem Yahiaoui remplace le professeur Aïda Borji au conseil d'administration de l'hôpital d'enfants Bechir Hamza de Tunis, à compter du 30 mars 2026.",
        "irrelevant": "Docteur Maha Said remplace le docteur Maroua Bahri au conseil d'administration de l'hôpital Aziza Othmana de Tunis, à compter du 15 janvier 2026."
    },
    {
        "query": "À quelle date le professeur Salem Yahiaoui prend-il ses fonctions au conseil d'administration de l'hôpital Bechir Hamza ?",
        "relevant": "Le professeur Salem Yahiaoui prend ses fonctions de membre au conseil d'administration de l'hôpital d'enfants Bechir Hamza de Tunis à compter du 30 mars 2026.",
        "irrelevant": "Docteur Racha Ghabara est nommée membre au conseil d'administration de l'hôpital Abderrahmane Mami de l'Ariana à compter du 26 janvier 2026."
    },
    {
        "query": "Quelle est la qualité du professeur Salem Yahiaoui au conseil d'administration de l'hôpital Bechir Hamza ?",
        "relevant": "Le professeur Salem Yahiaoui est nommé membre représentant du doyen de la faculté de médecine de Tunis au conseil d'administration de l'hôpital d'enfants Bechir Hamza de Tunis.",
        "irrelevant": "Monsieur Mohamed Amine Zoueghi est nommé membre représentant de la partie syndicale la plus représentative au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte."
    },
    {
        "query": "À quelle date Docteur Racha Ghabara est-elle nommée au conseil d'administration de l'hôpital Abderrahmane Mami ?",
        "relevant": "Docteur Racha Ghabara est nommée membre au conseil d'administration de l'hôpital de pneumo-phtisiologie Abderrahmane Mami de l'Ariana à compter du 26 janvier 2026.",
        "irrelevant": "Le professeur Salem Yahiaoui est nommé membre au conseil d'administration de l'hôpital Bechir Hamza de Tunis à compter du 30 mars 2026."
    },
    {
        "query": "Quelle est la qualité de Docteur Racha Ghabara au conseil d'administration de l'hôpital Abderrahmane Mami ?",
        "relevant": "Docteur Racha Ghabara est nommée membre représentant des médecins assistants hospitalo-universitaires exerçant au sein de l'hôpital au conseil d'administration de l'hôpital Abderrahmane Mami de l'Ariana.",
        "irrelevant": "Docteur Mohamed Turki est nommé membre représentant des médecins de libre pratique au conseil d'établissement de l'Instance nationale de l'évaluation et de l'accréditation en santé."
    },
    {
        "query": "À quelle date Docteur Maha Said prend-elle ses fonctions au conseil d'administration de l'hôpital Aziza Othmana ?",
        "relevant": "Docteur Maha Said est nommée membre au conseil d'administration de l'hôpital Aziza Othmana de Tunis à compter du 15 janvier 2026.",
        "irrelevant": "Monsieur Ayechi Jdidi est nommé membre au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie de Tunis à compter du 16 avril 2026."
    },
    {
        "query": "À quelle date Docteur Mohamed Turki prend-il ses fonctions à l'Instance nationale de l'évaluation en santé ?",
        "relevant": "Docteur Mohamed Turki est nommé membre au conseil d'établissement de l'Instance nationale de l'évaluation et de l'accréditation en santé à compter du 31 mars 2026.",
        "irrelevant": "Madame Amel Hadded est nommée membre au conseil d'établissement de l'Office de Rjim Maâtoug à compter du 1er mai 2026."
    },
    {
        "query": "Quel est l'hôpital dont relève l'Institut national El Mongi Ben Hmida ?",
        "relevant": "L'Institut national El Mongi Ben Hmida est un institut de neurologie situé à Tunis, dont Monsieur Ayechi Jdidi est nommé membre représentant de la commune de Tunis au conseil d'administration.",
        "irrelevant": "L'hôpital de pneumo-phtisiologie Abderrahmane Mami est situé à l'Ariana, dont Docteur Racha Ghabara est nommée membre représentant des médecins assistants hospitalo-universitaires."
    },
    {
        "query": "À quelle date Monsieur Ayechi Jdidi prend-il ses fonctions à l'Institut de neurologie de Tunis ?",
        "relevant": "Monsieur Ayechi Jdidi est nommé membre au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie de Tunis à compter du 16 avril 2026.",
        "irrelevant": "Monsieur Mohamed Amine Zoueghi est nommé membre au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte à compter du 24 février 2026."
    },
    {
        "query": "Qui est remplacé par Monsieur Mohamed Amine Zoueghi au conseil d'administration de l'hôpital Habib Bougatfa ?",
        "relevant": "Monsieur Mohamed Amine Zoueghi remplace Monsieur Adel Mohsni au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte, à compter du 24 février 2026.",
        "irrelevant": "Madame Amel Abida remplace Monsieur Kamel Yacoubi au conseil d'administration de la Compagnie des phosphates de Gafsa."
    },
    {
        "query": "À quelle date Monsieur Mohamed Amine Zoueghi prend-il ses fonctions à l'hôpital Habib Bougatfa ?",
        "relevant": "Monsieur Mohamed Amine Zoueghi est nommé membre au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte à compter du 24 février 2026.",
        "irrelevant": "Docteur Mohamed Turki est nommé membre au conseil d'établissement de l'Instance nationale de l'évaluation et de l'accréditation en santé à compter du 31 mars 2026."
    },
    {
        "query": "Quel est le ministère dont relève la Compagnie des phosphates de Gafsa selon l'arrêté de nomination de Madame Amel Abida ?",
        "relevant": "La nomination de Madame Amel Abida au conseil d'administration de la Compagnie des phosphates de Gafsa est faite par arrêté du ministre chargé à titre temporaire de diriger le ministère de l'industrie des mines et de l'énergie, en date du 8 mai 2026.",
        "irrelevant": "La nomination de Monsieur Mohamed Amine Zoueghi au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte est faite par arrêté du ministre de la santé du 15 mai 2026."
    },
    {
        "query": "Qui est remplacé par Madame Amel Abida au conseil d'administration de la Compagnie des phosphates de Gafsa ?",
        "relevant": "Madame Amel Abida remplace Monsieur Kamel Yacoubi au conseil d'administration de la Compagnie des phosphates de Gafsa en tant que représentante de la Banque centrale de Tunisie.",
        "irrelevant": "Monsieur Ayechi Jdidi remplace Monsieur Hsan Hmida au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie de Tunis."
    },
    {
        "query": "Quels sont les deux candidats nommés analystes en chef au ministère de l'agriculture ?",
        "relevant": "Les deux candidats nommés dans le grade d'analyste en chef au corps des analystes et des techniciens de l'informatique des administrations publiques au ministère de l'agriculture sont Nahla Bououni et Temime Horri, par arrêté du 7 mai 2026.",
        "irrelevant": "Hamouda Gabsi et Abderrazek Bouafif sont chargés d'emplois fonctionnels au sein du ministère de l'emploi et de la formation professionnelle par arrêté du 7 mai 2026."
    },
    {
        "query": "À quel emploi est nommée Madame Nadia Ferchichi au ministère de la jeunesse et des sports ?",
        "relevant": "Madame Nadia Ferchichi, manager en chef en sport, se voit attribuer la classe exceptionnelle à l'emploi de chef de service d'administration centrale au ministère de la jeunesse et des sports, par arrêté du 6 mai 2026.",
        "irrelevant": "Madame Samia Goumani épouse El Badri se voit attribuer la classe exceptionnelle à l'emploi de directeur d'administration centrale au ministère de la santé."
    },
    {
        "query": "Qui se voit mettre fin à ses fonctions au commissariat régional de la jeunesse de Siliana ?",
        "relevant": "Il est mis fin aux fonctions de Monsieur Marouen Marzouk, professeur principal émérite d'éducation physique, en qualité de chef de bureau du développement des sports et de l'éducation physique à l'unité des activités sportives au commissariat régional de la jeunesse de Siliana, par arrêté du 7 mai 2026.",
        "irrelevant": "Monsieur Ayechi Jdidi est nommé membre représentant la commune de Tunis au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie."
    },
    {
        "query": "Quel arrêté de 2017 est modifié par la ministre de la famille le 14 mai 2026 ?",
        "relevant": "L'arrêté de la ministre de la famille du 14 mai 2026 modifie l'arrêté de la ministre de la femme, de la famille et de l'enfance du 22 mars 2017 fixant le régime de formation, des études et de sortie du cycle de formation des inspecteurs de la jeunesse et de l'enfance.",
        "irrelevant": "L'arrêté du ministre de la santé du 14 mai 2026 abroge l'arrêté du 2 septembre 2025 fixant la liste des substituts du lait maternel."
    },
    {
        "query": "Qui est la ministre de la famille ayant signé l'arrêté du 14 mai 2026 sur la formation des inspecteurs ?",
        "relevant": "L'arrêté du 14 mai 2026 portant modification du régime de formation des inspecteurs de la jeunesse et de l'enfance a été signé par la ministre de la famille, de la femme, de l'enfance et des personnes âgées Asma Jabri.",
        "irrelevant": "L'arrêté du 15 mai 2026 sur la protection des monuments historiques a été signé par la ministre des affaires culturelles Amina Srarfi."
    },
    {
        "query": "Sur quel décret de 2019 se fonde l'arrêté sur la formation des inspecteurs de la jeunesse ?",
        "relevant": "L'arrêté de la ministre de la famille du 14 mai 2026 se fonde notamment sur le décret gouvernemental n°2019-920 du 26 septembre 2019 portant statut particulier des membres du corps de l'inspection pédagogique du ministère des affaires de la jeunesse et du sport et du ministère de la femme, de la famille, de l'enfance et des personnes âgées.",
        "irrelevant": "L'arrêté du ministre de la santé du 14 mai 2026 se fonde sur le décret n°84-1314 du 3 novembre 1984 fixant les attributions de la commission nationale pour la promotion de l'alimentation du nourrisson."
    },
    {
        "query": "Quels articles de l'arrêté de 2017 sont abrogés par le nouvel arrêté du 14 mai 2026 ?",
        "relevant": "L'arrêté du 14 mai 2026 abroge et remplace les dispositions des articles 2, 5, le premier tiret de l'article 7 et le premier alinéa de l'article 11 de l'arrêté du 22 mars 2017.",
        "irrelevant": "L'arrêté du ministre de la santé du 14 mai 2026 abroge toutes les dispositions antérieures contraires, notamment l'arrêté du 2 septembre 2025 fixant la liste des substituts du lait maternel."
    },
    {
        "query": "Quel article de l'arrêté de 2017 détermine les candidats pouvant s'inscrire au cycle de formation des inspecteurs ?",
        "relevant": "L'article 2 (nouveau) de l'arrêté du 14 mai 2026 précise que peuvent s'inscrire au cycle de formation des inspecteurs de la jeunesse et de l'enfance les candidats admis au concours d'entrée mentionné à l'article 40 du décret gouvernemental n°2019-920 du 26 septembre 2019.",
        "irrelevant": "L'article 5 (nouveau) de l'arrêté du 14 mai 2026 fixe la durée de la formation à deux ans avec un mois de congé annuel pour chaque année."
    },
    {
        "query": "Quel pourcentage de l'horaire de formation est consacré aux sciences humaines et sociales ?",
        "relevant": "Selon l'article 7 (premier tiret nouveau), trente-cinq pour cent (35%) de l'horaire global de la formation est consacré au domaine théorique portant sur les sciences humaines et sociales, les approches pédagogiques et fondées sur les droits, les méthodologies et les problématiques de l'enfance.",
        "irrelevant": "La formation au cycle de formation des inspecteurs de la jeunesse et de l'enfance dure deux ans au cours de laquelle les participants bénéficient d'un mois de congé annuel par année."
    },
    {
        "query": "Qui préside la commission pédagogique créée auprès du centre de formation des inspecteurs ?",
        "relevant": "Selon l'article 11 (premier alinéa nouveau), la commission pédagogique créée auprès du centre de formation est présidée par le directeur général du centre et composée de deux inspecteurs de la jeunesse et de l'enfance et d'un représentant de la direction de l'inspection pédagogique.",
        "irrelevant": "La commission nationale du patrimoine réunie le 13 février 2026 a émis un avis sur la protection des monuments historiques et archéologiques en Tunisie."
    },
    {
        "query": "Quel code juridique encadre la protection des monuments historiques et archéologiques en Tunisie ?",
        "relevant": "Les arrêtés de la ministre des affaires culturelles du 15 mai 2026 se fondent sur le code du patrimoine archéologique, historique et des arts traditionnels promulgué par la loi n°94-35 du 24 février 1994, modifié et complété par la loi n°2001-118 du 6 décembre 2001 et le décret-loi n°2011-43 du 25 mai 2011.",
        "irrelevant": "L'arrêté du ministre de la santé du 14 mai 2026 se fonde sur la loi n°83-24 du 4 mars 1983 relative aux substituts du lait maternel."
    },
    {
        "query": "Qui est la ministre des affaires culturelles signataire des arrêtés du 15 mai 2026 ?",
        "relevant": "La ministre des affaires culturelles Amina Srarfi a signé les deux arrêtés du 15 mai 2026 relatifs à la protection des monuments historiques et archéologiques, visés par la Cheffe du Gouvernement Sarra Zaafrani Zenzri.",
        "irrelevant": "La ministre de la famille, de la femme, de l'enfance et des personnes âgées Asma Jabri a signé l'arrêté du 14 mai 2026 modifiant le régime de formation des inspecteurs de la jeunesse."
    },
    {
        "query": "Quel arrêté de 2021 est modifié par le premier arrêté de la ministre des affaires culturelles du 15 mai 2026 ?",
        "relevant": "Le premier arrêté de la ministre des affaires culturelles du 15 mai 2026 modifie l'arrêté du ministre des affaires culturelles par intérim du 21 janvier 2021 relatif à la protection des monuments historiques et archéologiques.",
        "irrelevant": "Le second arrêté de la ministre des affaires culturelles du 15 mai 2026 modifie l'arrêté de la ministre des affaires culturelles du 15 avril 2022 relatif à la protection des monuments historiques et archéologiques."
    },
    {
        "query": "Quel arrêté de 2022 est modifié par le second arrêté de la ministre des affaires culturelles du 15 mai 2026 ?",
        "relevant": "Le second arrêté de la ministre des affaires culturelles du 15 mai 2026 modifie l'arrêté de la ministre des affaires culturelles du 15 avril 2022 relatif à la protection des monuments historiques et archéologiques.",
        "irrelevant": "Le premier arrêté de la ministre des affaires culturelles du 15 mai 2026 modifie l'arrêté du ministre des affaires culturelles par intérim du 21 janvier 2021."
    },
    {
        "query": "Dans quelle délégation se trouve l'Escargotière de Henchir el Magtaâ ?",
        "relevant": "L'Escargotière de Henchir el Magtaâ est située dans la délégation de Gafsa nord, selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026.",
        "irrelevant": "La Maison punique de Gammarth est située dans la délégation de la Marsa, selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026."
    },
    {
        "query": "Quel est le numéro du plan TPD de l'Escargotière de Henchir el Magtaâ ?",
        "relevant": "L'Escargotière de Henchir el Magtaâ est délimitée selon les plans des travaux particuliers divers TPD n°94943 du 13 octobre 2020.",
        "irrelevant": "La Maison punique de Gammarth est délimitée selon le plan TPD n°113851 du 19 mai 2025."
    },
    {
        "query": "Quel est le rayon de protection de l'Escargotière de Henchir el Magtaâ ?",
        "relevant": "L'Escargotière de Henchir el Magtaâ bénéficie d'un rayon de protection de 500 mètres aux abords, selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026.",
        "irrelevant": "La Maison punique de Gammarth bénéficie d'un rayon de protection de 200 mètres aux abords, selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026."
    },
    {
        "query": "Dans quelle délégation se trouve la Maison punique de Gammarth ?",
        "relevant": "La Maison punique de Gammarth est située dans la délégation de la Marsa, selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026.",
        "irrelevant": "L'Escargotière de Henchir el Magtaâ est située dans la délégation de Gafsa nord, selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026."
    },
    {
        "query": "Quel est le numéro du titre foncier de la Maison punique de Gammarth ?",
        "relevant": "La Maison punique de Gammarth correspond au titre foncier n°124041 Tunis, d'une superficie de 292 m² représentant la parcelle (A).",
        "irrelevant": "L'Escargotière de Henchir el Magtaâ a une superficie de 5000 m² selon les limites indiquées sur les plans TPD n°94943 du 13 octobre 2020."
    },
    {
        "query": "Quel est le numéro du plan TPD de la Maison punique de Gammarth ?",
        "relevant": "La Maison punique de Gammarth est indiquée sur le plan des travaux particuliers divers TPD n°113851 du 19 mai 2025.",
        "irrelevant": "L'Escargotière de Henchir el Magtaâ est délimitée selon les plans TPD n°94943 du 13 octobre 2020."
    },
    {
        "query": "Quelle commission a émis un avis sur la protection des monuments historiques en 2026 ?",
        "relevant": "La commission nationale du patrimoine a émis un avis lors de sa réunion du 13 février 2026, sur lequel se fondent les arrêtés de la ministre des affaires culturelles du 15 mai 2026 relatifs à la protection des monuments historiques.",
        "irrelevant": "La commission nationale pour la promotion de l'alimentation du nourrisson et de l'enfant a émis un avis lors de sa réunion du 29 août 2029, fondant l'arrêté du ministre de la santé sur les substituts du lait maternel."
    },
    {
        "query": "Quel décret régit la commission nationale du patrimoine ?",
        "relevant": "Le décret n°94-1475 du 4 juillet 1994 est relatif à la composition et au fonctionnement de la commission nationale du patrimoine, visé dans les arrêtés de la ministre des affaires culturelles du 15 mai 2026.",
        "irrelevant": "Le décret n°84-1314 du 3 novembre 1984 fixe les attributions, la composition et le mode de fonctionnement de la commission nationale pour la promotion de l'alimentation du nourrisson et de l'enfant."
    },
    {
        "query": "Quel est le grade d'Hamouda Gabsi chargé d'un emploi fonctionnel au ministère de l'emploi ?",
        "relevant": "Hamouda Gabsi est administrateur en chef et est chargé des fonctions de directeur de la planification à la direction générale de la planification, la programmation et des projets au ministère de l'emploi et de la formation professionnelle.",
        "irrelevant": "Abderrazek Bouafif est administrateur général de l'éducation et est chargé des fonctions de directeur d'appui des activités de formation des entreprises au ministère de l'emploi."
    },
    {
        "query": "À quelle direction est rattaché l'emploi fonctionnel d'Hamouda Gabsi ?",
        "relevant": "Hamouda Gabsi est chargé des fonctions de directeur de la planification à la direction générale de la planification, la programmation et des projets au ministère de l'emploi et de la formation professionnelle.",
        "irrelevant": "Abderrazek Bouafif est chargé des fonctions de directeur d'appui des activités de formation à la direction générale de la formation continue et de développement des compétences."
    },
    {
        "query": "Quel est le grade d'Abderrazek Bouafif chargé d'un emploi fonctionnel au ministère de l'emploi ?",
        "relevant": "Abderrazek Bouafif est administrateur général de l'éducation et est chargé des fonctions de directeur d'appui des activités de formation des entreprises et l'habilitation des individus à la direction générale de la formation continue et de développement des compétences.",
        "irrelevant": "Hamouda Gabsi est administrateur en chef et est chargé des fonctions de directeur de la planification à la direction générale de la planification, la programmation et des projets."
    },
    {
        "query": "Quel arrêté nomme les cadres chargés des emplois fonctionnels au ministère de l'emploi ?",
        "relevant": "C'est l'arrêté du ministre de l'emploi et de la formation professionnelle du 7 mai 2026 qui nomme Hamouda Gabsi et Abderrazek Bouafif à leurs emplois fonctionnels respectifs.",
        "irrelevant": "C'est l'arrêté du ministre de l'agriculture des ressources hydrauliques et de la pêche maritime du 7 mai 2026 qui nomme Nahla Bououni et Temime Horri dans le grade d'analyste en chef."
    },
    {
        "query": "Quel ministère relève Madame Amel Hadded selon l'arrêté du 11 mai 2026 ?",
        "relevant": "L'arrêté du 11 mai 2026 nommant Madame Amel Hadded au conseil d'établissement de l'Office de Rjim Maâtoug est un arrêté du ministre de la défense nationale.",
        "irrelevant": "L'arrêté du 8 mai 2026 nommant Madame Amel Abida au conseil d'administration de la Compagnie des phosphates de Gafsa est un arrêté du ministre chargé du ministère de l'industrie des mines et de l'énergie."
    },
    {
        "query": "Quel est le nom complet de l'Office où Madame Amel Hadded est nommée ?",
        "relevant": "Madame Amel Hadded est nommée au conseil d'établissement de l'Office de Rjim Maâtoug pour le développement du Sud et du Sahara.",
        "irrelevant": "Madame Amel Abida est nommée au conseil d'administration de la Compagnie des phosphates de Gafsa."
    },
    {
        "query": "Quelle est la qualité de Madame Amel Hadded au conseil d'établissement de l'Office de Rjim Maâtoug ?",
        "relevant": "Madame Amel Hadded est nommée membre représentant de l'Office de développement du Sud au conseil d'établissement de l'Office de Rjim Maâtoug pour le développement du Sud et du Sahara.",
        "irrelevant": "Madame Amel Abida est nommée administrateur représentant la Banque centrale de Tunisie au conseil d'administration de la Compagnie des phosphates de Gafsa."
    },
    {
        "query": "À quelle date prend effet la nomination de Madame Amel Hadded à l'Office de Rjim Maâtoug ?",
        "relevant": "La nomination de Madame Amel Hadded au conseil d'établissement de l'Office de Rjim Maâtoug prend effet à compter du 1er mai 2026.",
        "irrelevant": "La nomination du professeur Salem Yahiaoui au conseil d'administration de l'hôpital Bechir Hamza prend effet à compter du 30 mars 2026."
    },
    {
        "query": "Quelle société Voltalia a signé la convention de Menzel El Habib par rapport à celle de Segdoud ?",
        "relevant": "La convention de Menzel El Habib a été signée avec la société Voltalia SA, tandis que la convention de Segdoud a été signée avec la société Voltalia (sans la mention SA), les deux étant des entités liées au même groupe Voltalia.",
        "irrelevant": "La convention de la centrale photovoltaïque d'El Khobna a été signée avec la société Qair International, et celle de la centrale El Ksar également avec Qair International."
    },
    {
        "query": "Quelles centrales photovoltaïques approuvées le 15 mai 2026 sont situées dans le gouvernorat de Sidi Bouzid ?",
        "relevant": "Deux centrales photovoltaïques situées dans le gouvernorat de Sidi Bouzid ont été approuvées le 15 mai 2026 : la centrale d'El Khobna (loi 2026-9) avec Qair International et la centrale de Mezzouna (loi 2026-10) avec Scatec ASA.",
        "irrelevant": "Deux centrales photovoltaïques situées dans le gouvernorat de Gafsa ont été approuvées le 15 mai 2026 : la centrale El Ksar (loi 2026-11) et la centrale de Segdoud (loi 2026-12)."
    },
    {
        "query": "Quelles centrales photovoltaïques approuvées le 15 mai 2026 sont situées dans le gouvernorat de Gafsa ?",
        "relevant": "Deux centrales photovoltaïques situées dans le gouvernorat de Gafsa ont été approuvées le 15 mai 2026 : la centrale El Ksar (loi 2026-11) avec Qair International et la centrale de Segdoud (loi 2026-12) avec Voltalia.",
        "irrelevant": "La centrale photovoltaïque de Mezzouna au gouvernorat de Sidi Bouzid (loi 2026-10) a été signée avec la société Scatec ASA le 24 mars 2025."
    },
    {
        "query": "Quelle est la loi tunisienne de 1983 sur les substituts du lait maternel ?",
        "relevant": "La loi n°83-24 du 4 mars 1983 est relative au contrôle de la qualité, à la commercialisation et à l'information sur l'utilisation des substituts du lait maternel et produits apparentés, notamment son article 4 qui fonde la compétence du ministre pour fixer la liste de ces substituts.",
        "irrelevant": "La loi n°94-35 du 24 février 1994 est le code du patrimoine archéologique, historique et des arts traditionnels, fondement des arrêtés de protection des monuments historiques."
    },
    {
        "query": "Quel est l'ISSN du Journal Officiel de la République Tunisienne ?",
        "relevant": "L'ISSN du Journal Officiel de la République Tunisienne est le 0330.7921, tel qu'indiqué dans le bas de page du numéro 49 du 15 mai 2026.",
        "irrelevant": "Le numéro 49 du Journal Officiel de la République Tunisienne a été déposé au siège du gouvernorat de Tunis le 15 mai 2026."
    },
    {
        "query": "Depuis quelle année le Journal Officiel de la République Tunisienne est-il publié ?",
        "relevant": "Le Journal Officiel de la République Tunisienne est publié depuis 1860, comme indiqué sur la page de couverture du numéro 49 du 15 mai 2026 qui mentionne 'since 1860 منذ'.",
        "irrelevant": "Le JORT du 15 mai 2026 est le numéro 49 de la 169ème année, correspondant au vendredi 28 dhoulkaâda 1447."
    },
    {
        "query": "Quelle est la superficie de la Maison punique de Gammarth ?",
        "relevant": "La Maison punique de Gammarth a une superficie de 292 m² du titre foncier n°124041 Tunis, correspondant à la parcelle (A), selon l'arrêté de la ministre des affaires culturelles du 15 mai 2026.",
        "irrelevant": "L'Escargotière de Henchir el Magtaâ a une superficie de 5000 m² selon les plans TPD n°94943 du 13 octobre 2020."
    },
    {
        "query": "Quel est le numéro et la date de la loi approuvant la centrale photovoltaïque de Mezzouna ?",
        "relevant": "La loi n°2026-10 du 15 mai 2026 porte approbation de la convention de concession de production d'électricité et ses annexes de la centrale photovoltaïque de Mezzouna au gouvernorat de Sidi Bouzid, signée avec la société Scatec ASA.",
        "irrelevant": "La loi n°2026-9 du 15 mai 2026 porte approbation de la convention de la centrale photovoltaïque d'El Khobna au gouvernorat de Sidi Bouzid, signée avec Qair International."
    },
    {
        "query": "Qui certifie la conformité du Journal Officiel de la République Tunisienne ?",
        "relevant": "Le président directeur général de l'I.O.R.T. (Imprimerie Officielle de la République Tunisienne) certifie la conformité du Journal Officiel de la République Tunisienne.",
        "irrelevant": "Le Président de la République Kaïs Saïed promulgue les lois publiées dans le Journal Officiel de la République Tunisienne."
    },
    {
        "query": "Quelle est la qualité de Monsieur Mohamed Amine Zoueghi au conseil d'administration de l'hôpital Habib Bougatfa ?",
        "relevant": "Monsieur Mohamed Amine Zoueghi est nommé membre représentant de la partie syndicale la plus représentative au conseil d'administration de l'hôpital Habib Bougatfa de Bizerte.",
        "irrelevant": "Monsieur Ayechi Jdidi est nommé membre représentant la commune de Tunis au conseil d'administration de l'Institut national El Mongi Ben Hmida de neurologie de Tunis."
    }
]
results = []

for name, model_name in models.items():
    print(f"\nLoading {name}...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    gaps = []
    sim_rels = []
    sim_irrs = []

    start = time.time()
    for pair in pairs:
        q   = model.encode([pair["query"]])
        rel = model.encode([pair["relevant"]])
        irr = model.encode([pair["irrelevant"]])

        sim_rel = cosine_similarity(q, rel)[0][0]
        sim_irr = cosine_similarity(q, irr)[0][0]

        gaps.append(sim_rel - sim_irr)
        sim_rels.append(sim_rel)
        sim_irrs.append(sim_irr)

    elapsed = round(time.time() - start, 2)

    results.append({
        "Modèle":            name,
        "Dimensions":        dim,
        "Sim Pertinent ↑":   round(sum(sim_rels) / len(sim_rels), 4),
        "Sim Irrelevant ↓":  round(sum(sim_irrs) / len(sim_irrs), 4),
        "Gap Moyen ↑":       round(sum(gaps) / len(gaps), 4),
        "Temps (s)":         elapsed
    })

    del model  

df = pd.DataFrame(results)

# ✅ Affichage console
print("\n📊 Résultats de comparaison:\n")
print(df.to_string(index=False))

# ✅ Sauvegarde dans results.txt
output_path = "results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("   BENCHMARK COMPARAISON MODELES EMBEDDING - LexTN\n")
    f.write("   Source: JORT N°49 du 15 mai 2026\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Nombre de paires testées : {len(pairs)}\n")
    f.write(f"Modèles testés : {', '.join(models.keys())}\n\n")
    f.write("-" * 70 + "\n")
    f.write(df.to_string(index=False))
    f.write("\n\n" + "-" * 70 + "\n")
    f.write("Légende:\n")
    f.write("  Sim Pertinent ↑  : score cosinus moyen query/texte pertinent (plus élevé = mieux)\n")
    f.write("  Sim Irrelevant ↓ : score cosinus moyen query/texte irrelevant (plus bas = mieux)\n")
    f.write("  Gap Moyen ↑      : différence moyenne (pertinent - irrelevant) (plus élevé = mieux)\n")
    f.write("  Temps (s)        : temps total d'encodage des 20 paires\n")
    f.write("=" * 70 + "\n")

print(f"\n✅ Résultats sauvegardés dans : {output_path}")