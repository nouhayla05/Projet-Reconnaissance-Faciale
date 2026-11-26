# Rapport technique — Ingénierie des données

**Projet :** Système de contrôle d'assiduité basé sur la reconnaissance faciale

**Rôle :** Ingénieure données

## 1. Objectif

L'objectif de cette partie est de préparer les données faciales nécessaires au développement d'un système intelligent de contrôle d'assiduité.

Le travail consiste à collecter les images, détecter et recadrer les visages, extraire leurs caractéristiques numériques et construire une bibliothèque d'embeddings destinée au module de reconnaissance faciale.

## 2. Collecte des données

Un premier dataset expérimental a été constitué à partir de 25 photographies d'une même personne, identifiée par le code ID001.

Les images ont été capturées à l'aide d'une webcam et de la bibliothèque OpenCV.

Elles ont été enregistrées dans le dossier `data/raw/ID001`.

## 3. Prétraitement des images

Le prétraitement a été réalisé à l'aide de MTCNN (Multi-task Cascaded Convolutional Networks).

Cette méthode permet de détecter les visages et de préparer les régions faciales pour leur traitement par le modèle de reconnaissance.

Les images ont été recadrées et redimensionnées à 160 × 160 pixels.

**Résultats obtenus :**

- Images analysées : 25.
- Images traitées : 25.
- Images rejetées : 0.

## 4. Extraction des embeddings

L'extraction des caractéristiques faciales a été réalisée avec un modèle InceptionResnetV1 préentraîné sur VGGFace2, disponible dans la bibliothèque facenet-pytorch.

Chaque visage a été transformé en un vecteur numérique de 512 dimensions.

Les embeddings ont ensuite été normalisés.

Un embedding moyen a été calculé afin de construire une représentation de référence pour l'identifiant ID001.

**Résultats :**

- Nombre d'embeddings : 25.
- Dimensions : (25, 512).
- Dimensions de l'embedding moyen : (512,).

## 5. Vérification technique

Une vérification a été effectuée afin de contrôler la validité numérique des embeddings.

Les résultats ont confirmé que les vecteurs possèdent les dimensions attendues, ne contiennent pas de valeurs numériques invalides et présentent des normes proches de 1.

## 6. Analyse de la qualité

Une analyse basée sur la distance cosinus a été réalisée pour mesurer la proximité des embeddings individuels avec leur représentation moyenne.

La distance moyenne initialement observée était d'environ 0,0937.

Deux images, image_5.jpg et image_7.jpg, ont présenté des distances relativement élevées dans l'analyse de qualité et nécessitent une inspection visuelle.

Les différences constatées entre certaines exécutions doivent également être vérifiées avant validation définitive.

## 7. Livrables réalisés

Les principaux fichiers produits sont :

- `collecte.py` : collecte des photographies.
- `preprocessing.py` : détection et préparation des visages.
- `extraction.py` : extraction des embeddings.
- `verification.py` : vérification des vecteurs.
- `analyse_qualite.py` : analyse des distances.
- `ID001_embeddings.npy` : embeddings individuels.
- `embeddings_database.npz` : embedding moyen.
- `rapport_qualite.csv` : résultats de l'analyse de qualité.

## 8. Limites et perspectives

Le travail réalisé constitue un premier prototype utilisant les photographies d'une seule personne.

Les prochaines étapes consistent à compléter l'alignement géométrique, valider la qualité des images atypiques, constituer un jeu de test indépendant et étendre progressivement le dataset aux autres étudiants volontaires.

La reconnaissance faciale devra ensuite être évaluée sur plusieurs identités, dans différentes conditions de capture.

## 9. Conclusion

Cette partie du projet a permis de mettre en place une première chaîne de traitement des données faciales, depuis la collecte des photographies jusqu'à la génération des embeddings.

Les résultats confirment le bon fonctionnement technique du prototype sur les 25 images expérimentales. Des validations supplémentaires restent nécessaires avant son utilisation dans un système réel de contrôle d'assiduité.