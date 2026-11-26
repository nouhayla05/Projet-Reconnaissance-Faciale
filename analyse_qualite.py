import numpy as np
import pandas as pd
from pathlib import Path

# Dossiers
embeddings_folder = Path("embeddings")
processed_folder = Path("data/processed")
output_folder = Path("livrables")

output_folder.mkdir(parents=True, exist_ok=True)

# Charger les embeddings moyens
database = np.load(
    embeddings_folder / "embeddings_database.npz"
)

results = []

# Parcourir les 7 étudiants
for student_id in sorted(database.files):

    mean_embedding = database[student_id]

    # Charger les embeddings individuels
    embeddings = np.load(
        embeddings_folder / f"{student_id}_embeddings.npy"
    )

    # Récupérer les images
    images = sorted(
        (processed_folder / student_id).glob("*.jpg")
    )

    if len(images) != len(embeddings):
        raise ValueError(
            f"Nombre d'images incorrect pour {student_id}"
        )

    # Calculer les distances cosinus
    distances = 1 - (embeddings @ mean_embedding)

    for image, distance in zip(images, distances):

        results.append({
            "Etudiant": student_id,
            "Image": image.name,
            "Distance_cosinus": float(distance)
        })

# Créer le DataFrame
df = pd.DataFrame(results)

# Trier les distances
df = df.sort_values(
    by="Distance_cosinus",
    ascending=False
)

# Enregistrer le rapport
df.to_csv(
    output_folder / "rapport_qualite_global.csv",
    index=False
)

print("\n===== ANALYSE DE QUALITE GLOBALE =====")

print(df.head(15).to_string(index=False))

print("\nNombre total d'images :", len(df))

print("Distance moyenne :", round(
    df["Distance_cosinus"].mean(), 4
))

print("Distance maximale :", round(
    df["Distance_cosinus"].max(), 4
))

print("\nRapport sauvegarde avec succes")