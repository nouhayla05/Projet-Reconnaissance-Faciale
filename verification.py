import numpy as np
from pathlib import Path

# Charger la bibliothèque complète
database = np.load(
    "embeddings/embeddings_database.npz"
)

print("\n========== VERIFICATION GENERALE ==========")

total = 0

for student_id in sorted(database.files):

    # Charger l'embedding moyen
    mean_embedding = database[student_id]

    # Charger les embeddings individuels
    file_path = Path(
        f"embeddings/{student_id}_embeddings.npy"
    )

    if not file_path.exists():
        print(f"Fichier manquant : {student_id}")
        continue

    embeddings = np.load(file_path)

    total += len(embeddings)

    # Vérifier les dimensions
    dimensions_ok = (
        embeddings.ndim == 2
        and embeddings.shape[1] == 512
        and mean_embedding.shape == (512,)
    )

    # Vérifier les valeurs
    values_ok = (
        np.isfinite(embeddings).all()
        and np.isfinite(mean_embedding).all()
    )

    if not dimensions_ok or not values_ok:
        print(f"\n{student_id} : donnees invalides")
        continue

    # Vérifier la normalisation
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = np.linalg.norm(mean_embedding)

    # Calculer les distances cosinus
    distances = 1 - (embeddings @ mean_embedding)

    print(f"\n========== {student_id} ==========")
    print("Nombre d'embeddings :", len(embeddings))
    print("Dimensions :", embeddings.shape)
    print("Norme moyenne :", round(float(mean_norm), 6))
    print("Normes individuelles valides :",
          np.allclose(norms, 1.0, atol=1e-4))

    print("Distance minimale :",
          round(float(distances.min()), 4))

    print("Distance maximale :",
          round(float(distances.max()), 4))

    print("Distance moyenne :",
          round(float(distances.mean()), 4))

print("\n===================================")
print("Nombre d'etudiants :", len(database.files))
print("Total embeddings :", total)
print("VERIFICATION TERMINEE")