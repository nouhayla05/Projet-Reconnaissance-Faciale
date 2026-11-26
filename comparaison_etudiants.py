import numpy as np
import pandas as pd
from pathlib import Path

# Charger les embeddings moyens
database = np.load(
    "embeddings/embeddings_database.npz"
)

# Récupérer les identifiants
student_ids = sorted(database.files)

# Construire la matrice
embeddings = np.array([
    database[student_id]
    for student_id in student_ids
])

# Normaliser par sécurité
norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

embeddings = embeddings / np.maximum(norms, 1e-12)

# Calculer les similarités cosinus
similarity_matrix = embeddings @ embeddings.T

# Convertir en distances cosinus
distance_matrix = 1 - similarity_matrix

# Corriger les petites erreurs numériques
distance_matrix = np.maximum(distance_matrix, 0)
np.fill_diagonal(distance_matrix, 0)

# Créer un tableau
df = pd.DataFrame(
    distance_matrix,
    index=student_ids,
    columns=student_ids
)

print("\n===== MATRICE DES DISTANCES COSINUS =====")
print(df.round(4).to_string())

# Enregistrer le résultat
output_folder = Path("livrables")
output_folder.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output_folder / "matrice_distances.csv"
)

# Identifier les deux références les plus proches
matrix = distance_matrix.copy()

np.fill_diagonal(matrix, np.inf)

i, j = np.unravel_index(
    np.argmin(matrix),
    matrix.shape
)

print("\n===== REFERENCES LES PLUS PROCHES =====")

print("Etudiant 1 :", student_ids[i])
print("Etudiant 2 :", student_ids[j])
print("Distance :", round(float(matrix[i, j]), 4))

print("\nAnalyse terminee")