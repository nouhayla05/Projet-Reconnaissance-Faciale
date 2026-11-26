from pathlib import Path
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import csv

# =====================================
# 1. CONFIGURATION
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Processeur utilise :", device)

# Charger FaceNet
model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)

# Dossiers
input_folder = Path("data/processed")
output_folder = Path("embeddings")
report_folder = Path("livrables")

output_folder.mkdir(parents=True, exist_ok=True)
report_folder.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. INITIALISATION
# =====================================

database = {}
report = []

total_embeddings = 0

# =====================================
# 3. PARCOURIR LES ETUDIANTS
# =====================================

for student_folder in sorted(input_folder.iterdir()):

    if not student_folder.is_dir():
        continue

    student_id = student_folder.name

    print(f"\nTraitement de {student_id}")

    student_embeddings = []

    # =====================================
    # 4. PARCOURIR LES IMAGES
    # =====================================

    for image_path in sorted(student_folder.glob("*.jpg")):

        try:
            # Charger l'image
            image = Image.open(image_path).convert("RGB")

            # Convertir en tableau NumPy
            image_array = np.array(
                image,
                dtype=np.float32
            )

            # Normalisation pour FaceNet
            image_array = (
                image_array - 127.5
            ) / 128.0

            # Conversion en tenseur PyTorch
            image_tensor = torch.from_numpy(
                image_array
            ).permute(2, 0, 1)

            image_tensor = (
                image_tensor.unsqueeze(0).to(device)
            )

            # Extraction de l'embedding
            with torch.no_grad():
                embedding = model(image_tensor)

            embedding = embedding.cpu().numpy()[0]

            # Normalisation L2
            norm = np.linalg.norm(embedding)

            if norm < 1e-12:
                raise ValueError("Embedding nul")

            embedding = embedding / norm

            # Vérification
            if not np.isfinite(embedding).all():
                raise ValueError("Embedding invalide")

            student_embeddings.append(embedding)

            total_embeddings += 1

            print(f"Embedding extrait : {image_path.name}")

        except Exception as e:
            print(f"Erreur : {image_path.name} - {e}")

    # =====================================
    # 5. CALCUL DE L'EMBEDDING MOYEN
    # =====================================

    if len(student_embeddings) == 0:
        print(f"Aucun embedding pour {student_id}")
        continue

    student_embeddings = np.array(
        student_embeddings,
        dtype=np.float32
    )

    mean_embedding = np.mean(
        student_embeddings,
        axis=0
    )

    # Normalisation de la moyenne
    norm = np.linalg.norm(mean_embedding)

    if norm < 1e-12:
        print(f"Moyenne invalide pour {student_id}")
        continue

    mean_embedding = mean_embedding / norm

    # =====================================
    # 6. SAUVEGARDE
    # =====================================

    np.save(
        output_folder / f"{student_id}_embeddings.npy",
        student_embeddings
    )

    database[student_id] = mean_embedding

    report.append([
        student_id,
        len(student_embeddings),
        512,
        "OK"
    ])

    print(
        f"{student_id} : "
        f"{len(student_embeddings)} embeddings generes"
    )

# =====================================
# 7. SAUVEGARDE DE LA BASE COMPLETE
# =====================================

if not database:
    raise RuntimeError("Aucun embedding genere")

np.savez(
    output_folder / "embeddings_database.npz",
    **database
)

# =====================================
# 8. RAPPORT CSV
# =====================================

report_path = report_folder / "rapport_embeddings.csv"

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Etudiant",
        "Nombre_embeddings",
        "Dimensions",
        "Statut"
    ])

    writer.writerows(report)

# =====================================
# 9. RESULTATS
# =====================================

print("\n================================")
print("EXTRACTION FACENET TERMINEE")
print("================================")

print("Nombre d'etudiants :", len(database))
print("Total embeddings :", total_embeddings)

print("\nEtudiants enregistres :")

for student_id in database:
    print(
        student_id,
        database[student_id].shape
    )

print("\nBibliotheque sauvegardee avec succes")