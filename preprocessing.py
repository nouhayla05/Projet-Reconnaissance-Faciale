from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import csv

# 1. Configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=False,
    device=device
)

raw_folder = Path("data/raw")
processed_folder = Path("data/processed")

processed_folder.mkdir(parents=True, exist_ok=True)

# 2. Statistiques
total_success = 0
total_failed = 0
report = []

# 3. Parcourir les étudiants
for student_folder in sorted(raw_folder.iterdir()):

    if not student_folder.is_dir():
        continue

    student_id = student_folder.name

    output_folder = processed_folder / student_id
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nTraitement de {student_id}")

    # 4. Parcourir les images
    for image_path in sorted(student_folder.glob("*.jpg")):

        try:
            image = Image.open(image_path).convert("RGB")

            # Détecter les visages
            boxes, probs = mtcnn.detect(image)

            if boxes is None or len(boxes) != 1:
                print(f"Image rejetee : {image_path.name}")
                total_failed += 1

                report.append([
                    student_id,
                    image_path.name,
                    "Rejetee"
                ])
                continue

            # Extraire le visage
            output_path = output_folder / image_path.name

            face = mtcnn(
                image,
                save_path=str(output_path)
            )

            if face is None:
                total_failed += 1
                report.append([
                    student_id,
                    image_path.name,
                    "Rejetee"
                ])
                continue

            total_success += 1

            report.append([
                student_id,
                image_path.name,
                "Acceptee"
            ])

            print(f"Image traitee : {image_path.name}")

        except Exception as e:
            total_failed += 1
            print(f"Erreur : {image_path.name} - {e}")

            report.append([
                student_id,
                image_path.name,
                "Erreur"
            ])

# 5. Sauvegarder le rapport
report_path = Path("livrables/rapport_preprocessing.csv")
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Etudiant",
        "Image",
        "Statut"
    ])

    writer.writerows(report)

# 6. Résumé
print("\n==============================")
print("PRETRAITEMENT TERMINE")
print("Images traitees :", total_success)
print("Images rejetees :", total_failed)
print("==============================")