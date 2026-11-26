import numpy as np
import torch
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Detection et preparation du visage
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=True,
    device=device
)

# Charger FaceNet
model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)

# Charger les 7 references
database = np.load(
    "embeddings/embeddings_database.npz"
)

student_ids = sorted(database.files)

# Seuil provisoire
threshold = 0.4

# Images de test ID001
test_folder = Path("data/test/ID001")

if not test_folder.exists():
    raise RuntimeError("Dossier de test introuvable")

for image_path in sorted(test_folder.glob("*.jpg")):

    image = Image.open(image_path).convert("RGB")

    boxes, probs = mtcnn.detect(image)

    if boxes is None or len(boxes) != 1:
        print(f"{image_path.name} : visage non valide")
        continue

    face = mtcnn(image)

    if face is None:
        print(f"{image_path.name} : visage non detecte")
        continue

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face)

    embedding = embedding.cpu().numpy()[0]

    embedding = embedding / max(
        np.linalg.norm(embedding),
        1e-12
    )

    # Comparaison avec les 7 personnes
    distances = {}

    for student_id in student_ids:

        reference = database[student_id]

        distance = 1 - np.dot(
            embedding,
            reference
        )

        distances[student_id] = float(distance)

    # Identifier la meilleure correspondance
    best_id = min(
        distances,
        key=distances.get
    )

    best_distance = distances[best_id]

    # Decision
    if best_distance < threshold:
        result = best_id
    else:
        result = "INCONNU"

    print("\n-----------------------------")
    print("Image :", image_path.name)
    print("Distances :", {
        k: round(v, 4)
        for k, v in distances.items()
    })
    print("Meilleure correspondance :", best_id)
    print("Distance minimale :", round(best_distance, 4))
    print("Resultat :", result)

print("\nTEST TERMINE")