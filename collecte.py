import cv2
from pathlib import Path

# Identifiant de la personne
student_id = input(
    "Entrez l'identifiant (ex: ID002) : "
).strip().upper()

if not student_id.startswith("ID") or not student_id[2:].isdigit():
    raise ValueError("Identifiant invalide")

# Dossier de sauvegarde
folder = Path("data/raw") / student_id
folder.mkdir(parents=True, exist_ok=True)

# Chercher les images existantes
existing_numbers = []

for path in folder.glob("image_*.jpg"):
    try:
        number = int(path.stem.split("_")[1])
        existing_numbers.append(number)
    except (ValueError, IndexError):
        pass

next_number = max(existing_numbers, default=0) + 1

print(f"Images deja presentes : {len(existing_numbers)}")

# Ouvrir la webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

count = 0
target = 25

print("\nCOMMANDES :")
print("S ou ESPACE : prendre une photo")
print("Q ou ECHAP : quitter")

while count < target:

    ret, frame = cap.read()

    if not ret:
        print("Erreur de lecture webcam")
        break

    preview = frame.copy()

    cv2.putText(
        preview,
        f"{student_id} : {count}/{target}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        preview,
        "S / ESPACE = PHOTO | Q = QUITTER",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Collecte des images", preview)

    key = cv2.waitKey(30) & 0xFF

    # S, s ou ESPACE
    if key in [ord("s"), ord("S"), 32]:

        filename = folder / f"image_{next_number}.jpg"

        success = cv2.imwrite(str(filename), frame)

        if success:
            count += 1
            next_number += 1

            print(f"Photo {count}/{target} enregistree")
            print(f"Fichier : {filename}")
        else:
            print("ERREUR : photo non enregistree")

    # Q, q ou ECHAP
    elif key in [ord("q"), ord("Q"), 27]:
        print("Collecte interrompue")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nCollecte terminee : {count} nouvelles photos")
print(f"Dossier : {folder.resolve()}")