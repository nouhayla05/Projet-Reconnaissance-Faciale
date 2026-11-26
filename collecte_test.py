import cv2
from pathlib import Path

# 1. Identifiant de la personne
student_id = input(
    "Entrez l'identifiant (ID001 a ID007) : "
).strip().upper()

# 2. Vérifier l'identifiant
valid_ids = [f"ID{i:03d}" for i in range(1, 8)]

if student_id not in valid_ids:
    raise ValueError("Identifiant invalide")

# 3. Dossier des images de test
folder = Path("data/test") / student_id
folder.mkdir(parents=True, exist_ok=True)

# 4. Vérifier les images existantes
existing = list(folder.glob("test_*.jpg"))

if existing:
    print("Ce dossier contient deja des images de test.")
    print("Aucune image ne sera remplacee.")
    raise SystemExit

# 5. Ouvrir la webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

count = 0
target = 5

print("\nS ou ESPACE : prendre une photo")
print("Q : quitter")

# 6. Capturer les images
while count < target:

    ret, frame = cap.read()

    if not ret:
        print("Erreur de capture")
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

    cv2.imshow("Collecte des images de test", preview)

    key = cv2.waitKey(30) & 0xFF

    if key in [ord("s"), ord("S"), 32]:

        filename = folder / f"test_{count + 1}.jpg"

        if cv2.imwrite(str(filename), frame):
            count += 1
            print(f"Test {count}/{target} enregistre")

    elif key in [ord("q"), ord("Q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nCollecte terminee : {count} images")
print("Dossier :", folder.resolve())