# backend/build_features.py
import os
import cv2
import pandas as pd
from feature_extraction import extract_features_from_bgr

IMG_DIR = 'images'  # Dossier avec images: repose_001.jpg, fatigue_002.jpg, etc.
OUT_CSV = 'features.csv'

if not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"Dossier {IMG_DIR} introuvable. Crée-le et ajoute des images.")

rows = []
files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for f in files:
    path = os.path.join(IMG_DIR, f)
    img = cv2.imread(path)
    if img is None:
        print(f"Impossible de lire {f}")
        continue

    feat = extract_features_from_bgr(img)
    if feat is None:
        print(f"Pas de visage dans {f}, ignoré")
        continue

    # Extraire label du nom de fichier
    name = f.lower()
    if name.startswith('repos'):
        label = 'reposé'
    elif name.startswith('fatigu'):
        label = 'fatigué'
    elif name.startswith('tres') or name.startswith('very'):
        label = 'tres_fatigue'
    else:
        print(f"Label inconnu pour {f}, ignoré")
        continue

    row = list(feat) + [label]
    rows.append(row)

if not rows:
    raise ValueError("Aucune feature extraite. Vérifie les images et les noms.")

feat_len = len(rows[0]) - 1
cols = [f'feat_{i}' for i in range(feat_len)] + ['label']
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUT_CSV, index=False)
print(f"{len(df)} échantillons sauvegardés dans {OUT_CSV}")