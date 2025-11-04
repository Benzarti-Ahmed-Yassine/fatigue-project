# backend/feature_extraction.py
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# Params
IMAGE_SIZE = 128
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def extract_features_from_bgr(img_bgr):
    """Retourne un vecteur de features (HOG + LBP hist + géométriques) ou None si pas de visage."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # prendre le visage le plus large
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (IMAGE_SIZE, IMAGE_SIZE))

    # HOG
    hog_feat = hog(face_resized,
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)

    # LBP + hist
    lbp = local_binary_pattern(face_resized, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # Géométriques
    eyes = eye_cascade.detectMultiScale(face_resized, scaleFactor=1.1, minNeighbors=5)
    mouths = mouth_cascade.detectMultiScale(face_resized, scaleFactor=1.3, minNeighbors=20)
    nb_eyes = len(eyes)
    nb_mouths = len(mouths)
    brightness = np.mean(face_resized) / 255.0
    eye_open_ratio = 0.0
    if nb_eyes >= 1:
        h_vals = [eh for (ex, ey, ew, eh) in eyes]
        w_vals = [ew for (ex, ey, ew, eh) in eyes]
        eye_open_ratio = np.mean(h_vals) / (np.mean(w_vals) + 1e-7)
    mouth_area_ratio = 0.0
    if nb_mouths >= 1:
        mx, my, mw, mh = mouths[0]
        mouth_area_ratio = (mw * mh) / (w * h + 1e-7)

    # Normaliser HOG
    hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-7)

    feature_vector = np.concatenate([
        hog_feat,
        lbp_hist,
        np.array([nb_eyes, nb_mouths, brightness, eye_open_ratio, mouth_area_ratio], dtype=float)
    ])
    return feature_vector