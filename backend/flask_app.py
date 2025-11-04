# backend/flask_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from feature_extraction import extract_features_from_bgr
from utils import read_image_from_base64
import os

app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis le frontend

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'rna_fatigue.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.save')
LE_PATH = os.path.join(BASE_DIR, 'label_encoder.save')
RECOMMENDER_PATH = os.path.join(BASE_DIR, 'recommender.h5')

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LE_PATH)

recommender = None
recommender_scaler = None
recommender_le = None
if os.path.exists(RECOMMENDER_PATH):
    try:
        from tensorflow.keras.models import load_model
        recommender = load_model(RECOMMENDER_PATH)
        recommender_scaler = joblib.load(os.path.join(BASE_DIR, 'recommender_scaler.save'))
        recommender_le = joblib.load(os.path.join(BASE_DIR, 'recommender_label_encoder.save'))
    except Exception as e:
        print("Erreur chargement recommender:", e)
        recommender = None

RECOMMENDATION_MAP = {
    'reposé': "Aucune action requise",
    'fatigué': "Conseil : pause courte (10-15 min), hydratation",
    'tres_fatigue': "Alerte : Intervention requise, repos immédiat"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'Image (base64) non fournie'}), 400

    b64 = data['image']
    img = read_image_from_base64(b64)
    if img is None:
        return jsonify({'error': 'Image invalide'}), 400

    feat = extract_features_from_bgr(img)
    if feat is None:
        return jsonify({'error': 'Aucun visage détecté'}), 400

    feat_scaled = scaler.transform([feat])
    preds = model.predict(feat_scaled)
    probs = preds[0].tolist()
    idx = int(np.argmax(preds))
    label = label_encoder.inverse_transform([idx])[0]
    prob = float(preds[0][idx])

    recommendation = RECOMMENDATION_MAP.get(label, "Aucune recommandation")
    if recommender is not None:
        rec_input = np.concatenate([feat, probs])
        rec_input_scaled = recommender_scaler.transform([rec_input])
        rec_preds = recommender.predict(rec_input_scaled)
        rec_idx = int(np.argmax(rec_preds))
        recommendation = recommender_le.inverse_transform([rec_idx])[0]

    return jsonify({
        'label': str(label),
        'probability': prob,
        'probabilities': probs,
        'recommendation': recommendation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)