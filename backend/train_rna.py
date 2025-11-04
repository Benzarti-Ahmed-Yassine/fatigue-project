# backend/train_rna.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Hyperparams
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Charger features.csv
df = pd.read_csv('features.csv')
if 'label' not in df.columns:
    raise ValueError("Le fichier features.csv doit contenir une colonne 'label'.")

X = df.drop(columns=['label']).values
y = df['label'].values

# Encodage labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(np.unique(y_enc))
y_onehot = np.eye(num_classes)[y_enc]

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Modèle MLP
model = Sequential([
    Dense(256, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entraînement
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Évaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# Sauvegarde
model.save('rna_fatigue.h5')
joblib.dump(scaler, 'scaler.save')
joblib.dump(le, 'label_encoder.save')
print("Modèle, scaler et label encoder sauvegardés.")