# backend/train_recommender.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

df = pd.read_csv('recommender_dataset.csv')
y = df['recommendation_label'].values
X = df.drop(columns=['recommendation_label']).values

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_onehot = np.eye(len(np.unique(y_enc)))[y_enc]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=60, batch_size=16, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test)
print('Recommender test acc:', acc)

model.save('recommender.h5')
joblib.dump(scaler, 'recommender_scaler.save')
joblib.dump(le, 'recommender_label_encoder.save')