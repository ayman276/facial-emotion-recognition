import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from preprocessing import preprocess_data
from model import build_model

def train():
    # 1. Charger les données
    X_train, X_test, y_train, y_test = preprocess_data()

    # 2. Construire le modèle
    model = build_model()

    # 3. Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath='../models/best_model.h5',  # corrigé
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    # 4. Entraîner
    print("Début de l'entraînement...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # 5. Évaluer
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy finale : {accuracy * 100:.2f}%")

    return model, history


if __name__ == "__main__":
    train()