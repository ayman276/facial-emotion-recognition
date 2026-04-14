import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48

def load_data(data_dir):
    images = []
    labels = []

    for label, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(data_dir, emotion)

        if not os.path.exists(emotion_dir):
            print(f"Dossier introuvable : {emotion_dir}")
            continue

        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)

            # Lire en couleur RGB pour VGG16
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Redimensionner
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convertir BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


def preprocess_data():
    print("Chargement des données train...")
    X_train, y_train = load_data("../data/raw/train")

    print("Chargement des données test...")
    X_test, y_test = load_data("../data/raw/test")

    # Normaliser entre 0 et 1
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0

    # Shape finale : (nb_images, 48, 48, 3)
    print(f"Shape X_train : {X_train.shape}")
    print(f"Shape X_test  : {X_test.shape}")

    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=len(EMOTIONS))
    y_test  = to_categorical(y_test,  num_classes=len(EMOTIONS))

    print(f"Train : {X_train.shape[0]} images")
    print(f"Test  : {X_test.shape[0]} images")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Prétraitement terminé !")