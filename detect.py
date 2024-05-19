import os

import cv2
import numpy as np
import time

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils.utils import load_model

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from utils.data_traitement import load_data
import pickle


def infer_image(image_path, model_path):
    """
    Effectue une inférence sur une image en utilisant le modèle donné.

    Paramètres :
        image_path (str) : Chemin vers l'image à inférer.
        model_path (str) : Chemin vers le fichier de poids du modèle.

    Retourne :
        None
    """

    # Load num_classes and labels
    pkl = model_path.split('.h5')[0] + "_info.pkl"
    with open(pkl, 'rb') as file:
        info = pickle.load(file)
    labels = info['labels']
    
        # Construction du modèle complet
    model = load_model(model_path)
    start_time = time.time()

    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    class_names = labels.names
    predicted_label = class_names[predicted_class]

    end_time = time.time()
    inference_time = end_time - start_time
    print("Temps d'inférence:", inference_time, "secondes")
    print(predicted_label)

    return predicted_label

if __name__ == '__main__':
    print(infer_image("D:\\EPSI\\Bachelor\\B3\\MSPR\\1\\IA-from-scratch\\data\\test\\bear\\1.jpg", "./weights/modelv2/modelv2.h5"))
