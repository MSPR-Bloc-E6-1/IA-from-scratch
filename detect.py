import os

# Désactiver les options d'optimisation OneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils.utils import clear_terminal

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import load_model

from utils.data_traitement import data_load


def inferer_image(image_path, model_path):
    """
    Effectue une inférence sur une image à l'aide du modèle donné.

    Parameters:
        image_path (str): Chemin vers l'image à inférer.
        model_path (str): Chemin vers le fichier de poids du modèle.

    Returns:
        None
    """
    image = data_load(img_path = image_path)
    model = load_model(model_path)

    # Effectuer l'inférence
    predictions = model.predict(image)

    # Interpréter les résultats
    class_index = np.argmax(predictions)
    classes = {0: 'pas un chat', 1: 'chat'}
    predicted_class = classes[class_index]
    clear_terminal()
    print(f"Prédiction pour l'image {image_path}: {predicted_class}")
    return predicted_class


if __name__ == "__main__":
    inferer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\cat\\0.jpg", "weights/model.tf")
    inferer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\no-cat\\0a1a5a2140.jpg", "weights/model.tf")