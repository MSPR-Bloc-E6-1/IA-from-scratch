import os

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils.utils import clear_terminal

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import load_model

from utils.data_traitement import load_data


def infer_image(image_path, model_path):
    """
    Effectue une inférence sur une image en utilisant le modèle donné.

    Paramètres :
        image_path (str) : Chemin vers l'image à inférer.
        model_path (str) : Chemin vers le fichier de poids du modèle.

    Retourne :
        None
    """
    image = load_data(img_path=image_path)
    model = load_model(model_path)

    predictions = model.predict(image)

    class_index = np.argmax(predictions)
    classes = {0: 'background', 1: 'beaver', 2: 'cat', 3: 'dog', 4: 'coyote', 5: 'squirrel', 6: 'rabbit', 7: 'wolf', 8: 'lynx', 9: 'bear', 10: 'puma', 11: 'rat', 12: 'raccoon', 13: 'fox'}
    predicted_class = classes[class_index]
    clear_terminal()
    print(f"Prédiction pour l'image {image_path} : {predicted_class}")
    return predicted_class