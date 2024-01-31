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
    Performs inference on an image using the given model.

    Parameters:
        image_path (str): Path to the image to infer.
        model_path (str): Path to the model weights file.

    Returns:
        None
    """
    image = load_data(img_path=image_path)
    model = load_model(model_path)

    # Perform inference
    predictions = model.predict(image)

    # Interpret the results
    class_index = np.argmax(predictions)
    classes = {0: 'not a cat', 1: 'cat'}
    predicted_class = classes[class_index]
    clear_terminal()
    print(f"Prediction for image {image_path}: {predicted_class}")
    return predicted_class


if __name__ == "__main__":
    infer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\cat\\0.jpg", "weights/model.tf")
    infer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\no-cat\\0a1a5a2140.jpg", "weights/model.tf")
