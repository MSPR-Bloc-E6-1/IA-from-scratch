import os

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

def load_images(directory, label):
    images = []
    labels = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)

        try:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(file_path)
                image = image.convert('RGB')
                image = image.resize((224, 224))
                image = img_to_array(image)
                images.append(image)
                labels.append(label)
            else:
                print(f"The file {file_path} is not in a supported image format.")
        except Exception as e:
            print(f"Error processing image {file_path}: {str(e)}")

    return np.array(images), np.array(labels)

def load_image(path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = image / 255.0
        image_input = np.expand_dims(image, axis=0)
        return image_input


def clear_terminal():
    """
    Clears the terminal, works on Windows, Linux, and macOS.
    """
    os_name = os.name

    if os_name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux and macOS
        os.system('clear')


def load_model(model_path = None):
    """
    Load a model from a file.

    Parameters:
        model_path (str): The path to the model file.

    Returns:
        tensorflow.keras.models.Sequential: The model.
    """
    pkl = model_path.split('.h5')[0] + "_info.pkl"
    with open(pkl, 'rb') as file:
        info = pickle.load(file)
    num_classes = info['num_classes']
    labels = info['labels']
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    base_model.trainable = False
    model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    if model_path is not None:
        model.load_weights(model_path)
    return model
