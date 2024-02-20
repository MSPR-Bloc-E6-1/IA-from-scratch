import os

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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
                image = image.resize((64, 64))
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
        image = image.resize((64, 64))
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
