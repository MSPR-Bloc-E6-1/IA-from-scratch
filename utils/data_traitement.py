import os

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.utils import *
from utils.data_augmentation import data_augmentation
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
import numpy as np

_data_folder = "data"

def _is_image_file(file_path):
    try:
        with Image.open(file_path):
            return True
    except:
        return False

def _check_data_folder():
    """Check if the data folder is present and contains the necessary files.

    Returns:
        Bool: True if the data folder is present and contains the necessary files, False otherwise.
    """
    train_folder = os.path.join(_data_folder, "train")
    test_folder = os.path.join(_data_folder, "test")
    val_folder = os.path.join(_data_folder, "val")
    classes = ['background', 'beaver', 'cat', 'dog', 'coyote', 'squirrel', 'rabbit', 'wolf', 'lynx', 'bear', 'puma', 'rat', 'raccoon', 'fox']

    for class_name in classes:
        class_train_folder = os.path.join(train_folder, class_name)
        class_test_folder = os.path.join(test_folder, class_name)
        class_val_folder = os.path.join(val_folder, class_name)

        if not os.path.exists(class_train_folder) or not os.path.isdir(class_train_folder) or len(os.listdir(class_train_folder)) == 0:
            return False
        if not os.path.exists(class_test_folder) or not os.path.isdir(class_test_folder) or len(os.listdir(class_test_folder)) == 0:
            return False
        if not os.path.exists(class_val_folder) or not os.path.isdir(class_val_folder) or len(os.listdir(class_val_folder)) == 0:
            return False

    return True

def _destructure_data(path):
    """Move all data (images) that are subfolders of path to the root of path and delete the subfolders.

    Args:
        path (str): Path to the folder containing the organized data.
    
    Returns:
        Bool: True if the data has been disorganized.
    """
    
    def move_files(src, dest):
        for root, dirs, files in os.walk(src):
            for file in files:
                file_path = os.path.join(root, file)
                if _is_image_file(file_path):
                    shutil.move(file_path, os.path.join(dest, file))
                else:
                    os.remove(file_path)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        if os.path.isdir(file_path):
            move_files(file_path, path)
            try:
                if not _is_image_file(file_path):
                    shutil.rmtree(file_path)
            except OSError:
                # The folder could not be deleted, may be non-empty
                pass

    return True


def _organize_data(data_disorganized_path):
    """Organize the data in the data folder.

    Args:
        data_disorganized_path (str): Path to the folder containing the disorganized data.

    Returns:
        Bool: True if the data has been organized.
    """
    # Check if data_disorganized_path exists:

    if not os.path.exists(data_disorganized_path):
        raise ValueError(f"The path {data_disorganized_path} could not be found.")

    _data_folder = "data"
    train_folder = os.path.join(_data_folder, "train")
    test_folder = os.path.join(_data_folder, "test")
    val_folder = os.path.join(_data_folder, "val")

    if not os.path.exists(_data_folder):
        os.mkdir(_data_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)

    classes = ['background', 'beaver', 'cat', 'dog', 'coyote', 'squirrel', 'rabbit', 'wolf', 'lynx', 'bear', 'puma', 'rat', 'raccoon', 'fox']

    for class_name in classes:
        class_train_folder = os.path.join(train_folder, class_name)
        class_test_folder = os.path.join(test_folder, class_name)
        class_val_folder = os.path.join(val_folder, class_name)

        if not os.path.exists(class_train_folder):
            os.mkdir(class_train_folder)
        if not os.path.exists(class_test_folder):
            os.mkdir(class_test_folder)
        if not os.path.exists(class_val_folder):
            os.mkdir(class_val_folder)

        class_path = os.path.join(data_disorganized_path, class_name)
        _destructure_data(class_path)

        try:
            class_train, class_test = train_test_split(os.listdir(class_path), test_size=0.2, random_state=42)
            class_train, class_val = train_test_split(class_train, test_size=0.2, random_state=42)
        except Exception as e:
            raise ValueError(f"Please separate the data into {class_name} folders")

        for file in class_train:
            try:
                shutil.move(os.path.join(class_path, file), class_train_folder)
            except shutil.Error as e:
                pass
        for file in class_test:
            try:
                shutil.move(os.path.join(class_path, file), class_test_folder)
            except shutil.Error as e:
                pass
        for file in class_val:
            try:
                shutil.move(os.path.join(class_path, file), class_val_folder)
            except shutil.Error as e:
                pass

    return True


def _data_normalized(path_data):
    classes = ['background', 'beaver', 'cat', 'dog', 'coyote', 'squirrel', 'rabbit', 'wolf', 'lynx', 'bear', 'puma', 'rat', 'raccoon', 'fox']
    images = []
    labels = []

    for i, class_name in enumerate(classes):
        class_images, class_labels = load_images(os.path.join(path_data, class_name), class_name)
        images.append(class_images)
        labels.append(class_labels)

    X = np.concatenate(images, axis=0)
    Y = np.concatenate(labels, axis=0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)

    y_encoded = to_categorical(y_encoded, num_classes=len(classes))

    # Normalisation par z-score
    X_normalized = (X - np.mean(X)) / np.std(X)
    X_resize = np.array([resize(image, (128, 128)) for image in X_normalized])

    return X_resize, y_encoded




def load_data(type=None, img_path=None, augmentation=0.0):
    if type is not None:
        if not _check_data_folder():
            data_disorganized_path = input("The data is not organized in the correct format.\nChoose a path where your data is separated into an animals folders: \n")
            _organize_data(data_disorganized_path)
        if type != "train" and type != "test" and type != "val":
            raise ValueError("The type must be 'train', 'test' or 'val'")
        path_data = os.path.join(_data_folder, type)
        if type == "train":
            if augmentation > 0.0:
                total_data_count = 0
                for file in os.listdir(path_data):
                    total_data_count += len(os.listdir(os.path.join(path_data, file)))
                augmentation_count = int(total_data_count * augmentation)
                data_augmentation(path_data, augmentation_count)

        X, y = _data_normalized(path_data)
        return X, y
    else:
        if not os.path.exists(img_path):
            raise ValueError(f"The path {img_path} could not be found.")
        return load_image(img_path)
