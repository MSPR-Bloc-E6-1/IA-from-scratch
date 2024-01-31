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
    train_nocat_file = os.path.join(train_folder, "no-cat")
    train_cat_file = os.path.join(train_folder, "cat")
    test_nocat_file = os.path.join(test_folder, "no-cat")
    test_cat_file = os.path.join(test_folder, "cat")

    if os.path.exists(_data_folder) and os.path.isdir(_data_folder):
        if os.path.exists(train_folder) and os.path.isdir(train_folder):
            if os.path.exists(test_folder) and os.path.isdir(test_folder):
                if os.path.exists(train_nocat_file) and os.path.isdir(train_nocat_file) and len(os.listdir(train_nocat_file)) > 0:
                    if os.path.exists(train_cat_file) and os.path.isdir(train_cat_file) and len(os.listdir(train_cat_file)) > 0:
                        if os.path.exists(test_nocat_file) and os.path.isdir(test_nocat_file) and len(os.listdir(test_nocat_file)) > 0:
                            if os.path.exists(test_cat_file) and os.path.isdir(test_cat_file) and len(os.listdir(test_cat_file)) > 0:
                                return True

    return False

import os
import shutil

import os
import shutil

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
    train_nocat_file = os.path.join(train_folder, "no-cat")
    train_cat_file = os.path.join(train_folder, "cat")
    test_nocat_file = os.path.join(test_folder, "no-cat")
    test_cat_file = os.path.join(test_folder, "cat")

    if not os.path.exists(_data_folder):
        os.mkdir(_data_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(train_nocat_file):
        os.mkdir(train_nocat_file)
    if not os.path.exists(train_cat_file):
        os.mkdir(train_cat_file)
    if not os.path.exists(test_nocat_file):
        os.mkdir(test_nocat_file)
    if not os.path.exists(test_cat_file):
        os.mkdir(test_cat_file)

    nocat_path = os.path.join(data_disorganized_path, "no-cat")
    cat_path = os.path.join(data_disorganized_path, "cat")

    _destructure_data(nocat_path)
    _destructure_data(cat_path)

    try:
        nocat_train, nocat_test = train_test_split(os.listdir(nocat_path), test_size=0.2, random_state=42)
        cat_train, cat_test = train_test_split(os.listdir(cat_path), test_size=0.2, random_state=42)
    except Exception as e:
        raise ValueError("Please separate the data into 'cat' and 'no-cat' folders")

    for file in nocat_train:
        try:
            shutil.move(os.path.join(nocat_path, file), train_nocat_file)
        except shutil.Error as e:
            pass
    for file in cat_train:
        try:
            shutil.move(os.path.join(cat_path, file), train_cat_file)
        except shutil.Error as e:
            pass

    for file in nocat_test:
        try:
            shutil.move(os.path.join(nocat_path, file), test_nocat_file)
        except shutil.Error as e:
            pass
    for file in cat_test:
        try:
            shutil.move(os.path.join(cat_path, file), test_cat_file)
        except shutil.Error as e:
            pass

    return True


def _data_normalized(path_data):
    images_no_cat, labels_no_cat = load_images(os.path.join(path_data, "no-cat"), "no-cat")

    images_cat, labels_cat = load_images(os.path.join(path_data, "cat"), "cat")

    X = np.concatenate((images_no_cat, images_cat), axis=0)
    Y = np.concatenate((labels_no_cat, labels_cat), axis=0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)

    y_encoded = to_categorical(y_encoded, num_classes=2)

    X_normalized = X / 255.0

    return X_normalized, y_encoded



def load_data(type=None, img_path=None, augmentation=0.0):
    if type is not None:
        if not _check_data_folder():
            data_disorganized_path = input("The data is not organized in the correct format.\nChoose a path where your data is separated into 'cat' and 'no-cat' folders: \n")
            _organize_data(data_disorganized_path)
        if type != "train" and type != "test":
            raise ValueError("The type must be 'train' or 'test'")
        path_data = os.path.join(_data_folder, type)
        if type == "train":
            if augmentation > 0.0:
                # Number of images in the subfolders of path_data:
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
