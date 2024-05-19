import os

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.utils import *
from utils.data_augmentation import data_augmentation
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import numpy as np
import cv2
from datasets import load_dataset
from sklearn.model_selection import train_test_split

_data_folder = "data"


def _organize_data(data_disorganized_path, classes = ['background', 'beaver', 'cat', 'dog', 'coyote', 'squirrel', 'rabbit', 'wolf', 'lynx', 'bear', 'puma', 'rat', 'raccoon', 'fox'], split_size =[0.8,0.15,0.05], random_state = 42):
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

        try:
            class_train, class_test_val = train_test_split(os.listdir(class_path), test_size=(1-split_size[0]), random_state=random_state)
            class_test, class_val = train_test_split(class_test_val, test_size=split_size[2]/(split_size[1]+split_size[2]), random_state=random_state)
            class_train, class_val = train_test_split(class_train, test_size=split_size[1]/(split_size[0]+split_size[1]), random_state=random_state)
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


def convert_dataset_to_arrays(dataset):
    images = []
    labels = []
    for example in dataset:
        image_path = example['image'].filename
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)  # Convert image array to float32 data type
        image /= 255.0
        images.append(image)
        labels.append(example['label'])

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    return np.array(images), np.array(labels), num_classes

def _load_data_and_aug(augmentation):
    ds = load_dataset("imagefolder", data_dir=_data_folder)
    path_data_train = os.path.join(_data_folder, "train")

    if augmentation > 0.0:
        total_data_count = 0
        for file in os.listdir(path_data_train):
            total_data_count += len(os.listdir(os.path.join(path_data_train, file)))
        augmentation_count = int(total_data_count * augmentation)
        data_augmentation(path_data_train, augmentation_count)
        ds = load_dataset("imagefolder", data_dir=_data_folder)
    return ds

def load_data(img_path=None, augmentation=0.0):
    if img_path is not None:
        if not os.path.exists(img_path):
            raise ValueError(f"The path {img_path} could not be found.")
        return load_image(img_path)
    else:
        try:
            ds = _load_data_and_aug(augmentation)
        except :
            data_disorganized_path = input("The data is not organized in the correct format.\nChoose a path where your data is separated into an animals folders: \n")
            _organize_data(data_disorganized_path)
            ds = _load_data_and_aug(augmentation)


        labels = ds["train"].features["label"]
        train_images, train_labels, num_classes = convert_dataset_to_arrays(ds['train'])
        val_images, val_labels, _ = convert_dataset_to_arrays(ds['validation'])
        test_images, test_labels, _ = convert_dataset_to_arrays(ds['test'])
        df = {"train_images": train_images, "train_labels" : train_labels, "val_images" : val_images, "val_labels" : val_labels, "test_images" : test_images, "test_labels" : test_labels, "num_classes" : num_classes}

        return df, labels, num_classes
