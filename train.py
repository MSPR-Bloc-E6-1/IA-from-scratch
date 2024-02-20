import os
import datetime

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.utils import clear_terminal

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.optimizers import SGD, Adam
from utils.data_traitement import load_data

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import Model

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(epoch=5, batch_size=32, weight_name="model", learning_rate=0.01, augmentation=0.0):
    weight_name = "weights/" + weight_name
    i = 1
    while os.path.exists(weight_name+".h5"):
        weight_name = weight_name[:-3] + "_" + str(i)
        i += 1
    clear_terminal()
    X_train, y_train = load_data(type="train", augmentation=augmentation)
    X_val, y_val = load_data(type="val")

    # Charger le modèle VGG16 pré-entraîné sans la couche fully connected
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Congeler les couches pré-entraînées
    for layer in base_model.layers:
        layer.trainable = False

    # Ajouter des couches convolutionnelles personnalisées
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    # Ajouter des couches fully connected personnalisées
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(14, activation='softmax')(x)

    model = Model(base_model.input, x)

    # Compiler le modèle
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Ajouter le callback ModelCheckpoint
    checkpoint = ModelCheckpoint(weight_name + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    clear_terminal()
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])

    print("\n \n \n \n The best model has been saved in the file " + weight_name + ".h5")

if __name__ == "__main__":
    train_model(epoch=10)
