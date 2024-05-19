import os
import datetime
import pickle
# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.utils import clear_terminal
from utils.data_traitement import load_data

from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def train_model(epochs=10, batch_size=8, weight_name="model", learning_rate=0.01, augmentation=0.0):
    ds, labels, num_classes = load_data(augmentation=augmentation)
    now = datetime.now()
    date_time = now.strftime("%m/%d%Y_%Hh_%Mm_%Ss")

    checkpoint_path = "training"+str(date_time)+"_cp.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    weight_path = "./weights/" + weight_name
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    else:
        i = 1
        while os.path.exists(weight_path):
            weight_path = weight_path.split("_")[0] + "_" + str(i)
    i = 1

    train_images, train_labels, val_images, val_labels = ds["train_images"], ds["train_labels"], ds["val_images"], ds["val_labels"]
    num_classes = ds["num_classes"]

    # Convertir les étiquettes en one-hot encoded
    train_labels = tf.keras.utils.to_categorical(train_labels)
    val_labels = tf.keras.utils.to_categorical(val_labels)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Construction du modèle complet
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Geler les poids du modèle de base (CNN pré-entraîné)
    base_model.trainable = False
    weight_names = weight_path + "/" + weight_name + ".h5"
    # ModelCheckpoint
    checkpoint = ModelCheckpoint(weight_names, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Compiler le modèle
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[checkpoint, cp_callback])

    info = {
        'labels': labels,
        'num_classes': num_classes,
        'hist' : hist
    }

    with open(weight_names.split(".h5")[0] + "_info.pkl", 'wb') as file:
        pickle.dump(info, file)

    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(weight_path + "/accuracy_plot.png")

    print("\n \n \n \n The best model has been saved in the file " + weight_names + ".h5")

    return hist, model, weight_names + ".h5"


if __name__ == "__main__":
    hist, model = train_model(10, 8, "modelv2", 0.01, 0.0)
