import os

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.utils import clear_terminal

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from utils.data_traitement import load_data

def train_model(epoch=5, batch_size=32, weight_name="model", learning_rate=0.01, augmentation = 0.0):
    weight_name = "weights/" + weight_name
    i = 1
    while os.path.exists(weight_name):
        weight_name = weight_name[:-3] + "_" + str(i)
        i += 1
    clear_terminal()
    X, y = load_data(type="train", augmentation=augmentation)

    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Use SGD optimizer with a learning rate of 0.01 (adjust if necessary)
    optimizer = SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    clear_terminal()
    model.fit(X, y, epochs=epoch, batch_size=batch_size, validation_split=0.2)

    model.save(weight_name + ".tf")
    print("\n \n \n \n The model has been saved in the file " + weight_name + ".tf")


if __name__ == "__main__":
    train_model(epoch=10)
