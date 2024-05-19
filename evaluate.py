import os

# Disable OneDNN optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.utils import clear_terminal

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.data_traitement import load_data

def evaluate_model(model_path, metrics=['accuracy', 'confusion_matrix', 'classification_report'], save_cm=True, save_txt=True):
    """
    Evaluate the model on test data.

    Parameters:
        model (tensorflow.keras.models.Sequential): The model to evaluate.
        X (numpy.ndarray): The test data.
        y_test (numpy.ndarray): The test labels.
        metrics (list): The metrics to calculate. Options: ['accuracy', 'confusion_matrix', 'classification_report']
        save_cm (bool): If True, save the plots in a specific folder.
        save_txt (bool): If True, save the textual metrics in a text file.

    Returns:
        None
    """
    model_path = model_path.split('/')[-1]
    depo = "metrics/"+model_path.split("/")[-1]+"/"
    depo = depo.split(".")[0]
    if not os.path.exists(depo):
        os.makedirs(depo)

    df, _, _ = load_data()

    X_test, y_test = df["test_images"], df["test_labels"]

    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_test_encoded = to_categorical(y_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_encoded, axis=1)

    accuracy = np.mean(y_pred_classes == y_true_classes)
    results = {}

    if 'accuracy' in metrics:
        results['accuracy'] = accuracy
        print("Accuracy:", accuracy)

    if 'confusion_matrix' in metrics:
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        results['confusion_matrix'] = cm
        if save_cm:
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(depo+'/confusion_matrix.png')

    if 'classification_report' in metrics:
        cr = classification_report(y_true_classes, y_pred_classes)
        results['classification_report'] = cr
        print(cr)


    if save_txt:
        with open(depo+'/metrics.txt', 'w') as file:
            for metric, value in results.items():
                file.write(f'{metric}: {value}\n')
    clear_terminal()
    print(f"The metrics have been saved in the folder {depo}.")

if __name__ == "__main__":
    evaluate_model("./weights/modelv2/modelv2.h5", metrics=['accuracy', 'confusion_matrix', 'classification_report'])
