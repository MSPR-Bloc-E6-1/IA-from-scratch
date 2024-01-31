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
    depo = "metrics/"+model_path.split("/")[-1]
    depo = depo.split(".")[0]
    if not os.path.exists(depo):
        os.makedirs(depo)

    X, y_test = load_data(type="test")
    
    model = load_model(model_path)

    # Predictions on test data
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)

    results = {}

    # Calculate accuracy
    if 'accuracy' in metrics:
        accuracy = np.sum(y_pred_classes == np.argmax(y_test, axis=1)) / len(y_test)
        results['accuracy'] = accuracy

    # Calculate confusion matrix
    if 'confusion_matrix' in metrics:
        conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
        results['confusion_matrix'] = conf_matrix
        if save_cm:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No-Chat', 'Chat'], yticklabels=['No-Chat', 'Chat'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predictions')
            plt.ylabel('True Labels')
            plt.savefig(depo+'confusion_matrix.png')

    # Calculate classification report
    if 'classification_report' in metrics:
        class_report = classification_report(np.argmax(y_test, axis=1), y_pred_classes, target_names=['No-Chat', 'Chat'])
        results['classification_report'] = class_report

    # Save textual metrics in a file
    if save_txt:
        with open(depo+'metrics.txt', 'w') as file:
            for metric, value in results.items():
                file.write(f'{metric}: {value}\n')
    clear_terminal()
    print(f"The metrics have been saved in the folder {depo}.")

if __name__ == "__main__":
    evaluate_model("weights/model.tf", metrics=['accuracy', 'confusion_matrix', 'classification_report'])
