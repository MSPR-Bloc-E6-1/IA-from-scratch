import os

# Désactiver les options d'optimisation OneDNN
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

from utils.data_traitement import data_load

def evaluate_model(model_path, metrics=['accuracy', 'confusion_matrix', 'classification_report'], save_cm=True, save_txt=True):
    """
    Évalue le modèle sur les données de test.

    Parameters:
        model (tensorflow.keras.models.Sequential): Le modèle à évaluer.
        X (numpy.ndarray): Les données de test.
        y_test (numpy.ndarray): Les étiquettes de test.
        metrics (list): Les métriques à calculer. Options : ['accuracy', 'confusion_matrix', 'classification_report']
        save_cm (bool): Si True, sauvegarde les graphiques dans un dossier spécifique.
        save_txt (bool): Si True, enregistre les métriques textuelles dans un fichier texte.

    Returns:
        None
    """
    depo = "metrics/"+model_path.split("/")[-1]
    depo = depo.split(".")[0]
    if not os.path.exists(depo):
        os.makedirs(depo)

    X, y_test = data_load(type="test")
    
    model = load_model(model_path)

    # Prédictions sur les données de test
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)

    results = {}

    # Calcul de l'exactitude (accuracy)
    if 'accuracy' in metrics:
        accuracy = np.sum(y_pred_classes == np.argmax(y_test, axis=1)) / len(y_test)
        results['accuracy'] = accuracy

    # Calcul de la matrice de confusion
    if 'confusion_matrix' in metrics:
        conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
        results['confusion_matrix'] = conf_matrix
        if save_cm:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No-Chat', 'Chat'], yticklabels=['No-Chat', 'Chat'])
            plt.title('Matrice de Confusion')
            plt.xlabel('Prédictions')
            plt.ylabel('Vraies étiquettes')
            plt.savefig(depo+'confusion_matrix.png')

    # Calcul du rapport de classification
    if 'classification_report' in metrics:
        class_report = classification_report(np.argmax(y_test, axis=1), y_pred_classes, target_names=['No-Chat', 'Chat'])
        results['classification_report'] = class_report

    # Enregistrement des métriques textuelles dans un fichier
    if save_txt:
        with open(depo+'metrics.txt', 'w') as file:
            for metric, value in results.items():
                file.write(f'{metric}: {value}\n')
    clear_terminal()
    print(f"Les métriques ont été enregistrées dans le dossier {depo}.")

if __name__ == "__main__":
    evaluate_model("weights/model.tf", metrics=['accuracy', 'confusion_matrix', 'classification_report'])
