# Classification de Chats et no_cats

Ce projet vise à réaliser une classification d'images de chats et non-chats. Les test on été effectué sur des datasets qui peuvent être téléchargé à partir des lien suivants : [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=download) pour les chats et [Archive.org](https://archive.org/details/CAT_DATASET) pour les non-chats.

## Prérequis

Assurez-vous d'avoir les éléments suivants installés sur votre environnement :

- imgaug==0.4.0
- matplotlib==3.6.3
- numpy==1.24.1
- opencv_python==4.7.0.68
- Pillow==10.2.0
- scikit_learn==1.3.2
- seaborn==0.13.1
- tensorflow==2.15.0
- tqdm==4.66.1


## Utilisation

Pour utiliser ce projet, vous pouvez exécuter le fichier `run.py` avec les arguments suivants :

- `--train` : pour entraîner le modèle.
   - `--epoch` : nombre d'époques d'entraînement (par défaut: 5).
   - `--batch_size` : taille du batch d'entraînement (par défaut: 32).
   - `--weight_name` : nom du fichier de sauvegarde des poids du modèle (par défaut: 'model').
   - `--learning_rate` : taux d'apprentissage du modèle (par défaut: 0.01).
   - `--augmentation` : taux d'augmentation de la donnée (par défaut: 0.0).

- `--eval` : pour évaluer le modèle.
   - `--model_path` : chemin vers le fichier de poids du modèle (par défaut: 'weights/model.tf').
   - `--metrics` : liste des métriques à calculer (par défaut: ['accuracy', 'confusion_matrix', 'classification_report']).
   - `--no_save_cm` : désactiver la sauvegarde de la matrice de confusion (par défaut: activée).
   - `--no_save_txt` : désactiver la sauvegarde du rapport de classification (par défaut: activée).

- `--detect` : pour détecter une image.
   - `--image_path` : chemin vers l'image à détecter (obligatoire).
   - `--model_path` : chemin vers le fichier de poids du modèle (par défaut: 'weights/model.tf').

Si vous spécifiez plusieurs arguments parmi 'train', 'eval' et 'detect', une erreur sera levée.

Si vous ne spécifiez aucun arguments vosu pourrez acceder a l'interface pour effectuer une inference sur l'iamge et avec le poid de votre choix.

Exemples d'utilisation :
```bash

python run.py 
```
```bash

python run.py --eval --model_path weights/model_890.tf
```

### Entrainement

1. Clonez ce dépôt sur votre machine locale.
2. Téléchargez le(s) dataset(s) de votre choix.
3. Si les données sont déjà séparé ou que vous souahitez vous même le faire, extrayez les fichiers du dataset dans le répertoire `data` du projet. et séparer les fichiers de la facon suivante :
   ```
   > data
   >> train
   >>> no-cat
   >>> cat
   >> test
   >>> no-cat
   >>> cat
   ```
4. Sinon séparez les siplement en dossiers "cat" et "no-cat" dans un repertoir de votre choix, tel que "data_desoranizeds" par exemple.

```bash
python run.py --train --epoch 10 --batch_size 64 --weight_name my_personal_weight --learning_rate 0.0001 --augmentation 0.2
```


## Information sur le modèle

[Diaporama de présentation](https://www.canva.com/design/DAF5bDzfc-8/TaWygvwAZFYkLeaKwR2bTw/edit?utm_content=DAF5bDzfc-8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

### Architechure du modèle

1. **Couche d'Entrée (Flatten) :**
   - **Fonction :** Cette couche est responsable de la transformation des images d'entrée en un format adapté pour le réseau de neurones.
   - **Description :** Chaque image d'entrée, de taille 64x64 pixels avec 3 canaux de couleur (Rouge, Vert, Bleu), est aplatie dans un vecteur unidimensionnel. Cela signifie que chaque pixel de l'image est traité comme une caractéristique distincte.

2. **Couche Cachée (Dense, Activation 'relu') :**
   - **Fonction :** Cette couche est responsable de l'apprentissage des caractéristiques importantes des données d'entrée.
   - **Description :** Les 64x64x3 caractéristiques (uniques pour chaque pixel et canal de couleur) sont connectées à 128 neurones. Chaque connexion est associée à un poids qui sera ajusté pendant l'entraînement. L'activation 'relu' signifie que seules les valeurs positives sont transmises au neurone suivant, introduisant ainsi une non-linéarité dans le modèle.

3. **Couche de Sortie (Dense, Activation 'softmax') :**
   - **Fonction :** Cette couche est responsable de la génération des prédictions du modèle.
   - **Description :** Les 128 valeurs issues de la couche cachée sont connectées à 2 neurones de sortie, représentant les classes "No-Chat" et "Chat". L'activation 'softmax' normalise ces valeurs en probabilités, indiquant la probabilité que l'image appartienne à chaque classe. La classe avec la probabilité la plus élevée est alors considérée comme la prédiction finale.

4. **Fonction de Perte (Categorical Crossentropy) :**
   - **Fonction :** Mesure la différence entre les prédictions du modèle et les étiquettes réelles.
   - **Description :** L'objectif est de minimiser cette fonction pendant l'entraînement, afin que les prédictions du modèle se rapprochent le plus possible des vraies étiquettes.

5. **Optimiseur (Stochastic Gradient Descent, SGD) :**
   - **Fonction :** Optimise les poids du modèle pour minimiser la fonction de perte.
   - **Description :** L'optimiseur ajuste itérativement les poids du modèle pour réduire l'erreur entre les prédictions et les vraies étiquettes.

En résumé, ce modèle prend des images de 64x64 pixels en entrée, apprend des caractéristiques importantes dans une couche cachée, et génère des prédictions pour deux classes (No-Chat et Chat) à l'aide d'une couche de sortie. L'entraînement vise à ajuster les poids pour que les prédictions soient aussi précises que possible.

