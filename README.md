# Classification des empreintes d'animaux

Ce projet vise à réaliser une classification d'images d'empreintes d'animaux parmi 12 classes différentes. Les empreintes d'animaux sont réparties dans les catégories suivantes : ['background', 'beaver', 'cat', 'dog', 'coyote', 'squirrel', 'rabbit', 'wolf', 'lynx', 'bear', 'puma', 'rat', 'raccoon', 'fox']. Le modèle utilisé pour cette classification est une adaptation du modèle VGG16.

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

Si vous ne spécifiez aucun argument, vous pourrez accéder à l'interface pour effectuer une inférence sur l'image avec le poids de votre choix.

Exemples d'utilisation :
```bash

python run.py 
```
```bash

python run.py --eval --model_path weights/model_890.tf
```

### Entraînement

1. Clonez ce dépôt sur votre machine locale.
2. Téléchargez le(s) dataset(s) de votre choix.
3. Si les données sont déjà séparées ou que vous souhaitez le faire vous-même, extrayez les fichiers du dataset dans le répertoire `data` du projet. Séparez ensuite les fichiers en dossiers "cat" et "no-cat" comme suit :
   ```
   > data
   >> train
   >>> no-cat
   >>> cat
   >> test
   >>> no-cat
   >>> cat
   ```
   Sinon, séparez-les simplement en dossiers "cat" et "no-cat" dans un répertoire de votre choix, tel que "data_desorganized" par exemple.

```bash
python run.py --train --epoch 10 --batch_size 64 --weight_name my_personal_weight --learning_rate 0.0001 --augmentation 0.2
```

### Information sur le modèle

Le modèle utilisé est une adaptation de l'architecture VGG16, pré-entraînée sur ImageNet. Certaines couches de ce modèle ont été gardées tandis que d'autres ont été modifiées ou ajoutées pour répondre aux besoins spécifiques de la tâche de classification des empreintes d'animaux.
![Schéma du modèle VGG16](https://miro.medium.com/v2/resize:fit:1400/1*NNifzsJ7tD2kAfBXt3AzEg.png)

#### Couches conservées de VGG16 :

- **Couches convolutives :** Les couches convolutives de VGG16 ont été conservées car elles sont efficaces pour extraire des caractéristiques de bas niveau dans les images. Ces couches sont importantes pour la détection des motifs visuels dans les empreintes d'animaux.

#### Couches modifiées ou ajoutées :

- **Couche de pooling globale :** Une couche de pooling globale a été ajoutée après les couches convolutives pour réduire la dimensionnalité de la sortie des couches précédentes tout en préservant les informations importantes.

- **Couches fully connected personnalisées :** Deux couches fully connected ont été ajoutées après la couche de pooling globale pour permettre au modèle de combiner les caractéristiques extraites par les couches convolutives et de produire des prédictions pour les différentes classes d'empreintes d'animaux.

- **Couche de dropout :** Une couche de dropout a été ajoutée pour réduire le surajustement en désactivant aléatoirement un certain pourcentage des neurones lors de l'entraînement, ce qui permet d'améliorer la généralisation du modèle.

- **Couche de sortie :** Une couche de sortie avec une fonction d'activation softmax a été ajoutée pour produire des probabilités de classification pour chaque classe d'empreintes d'animaux.

#### Rôle des couches ajoutées :

- La couche de pooling globale réduit la dimensionnalité de la sortie des couches convolutives tout en préservant les informations importantes, ce qui permet d'économiser du temps de calcul et de réduire le risque de surajustement.

- Les couches fully connected personnalisées combinent les caractéristiques extraites par les couches convolutives et les transforment en vecteurs de caractéristiques, qui sont ensuite utilisés pour produire des prédictions pour les différentes classes d'empreintes d'animaux.

- La couche de dropout aide à régulariser le modèle en réduisant le surajustement pendant l'entraînement en désactivant aléatoirement un certain pourcentage des neurones, ce qui permet d'améliorer la généralisation du modèle.

- La couche de sortie avec une fonction d'activation softmax produit des probabilités de classification pour chaque classe d'empreintes d'animaux, permettant ainsi de déterminer la classe la plus probable pour une empreinte donnée.
