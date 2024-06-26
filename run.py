import argparse

from utils.utils import clear_terminal

from train import train_model
from evaluate import evaluate_model
from detect import infer_image

import os

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--detect', action='store_true')

if parser.parse_known_args()[0].train:
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_name', type=str, default='model')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--augmentation', type=float, default=0.0)
    parser.add_argument('--auto_eval', action='store_false', default=True, help="Set to False to disable auto eval model")
elif parser.parse_known_args()[0].eval:
    parser.add_argument('--model_path', type=str, default='weights/model.tf')
    parser.add_argument('--metrics', type=list, default=['accuracy', 'confusion_matrix', 'classification_report'])
    parser.add_argument('--no_save_cm', action='store_false', default=True, help="Set to False to disable saving confusion matrix")
    parser.add_argument('--no_save_txt', action='store_false', default=True, help="Set to False to disable saving classification report")
elif parser.parse_known_args()[0].detect:
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='weights/model.tf')

args = parser.parse_args()

if sum([args.train, args.eval, args.detect]) > 1:
    raise ValueError("Only one argument among 'train', 'eval', and 'detect' can be specified.")

elif args.train:
    clear_terminal()
    print("Training the model: \n \n \n")
    _, _, model_path = train_model(args.epoch, args.batch_size, args.weight_name, args.learning_rate, args.augmentation)
    if args.augmentation != 0.0 :
        for dossier_racine, _, fichiers in os.walk('./data'):
            [os.remove(os.path.join(dossier_racine, fichier)) for fichier in fichiers if fichier.split(".")[0].endswith('_augmented')]
    if args.auto_eval == True:
        clear_terminal()
        print("Evaluating the model: \n \n \n")
        evaluate_model(model_path[0:-3], ['accuracy', 'confusion_matrix', 'classification_report'])

elif args.eval:
    clear_terminal()
    print("Evaluating the model: \n \n \n")
    evaluate_model(args.model_path, args.metrics, args.no_save_cm, args.no_save_txt)

elif args.detect:
    clear_terminal()
    print("Detecting the class: \n \n \n")
    infer_image(args.image_path, args.model_path)
    
else:
    clear_terminal()
    print("No argument specified. Please specify one argument among 'train', 'eval', and 'detect'.")
 
