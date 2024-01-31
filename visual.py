import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from detect import inferer_image

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Bienvenue dans votre application de classification d'images de chats et no_cats")

        self.welcome_label = tk.Label(self.master, text="Bienvenue dans votre application de classification d'images de chats et no_cats")
        self.welcome_label.pack(pady=20)
        self.continue_button = tk.Button(self.master, text="Continuer", command=self.show_main_interface)
        self.continue_button.pack()

    def show_main_interface(self):
        # Détruire les widgets de la page d'accueil
        self.welcome_label.destroy()
        self.continue_button.destroy()

        # Créer les widgets de l'interface principale
        self.image_label = tk.Label(self.master, text="Aucune image sélectionnée")
        self.image_label.pack()

        self.choose_button = tk.Button(self.master, text="Choisir une image", command=self.choose_image)
        self.choose_button.pack(side=tk.LEFT, padx=10)

        self.choose_weight_button = tk.Button(self.master, text="Choisir un dossier de poids", command=self.choose_weight)
        self.choose_weight_button.pack(side=tk.LEFT, padx=10)

        self.classify_button = tk.Button(self.master, text="Classer l'image", command=self.classify_image)
        self.classify_button.pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack(pady=20)

    def choose_image(self):
        file_path = filedialog.askopenfilename(title="Choisir une image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.show_image(file_path)

    def show_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.selected_image_path = file_path

    def choose_weight(self):
        weight_path = filedialog.askdirectory(title="Choisir un dossier de poids")
        if weight_path:
            self.selected_weight_path = weight_path

    def classify_image(self):
        if hasattr(self, 'selected_image_path') and hasattr(self, 'selected_weight_path'):
            detection = inferer_image(self.selected_image_path, self.selected_weight_path)
            result = "Image classée comme " + detection  
            self.result_label.config(text=result)
        else:
            messagebox.showinfo("Info", "Veuillez choisir une image et un dossier de poids avant de classer.")

def interface():
    root = tk.Tk()
    root.title("Image Classifier App")
    root.geometry("600x400")
    app = ImageClassifierApp(root)
    root.mainloop()