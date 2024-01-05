import torch
from PIL import Image
import os
import pytorch_lightning as pl

class Dataset:
    def __init__(self, id, species, type, comment, tensor):
        self.id = id
        self.species = species
        self.type = type
        self.comment = comment
        self.tensor = tensor

    @classmethod
    def from_txt_and_image(cls, line, img_folder="C:/Malaria/Dataset/train/cells_train_processed"):
        # Asegúrate de que hay al menos 4 elementos en la línea
        elements = line.strip().split(',')
        if len(elements) >= 4:
            id, species, type, comment = elements
            image_path = os.path.join(img_folder, f"{id}.jpg")
            tensor = cls.image_to_tensor(image_path)
            return cls(id, species, type, comment, tensor)
        else:
            # Manejar el caso donde no hay suficientes elementos en la línea
            print(f"Error: No hay suficientes elementos en la línea: {line}")
            return None

    @classmethod
    def image_to_tensor(cls, image_path):
        with Image.open(image_path) as img:
            # Convertir a escala de grises y redimensionar si es necesario
            tensor = torch.Tensor(list(img.getdata())).view(*img.size, -1)
            return tensor
    
    @classmethod
    def get_item(cls, id):
        print(datasets_dict[id].id, datasets_dict[id].species)
        return datasets_dict[id]

def load_datasets_into_dict(txt_path, img_folder="C:/Malaria/Dataset/train/cells_train_processed"):
    datasets_dict = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset = Dataset.from_txt_and_image(line, img_folder)
            if dataset:
                datasets_dict[dataset.id] = dataset
    return datasets_dict

# Ejemplo de uso:
datasets_dict = load_datasets_into_dict("C:/Malaria/Dataset/train/cells_train_processed/info_train.txt")

# Imprime la información para verificar:
for id, dataset in datasets_dict.items():
    print(f"ID: {id}")
    print(f"Species: {dataset.species}")
    print(f"Type: {dataset.type}")
    print(f"Comment: {dataset.comment}")
    print(f"Tensor Size: {dataset.tensor.size()}\n")
