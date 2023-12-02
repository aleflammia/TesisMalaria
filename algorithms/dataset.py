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
    def from_txt_and_image(cls, line, img_folder="C:/Pruebax/cells/"):
        id, species, type, comment = line.strip().split(',')
        image_path = os.path.join(img_folder, f"{id}.jpg")
        tensor = cls.image_to_tensor(image_path)
        return cls(id, species, type, comment, tensor)

    @staticmethod
    def image_to_tensor(image_path):
        # Abre la imagen y conviértela a tensor
        with Image.open(image_path) as img:
            # Convertir a escala de grises y redimensionar si es necesario
            tensor = torch.Tensor(list(img.getdata())).view(*img.size, -1)
            return tensor
        
    

    def get_item(id):

        print(datasets_dict[id].id, datasets_dict[id].species)
        return datasets_dict[id]

        
def load_datasets_into_dict(txt_path, img_folder="C:/Pruebax/cells/"):
    datasets_dict = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset = Dataset.from_txt_and_image(line, img_folder)
            datasets_dict[dataset.id] = dataset
    return datasets_dict

# Ejemplo de uso:
datasets_dict = load_datasets_into_dict("C:/Pruebax/cells/prueba.txt")

# Imprime la información para verificar:
""" for id, dataset in datasets_dict.items():
    print(f"ID: {id}")
    print(f"Species: {dataset.species}")
    print(f"Type: {dataset.type}")
    print(f"Comment: {dataset.comment}")
    print(f"Tensor Size: {dataset.tensor.size()}\n") """

id = input("Ingrese el beta")
print(get_item(id))