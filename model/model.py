import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import os
from dataset import *

# Define tu modelo
class MalariaModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super(MalariaModel, self).__init__()
        # Define la arquitectura de tu red neuronal aquí
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Número de clases

    def forward(self, x):
        # Propagación hacia adelante
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # Lógica para un paso de entrenamiento
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        # Configura el optimizador
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Define tu conjunto de datos personalizado
class MalariaDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_dict, img_folder="C:/Malaria/Dataset/train/cells_train_processed"):
        self.datasets_dict = datasets_dict
        self.img_folder = img_folder

    def __len__(self):
        return len(self.datasets_dict)

    def __getitem__(self, idx):
        dataset = list(self.datasets_dict.values())[idx]
        image_path = os.path.join(self.img_folder, f"{dataset.id}.jpg")
        image_tensor = self.image_to_tensor(image_path)

        # Asigna etiquetas según las clases especificadas
        label_dict = {"Uninfected": 0, "Falciparum": 1, "Vivax": 2}
        label = label_dict.get(dataset.species, 0)

        return image_tensor, label

    def image_to_tensor(self, image_path):
        with Image.open(image_path) as img:
            tensor = torch.Tensor(list(img.getdata())).view(*img.size, -1)
            return tensor

# Configura tus DataLoader
def setup_data_loaders(datasets_dict, img_folder="C:/Malaria/Dataset/train/cells_train_processed", batch_size=32):
    # Divide tu conjunto de datos en entrenamiento y validación
    train_size = int(0.8 * len(datasets_dict))
    val_size = len(datasets_dict) - train_size
    train_dataset, val_dataset = random_split(list(datasets_dict.values()), [train_size, val_size])

    # Crea instancias de tu conjunto de datos personalizado
    train_dataset = MalariaDataset(train_dataset, img_folder=img_folder)
    val_dataset = MalariaDataset(val_dataset, img_folder=img_folder)

    # Configura los DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

# Configura y entrena tu modelo
def train_model(model, train_loader, val_loader, max_epochs=10):
    trainer = pl.Trainer(max_epochs=max_epochs)

    # Entrenamiento
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    # Carga tu diccionario de conjuntos de datos
    datasets_dict = load_datasets_into_dict("C:/Malaria/Dataset/train/cells_train_processed/info_train.txt")

    # Configura tu modelo
    model = MalariaModel(num_classes=3)  # Ajusta el número de clases

    # Configura tus DataLoader
    train_loader, val_loader = setup_data_loaders(datasets_dict)

    # Entrena tu modelo
    train_model(model, train_loader, val_loader)
