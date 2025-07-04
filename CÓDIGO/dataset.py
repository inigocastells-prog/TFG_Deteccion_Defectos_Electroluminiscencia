import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class CustomImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform_normal=None, transform_defective=None):
        self.data = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2])
        self.root_dir = root_dir
        self.transform_normal = transform_normal
        self.transform_defective = transform_defective

        # cambiar 'mono' a 0 y 'poly' a 1
        self.data[2] = self.data[2].map({'mono': 0, 'poly': 1})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel_path = self.data.iloc[idx, 0]
        full_path = os.path.join(self.root_dir, rel_path)
        label = float(self.data.iloc[idx, 1])
        cell_type = float(self.data.iloc[idx, 2])  # tipo de celda como float (0 o 1)

        image = Image.open(full_path).convert("L")

        if label == 1 and self.transform_defective:
            image = self.transform_defective(image)
        elif self.transform_normal:
            image = self.transform_normal(image)

        return image, torch.tensor(label), torch.tensor(cell_type)
