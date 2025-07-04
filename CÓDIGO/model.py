import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Bloque convolucional para la imagen
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  
        )

        # Subred densa para la imagen 
        self.img_dense = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Capa final combinada con el tipo de celda
        self.final = nn.Sequential(
            nn.Linear(256 + 1, 128),       # +1 por el tipo de celda
            nn.ReLU(),
            nn.Linear(128, 1)             
        )

    def forward(self, imagen, tipo_celda):
        x = self.features(imagen)        
        x = self.img_dense(x)             

        tipo_celda = tipo_celda.view(-1, 1)  
        x = torch.cat((x, tipo_celda), dim=1)  

        out = self.final(x)              
        return out
