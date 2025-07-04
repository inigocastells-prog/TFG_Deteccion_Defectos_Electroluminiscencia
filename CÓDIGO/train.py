import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from dataset import CustomImageDataset
from model import CNN  

# Rutas
root_dir = "C:/Users/caste/OneDrive/Escritorio/TFG/ETAPA3-REDNEURONAL/REDNEURONAL/IMAGENES/DATASET_DESBALANCEADA"
csv_path = os.path.join(root_dir, "new_labels.csv")
modelo_guardado = "modelo_entrenado.pth"

# Parámetros
batch_size = 16
num_epochs = 30
learning_rate = 0.0005

# Transformaciones
transform_normal = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

transform_defective = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=1.0),  
    transforms.ToTensor()
])

# Dataset
dataset = CustomImageDataset(csv_path, root_dir, transform_normal=transform_normal, transform_defective=transform_defective)


# División entrenamiento/test
indices = list(range(len(dataset))) # Crea una lista de índices con todas las imágenes que hay en el dataset
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42) # Divide esa lista en subconjuntos, uno del 30% para la prueba y el de 70% para el entrenamiento
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelo
modelo = CNN()  
modelo.train()

# Cálculo del pos_weight para desbalance
num_pos = sum([1 for i in train_idx if dataset.data.iloc[i, 1] == 1])
num_neg = sum([1 for i in train_idx if dataset.data.iloc[i, 1] == 0])
pos_weight = torch.tensor([num_neg / num_pos])

# Función de pérdida y optimizador
criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizador = optim.Adam(modelo.parameters(), lr=learning_rate)

# Entrenamiento
best_loss = float("inf")
patience = 7  # Número de épocas sin mejorar antes de parar
counter = 0

for epoch in range(num_epochs):
    modelo.train()
    perdida_total = 0.0

    for imagenes, etiquetas, tipos in train_loader:
        etiquetas = etiquetas.unsqueeze(1).float()
        tipos = tipos.float()

        salida = modelo(imagenes, tipos)
        perdida = criterio(salida, etiquetas)

        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item()

    epoch_loss = perdida_total / len(train_loader)

    # Cálculo del validation loss
    modelo.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imagenes, etiquetas, tipos in test_loader:
            etiquetas = etiquetas.unsqueeze(1).float()
            tipos = tipos.float()

            salida = modelo(imagenes, tipos)
            perdida = criterio(salida, etiquetas)
            val_loss += perdida.item()
    val_loss = val_loss / len(test_loader)

    print(f"Época {epoch+1}/{num_epochs} - Pérdida media (train): {epoch_loss:.4f} - Pérdida validación: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(modelo.state_dict(), "mejor_modelo.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Parando entrenamiento por early stopping (sin mejora en val_loss).")
            break
        
# Guardar modelo
torch.save(modelo.state_dict(), modelo_guardado)
print(f"\n Modelo guardado como '{modelo_guardado}'")

# Evaluación
modelo.load_state_dict(torch.load("mejor_modelo.pth"))
modelo.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for imagenes, etiquetas, tipos in test_loader:
        etiquetas = etiquetas.unsqueeze(1).float()
        tipos = tipos.float()

        salida = modelo(imagenes, tipos)
        probabilidades = torch.sigmoid(salida)
        predicciones = (probabilidades > 0.5).float()

        all_labels.extend(etiquetas.cpu().numpy())
        all_preds.extend(predicciones.cpu().numpy())

all_labels = np.array(all_labels).astype(int)
all_preds = np.array(all_preds).astype(int)


# Conteo de clases
print("\n Conteo de clases en el conjunto de prueba:")
print("Etiquetas reales:", Counter(all_labels.flatten().tolist()))
print("Predicciones del modelo:", Counter(all_preds.flatten().tolist()))

# Precisión total
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n Precisión total (Test Accuracy): {accuracy:.2%}")

# Reporte
print("\n📋 Reporte de Clasificación:")
print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1],
    target_names=["No defectuosa", "Defectuosa"]
))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No defectuosa", "Defectuosa"])

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title(" Matriz de Confusión")
plt.show()


# Mostrar imágenes mal clasificadas: defectuosas predichas como no defectuosas
from torchvision.transforms.functional import to_pil_image

mal_clasificadas = []

# Volvemos a recorrer el test_loader para guardar imágenes mal clasificadas
with torch.no_grad():
    for imagenes, etiquetas, tipos in test_loader:
        etiquetas = etiquetas.unsqueeze(1).float()
        tipos = tipos.float()

        salida = modelo(imagenes, tipos)
        probabilidades = torch.sigmoid(salida)
        predicciones = (probabilidades > 0.5).float()

        for i in range(len(imagenes)):
            real = int(etiquetas[i].item())
            pred = int(predicciones[i].item())

            if real == 1 and pred == 0:  #  es defectuosa pero no la detectó
                mal_clasificadas.append(to_pil_image(imagenes[i]))

# Mostrar todas las imágenes mal clasificadas en un solo gráfico
print(f"\n Mostrando {len(mal_clasificadas)} imágenes mal clasificadas (NO defectuosas pero sí lo eran):")

if len(mal_clasificadas) > 0:
    cols = 5  # Número de columnas en la cuadrícula
    rows = (len(mal_clasificadas) + cols - 1) // cols  # Cálculo de filas necesarias

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, img in enumerate(mal_clasificadas):
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(" Incorrecta")

    # Ocultar ejes vacíos
    for i in range(len(mal_clasificadas), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle("Clasificadas como NO defectuosas (pero sí lo eran)", fontsize=14, y=1.02)
    plt.show()
else:
    print(" No hay errores de ese tipo en esta ejecución.")
