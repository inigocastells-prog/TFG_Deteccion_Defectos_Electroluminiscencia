import pco
import os
import csv
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage

from model import CNN
from prueba_PREPROCESADO import procesar_imagen  

def escalar_grises(image, modo="auto", bits=12, aplicar_clahe=True, gamma=1.5):
    if modo == "fijo":
        escala_max = float(2**bits - 1)
        norm_img = (image / escala_max * 255).clip(0, 255).astype(np.uint8)
    else:
        norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if aplicar_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        norm_img = clahe.apply(norm_img)

    if gamma != 1.0:
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
        norm_img = cv2.LUT(norm_img, lut)

    return norm_img

def cargar_configuracion_csv(ruta_csv):
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    with open(ruta_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            return row

def main():
    config = cargar_configuracion_csv("configuracion.csv")
    nombre_base = os.path.splitext(config["nombre_imagen"])[0]
    tiempo_exposicion_ms = float(config["tiempo_exposicion"])
    exposure_time = tiempo_exposicion_ms / 1000.0
    tipo_panel = config["tipo"].strip().upper()
    tipo_valor = 0.0 if tipo_panel == "MONO" else 1.0

    filas = int(config["filas"])
    columnas = int(config["columnas"])

    # Rutas de guardado
    preview_path = os.path.abspath(f"{nombre_base}_preview.png")
    salida_celdas_path = os.path.abspath(f"{nombre_base}_solo_celdas.png")

    with pco.Camera() as cam:
        print("Cámara conectada")
        cam.exposure_time = exposure_time
        print(f"Exposición configurada: {cam.exposure_time:.3f} s")

        try:
            # Capturar imagen
            cam.record(number_of_images=1, mode="sequence")
            image = cam.image()[0]
            print("Imagen capturada")
            print(f"RAW: dtype={image.dtype}, min={image.min()}, max={image.max()}")

            # Escalar para visualización
            norm_img = escalar_grises(image, modo="auto", bits=12, aplicar_clahe=True, gamma=1.5)

            # Guardar vista previa
            Image.fromarray(norm_img).save(preview_path)
            print(f"Vista previa guardada en: {preview_path}")

            # Procesamiento
            imagen_cv = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
            solo_celdas = procesar_imagen(imagen_cv, nombre_base_salida=nombre_base)

            # Guardar imagen final corregida
            cv2.imwrite(salida_celdas_path, solo_celdas)
            print(f"Imagen final (solo celdas) guardada en: {salida_celdas_path}")

            # Crear carpeta para celdas individuales
            carpeta_celdas = os.path.abspath(nombre_base)
            os.makedirs(carpeta_celdas, exist_ok=True)

            alto, ancho = solo_celdas.shape[:2]
            alto_celda = alto // filas
            ancho_celda = ancho // columnas

            print("🔄 Dividiendo imagen en celdas...")
            for i in range(filas):
                for j in range(columnas):
                    y1 = i * alto_celda
                    y2 = (i + 1) * alto_celda
                    x1 = j * ancho_celda
                    x2 = (j + 1) * ancho_celda
                    celda = solo_celdas[y1:y2, x1:x2]

                    nombre_celda = f"{nombre_base}_f{i+1}c{j+1}.png"
                    ruta_celda = os.path.join(carpeta_celdas, nombre_celda)
                    cv2.imwrite(ruta_celda, celda)

            print("Celdas individuales guardadas correctamente.")

        except Exception as e:
            print(f"Error durante la ejecución: {e}")

    print("🔌 Cámara cerrada")

    # Predicción
    print("\nCargando modelo y haciendo predicciones...\n")

    modelo = CNN()
    modelo.load_state_dict(torch.load("mejor_modelo.pth", map_location=torch.device('cpu')))
    modelo.eval()

    # Transformación igual a la del entrenamiento
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    tipo_tensor = torch.tensor([[tipo_valor]])

    # Crear archivo para guardar resultados
    ruta_resultado_txt = os.path.abspath(f"{nombre_base}_resultados.txt")
    archivo_resultado = open(ruta_resultado_txt, "w", encoding="utf-8")

    # Crear carpeta para imágenes marcadas
    carpeta_marcadas = os.path.abspath(f"{nombre_base}_marcadas")
    os.makedirs(carpeta_marcadas, exist_ok=True)

    for i in range(filas):
        for j in range(columnas):
            nombre_celda = f"{nombre_base}_f{i+1}c{j+1}.png"
            ruta_celda = os.path.join(carpeta_celdas, nombre_celda)

            img = cv2.imread(ruta_celda, cv2.IMREAD_COLOR)
            if img is None:
                print(f"No se pudo cargar {nombre_celda}")
                continue

            img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            entrada = transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                salida = modelo(entrada, tipo_tensor)
                prob = torch.sigmoid(salida).item()
                pred = 1 if prob > 0.5 else 0

            linea = f"{nombre_celda} → Predicción: {pred} (prob: {prob:.2f})"
            print(f" {linea}")
            archivo_resultado.write(linea + "\n")

            # Crear imagen marcada
            imagen_marcada = img.copy()
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)  # verde o rojo
            cv2.circle(imagen_marcada, (10, 10), 6, color, -1)  # punto en (10,10)
            ruta_marcada = os.path.join(carpeta_marcadas, nombre_celda)
            cv2.imwrite(ruta_marcada, imagen_marcada)

    archivo_resultado.close()
    print(f"\n Resultados guardados en: {ruta_resultado_txt}")
    print(f" Imágenes marcadas guardadas en: {carpeta_marcadas}")

if __name__ == "__main__":
    main()
