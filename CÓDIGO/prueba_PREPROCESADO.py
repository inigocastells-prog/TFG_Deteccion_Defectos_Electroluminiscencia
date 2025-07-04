import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_imagen(imagen, nombre_base_salida="salida"):
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicación de CLAHE para mejorar el contraste 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Umbral binario 
    _, umbral = cv2.threshold(gray_clahe, 120, 255, cv2.THRESH_BINARY)

    # Cierre morfológico para eliminar líneas internas
    kernel = np.ones((12, 12), np.uint8)
    filled_image = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar contornos
    contours, _ = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = imagen.copy()

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]

        # Aproximar a polígono
        epsilon = 0.04 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:
            print(" Se detectaron 4 esquinas.")
            approx_contour = approx.reshape(4, 2)
        else:
            print(f" Solo se detectaron {len(approx)} esquinas. Usando rectángulo mínimo.")
            rect = cv2.minAreaRect(largest_contour)
            approx_contour = cv2.boxPoints(rect).astype(np.int32)

        # Dibujar contorno
        cv2.drawContours(contoured_image, [approx_contour], -1, (0, 0, 255), 3)

        # Ordenar puntos
        rect = np.zeros((4, 2), dtype="float32")
        s = approx_contour.sum(axis=1)
        rect[0] = approx_contour[np.argmin(s)]
        rect[2] = approx_contour[np.argmax(s)]
        diff = np.diff(approx_contour, axis=1)
        rect[1] = approx_contour[np.argmin(diff)]
        rect[3] = approx_contour[np.argmax(diff)]

        # Dimensiones del nuevo plano
        width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
        height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Corregir perspectiva
        M = cv2.getPerspectiveTransform(rect, dst)
        corrected_image = cv2.warpPerspective(imagen, M, (int(width), int(height)))

        # Recorte manual 
        recorte_arriba = 50
        recorte_abajo = 30
        recorte_izquierda = 80
        recorte_derecha = 100

        alto, ancho = corrected_image.shape[:2]
        corrected_image = corrected_image[
            recorte_arriba : alto - recorte_abajo,
            recorte_izquierda : ancho - recorte_derecha
]

    else:
        print(" No se encontraron contornos.")
        corrected_image = imagen.copy()

    # Mostrar resultados
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ax = ax.flatten()

    ax[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    ax[1].imshow(gray_clahe, cmap="gray")
    ax[1].set_title("Imagen con contraste (CLAHE)")
    ax[1].axis("off")

    ax[2].imshow(umbral, cmap="gray")
    ax[2].set_title("Imagen umbralizada")
    ax[2].axis("off")

    ax[3].imshow(filled_image, cmap="gray")
    ax[3].set_title("Imagen rellena")
    ax[3].axis("off")

    ax[4].imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
    ax[4].set_title("Área detectada en imagen original")
    ax[4].axis("off")

    ax[5].imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    ax[5].set_title("Imagen corregida")
    ax[5].axis("off")

    plt.tight_layout()
    plt.show()

    return corrected_image
