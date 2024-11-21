import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def global_histogram_equalization(image):
    """
    Aplica la ecualización de histograma global a una imagen en escala de grises.
    :param image: Imagen en escala de grises.
    :return: Imagen con histograma ecualizado globalmente.
    """
    return cv2.equalizeHist(image)
1
def local_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica la ecualización de histograma local (CLAHE) a una imagen en escala de grises.
    :param image: Imagen en escala de grises.
    :param clip_limit: Limite de corte para la ecualización local. Valores más altos aumentan el contraste.
    :param tile_grid_size: Tamaño de la cuadrícula de tiles para el ecualizador.
    :return: Imagen con ecualización de histograma local aplicada.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def global_mean_and_variance(image):
    """
    Calcula la media y varianza global de una imagen en escala de grises.
    :param image: Imagen en escala de grises.
    :return: Tupla (media, varianza).
    """
    mean = np.mean(image)
    variance = np.var(image)
    return mean, variance

def local_mean_and_variance(image, kernel_size=(3, 3)):
    """
    Calcula la media y varianza local de una imagen en escala de grises utilizando un kernel.
    :param image: Imagen en escala de grises.
    :param kernel_size: Tamaño del kernel para calcular las estadísticas locales.
    :return: Tupla (media_local, varianza_local).
    """
    local_mean = cv2.blur(image, kernel_size) 
    # Calcula la varianza local usando la fórmula: varianza = E[X^2] - (E[X])^2
    local_mean_square = cv2.blur(image**2, kernel_size)
    local_variance = local_mean_square - local_mean**2
    
    return local_mean, local_variance

def enhanced_contrast_filter(image, E, k0, k1, k2, mask_size=(3, 3)):
    """
    Aplica una transformación de contraste a la imagen basada en la media y varianza globales y locales.
    g(x, y) = E * f(x, y) si m_sxy <= k0 * m_g y k1 * v_g <= v_sxy <= k2 * v_g, 
              f(x, y) en caso contrario.

    :param image: Imagen en escala de grises.
    :param E: Constante de realce.
    :param k0: Umbral para comparar con la media local.
    :param k1: Umbral inferior para comparar con la varianza local.
    :param k2: Umbral superior para comparar con la varianza local.
    :param mask_size: Tamaño de la máscara (por defecto 3x3).
    :return: Imagen transformada.
    """
    m_g, v_g = global_mean_and_variance(image)
    pad_y, pad_x = mask_size[0] // 2, mask_size[1] // 2
    padded_image = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
    
    result = np.copy(image)

    for y in range(pad_y, padded_image.shape[0] - pad_y):
        for x in range(pad_x, padded_image.shape[1] - pad_x):
            neighborhood = padded_image[y - pad_y : y + pad_y + 1, x - pad_x : x + pad_x + 1]
        
            m_sxy = np.mean(neighborhood)
            v_sxy = np.var(neighborhood)
            
            if m_sxy <= k0 * m_g and k1 * v_g <= v_sxy <= k2 * v_g:
                result[y - pad_y, x - pad_x] = np.clip(E * image[y - pad_y, x - pad_x], 0, 255)
            else:
                result[y - pad_y, x - pad_x] = image[y - pad_y, x - pad_x]

    return result

def load_image():
    print("\n--- Menú de Imagenes ---")
    print("1. Imagen de alta iluminacion")
    print("2. Imagen de baja iluminacion")
    print("3. Imagen de alto contraste")
    print("4. Imagen de bajo contraste")
    option = input("Selecciona tu imagen: ")
    if option == '1':
        image = cv2.imread("../imagenesPractica4/altaIluminacion.jpg", cv2.IMREAD_GRAYSCALE)
    elif option == '2':
        image = cv2.imread("../imagenesPractica4/bajaIluminacion.png", cv2.IMREAD_GRAYSCALE)
    elif option == '3':
        image = cv2.imread("../imagenesPractica4/altoContraste.png", cv2.IMREAD_GRAYSCALE)
    elif option == '4':
        image = cv2.imread("../imagenesPractica4/bajoContraste.png", cv2.IMREAD_GRAYSCALE)
    else:
        print("Seleccion invalida")
        return None
    return image

def display_results(original, transformed, title):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Imagen Original")
    axs[0, 0].axis('off')
    
    axs[1, 0].hist(original.ravel(), bins=256, range=(0, 256), color='black')
    axs[1, 0].set_title("Histograma Original")

    axs[0, 1].imshow(transformed, cmap='gray')
    axs[0, 1].set_title("Imagen Transformada")
    axs[0, 1].axis('off')
    
    axs[1, 1].hist(transformed.ravel(), bins=256, range=(0, 256), color='black')
    axs[1, 1].set_title("Histograma Transformado")

    plt.tight_layout()
    plt.show()

def display_image(image, title="Imagen"):
    """
    Muestra una única imagen en escala de grises.
    :param image: Imagen en formato numpy array (en escala de grises).
    :param title: Título de la imagen a mostrar.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def are_images_equal(img1, img2):
    """
    Compara si dos imágenes son exactamente iguales.
    :param img1: Primera imagen (numpy array).
    :param img2: Segunda imagen (numpy array).
    :return: True si son iguales, False en caso contrario.
    """
    return np.array_equal(img1, img2)

def filtro_promedio(image, kernel_size = 3):
    """
    Aplica un filtro promedio a la imagen.

    Calcula el promedio de los píxeles dentro de una vecindad definida por el tamaño de la máscara
    y reemplaza cada píxel por dicho promedio, suavizando la imagen.

    :param image: Imagen en escala de grises (2D array).
    :param kernel_size: Tamaño de la máscara (entero positivo).
    :return: Imagen filtrada (2D array).
    """
    return cv2.blur(image, (kernel_size, kernel_size))

def filtro_mediano(image, kernel_size = 3):
    """
    Aplica un filtro mediano a la imagen.

    Reemplaza cada píxel por el valor mediano de los píxeles en una vecindad definida por el tamaño
    de la máscara, eliminando ruido sal y pimienta.

    :param image: Imagen en escala de grises (2D array).
    :param kernel_size: Tamaño de la máscara (entero positivo, debe ser impar).
    :return: Imagen filtrada (2D array).
    """
    return cv2.medianBlur(image, kernel_size)

def filtro_maximo(image, kernel_size = 3):
    """
    Aplica un filtro máximo a la imagen.

    Reemplaza cada píxel por el valor máximo dentro de una vecindad definida por el tamaño de la máscara,
    destacando características brillantes.

    :param image: Imagen en escala de grises (2D array).
    :param kernel_size: Tamaño de la máscara (entero positivo).
    :return: Imagen filtrada (2D array).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel)

def filtro_minimo(image, kernel_size = 3):
    """
    Aplica un filtro mínimo a la imagen.

    Reemplaza cada píxel por el valor mínimo dentro de una vecindad definida por el tamaño de la máscara,
    destacando características oscuras.

    :param image: Imagen en escala de grises (2D array).
    :param kernel_size: Tamaño de la máscara (entero positivo).
    :return: Imagen filtrada (2D array).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel)

def filtro_laplaciano(image):
    """
    Aplica un filtro laplaciano a la imagen.

    Calcula la segunda derivada de la intensidad en la imagen, resaltando bordes y áreas de cambio
    rápido en los niveles de intensidad.

    :param image: Imagen en escala de grises (2D array).
    :return: Imagen filtrada (2D array).
    """
    return cv2.Laplacian(image, cv2.CV_64F)

def filtro_gradiente(image):
    """
    Aplica un filtro de gradiente a la imagen.

    Calcula la magnitud del gradiente utilizando los operadores Sobel en las direcciones x e y,
    resaltando bordes y cambios de intensidad.

    :param image: Imagen en escala de grises (2D array).
    :return: Imagen filtrada con la magnitud del gradiente (2D array).
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)

def sumar_imagenes(image1, image2, alpha=0.5, beta=0.5, gamma=0):
    """
    Suma dos imágenes con pesos específicos utilizando OpenCV.

    :param image1: Primera imagen (2D array o 3D array para imágenes RGB).
    :param image2: Segunda imagen (2D array o 3D array para imágenes RGB).
    :param alpha: Peso para la primera imagen (por defecto 0.5).
    :param beta: Peso para la segunda imagen (por defecto 0.5).
    :param gamma: Valor escalar adicional para ajustar la suma (por defecto 0).
    :return: Imagen resultante de la suma ponderada.
    """
    if image1.shape != image2.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño y número de canales.")

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    resultado = cv2.addWeighted(image1, alpha, image2, beta, gamma)
    
    return resultado.astype(np.uint8)

def sumar_imagenes2(img1, img2=None, valor=0):
    if img2 is not None:
        # Suma de dos imágenes
        return cv2.add(img1, img2)
    else:
        # Suma de una imagen con un valor constante
        return cv2.add(img1, np.full(img1.shape, valor, dtype=img1.dtype))

def escalamiento_abs(image):
    return cv2.convertScaleAbs(image)