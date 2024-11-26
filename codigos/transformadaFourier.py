import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import bibliotecaPDI as bp
 
img = cv.imread('../imagenesPractica4/cuadrado.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "No se pudo cargar la imagen"
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
 
#bp.display_results(img, f, "Magnitud del espectro")
bp.display_image(fshift, "Mangitud del espectro")