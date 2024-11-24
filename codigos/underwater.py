import bibliotecaPDI as bP
import numpy as np

def main_menu():
    while True:
        print("\n--- Menú de Procesamiento de Imagen ---")
        print("1. Cargar imagen")
        print("2. Underwater")
        print("3. Rotar imagen")
        print("4. Desenfoque gaussiano")
        print("5. Cambio de perspectiva")
        print("6. Mostrar imagen")
        print("0. Salir")
        choice = input("Elige una opción: ")

        if choice == '1':
            image = bP.load_image()
            if image is not None:
                print("Imagen cargada correctamente.")
            else:
                print("-"*32)
                print("No se pudo cargar la imagen.")
                print("-"*32)
        
        elif choice == '2' and 'image' in locals():
            transformed_image = bP.underwater_effect(image)
            if(bP.are_images_equal(transformed_image, image)):
                print("Son la misma imagen")
            bP.display_color_results(image, transformed_image, "Underwater")

        elif choice == '3' and 'image' in locals():
            rotate = int(input('Ingresa un angulo (en grados) para rotar la imagen: '))
            transformed_image = bP.rotate_image(image, rotate)
            if(bP.are_images_equal(image, transformed_image)):
                print("Son la misma imagen")
            bP.display_color_results2(image, transformed_image, "Rotado a "+str(rotate)+" grados")

        elif choice == '4' and 'image' in locals():
            kernel_size = int(input('Ingresa un numero para la matriz, solo numeros impares ej. 3 para 3x3: '))
            if(kernel_size%2 == 0):
                print("El numero debe ser impar")
                break
            transformed_image = bP.apply_gaussian_blur(image, (kernel_size,kernel_size))
            if(bP.are_images_equal(image,transformed_image)):
                print("Son iguales")
            bP.display_color_results2(image, transformed_image, "Desenfoque con matriz "+str(kernel_size)+"x"+str(kernel_size))
        
        elif choice == '5' and 'image' in locals():
            # Puntos de origen (cuatro esquinas de un área de interés)
            src_points = np.float32([[50, 50], [400, 50], [50, 400], [400, 400]])
            # Puntos de destino (transformación deseada)
            dst_points = np.float32([[10, 100], [500, 50], [100, 500], [400, 400]])
            transformed_image = bP.apply_perspective_transform(image,src_points,dst_points)
            if(bP.are_images_equal(image,transformed_image)):
                print("Son iguales")
            bP.display_color_results2(image,transformed_image,"Cambio de perspectiva")

        elif choice == '6' and 'image' in locals():
            bP.display_image(image)
        
        elif choice == '0':
            print("Saliendo del programa.")
            break
        
        else:
            if 'image' not in locals():
                print("Primero debes cargar una imagen.")
            else:
                print("Opción no válida. Inténtalo de nuevo.")

if __name__ == "__main__":
    main_menu()
