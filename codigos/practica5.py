import bibliotecaPDI as bP

def main_menu():
    while True:
        print("\n--- Menú de Procesamiento de Imagen ---")
        print("1. Cargar imagen")
        print("2. Filtro promedio")
        print("3. Filtro de la mediana")
        print("4. Filtro del maximo")
        print("5. Filtro del minimo")
        print("6. Filtro Laplaciano")
        print("7. Filtro Gradiente")
        print("8. Mostrar imagen cargada")
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
            kernel_size = int(input("Ingresa el tamaño del filtro (ej. 3 para 3x3): "))
            transformed_image = bP.filtro_promedio(image=image, kernel_size=kernel_size)
            if(bP.are_images_equal(transformed_image, image)):
                print("Son la misma imagen")
            bP.display_results(image, transformed_image, "Filtro Promedio")
        
        elif choice == '3' and 'image' in locals():
            kernel_size = int(input("Ingresa el tamaño del filtro (ej. 3 para 3x3): "))
            transformed_image = bP.filtro_mediano(image=image, kernel_size=kernel_size)
            if(bP.are_images_equal(transformed_image, image)):
                print("Son la misma imagen")
            bP.display_results(image, transformed_image, "Filtro de la mediana")
        
        elif choice == '4' and 'image' in locals():
            kernel_size = int(input("Ingresa el tamaño del filtro (ej. 3 para 3x3): "))
            transformed_image = bP.filtro_maximo(image=image, kernel_size=kernel_size)
            if(bP.are_images_equal(transformed_image, image)):
                print("Son la misma imagen")
            bP.display_results(image, transformed_image, "Filtro del maximo")
        
        elif choice == '5' and 'image' in locals():
            kernel_size = int(input("Ingresa el tamaño del filtro (ej. 3 para 3x3): "))
            transformed_image = bP.filtro_minimo(image=image, kernel_size=kernel_size)
            if(bP.are_images_equal(transformed_image, image)):
                print("Son la misma imagen")
            bP.display_results(image, transformed_image, "Filtro del minimo")

        elif choice == '6' and 'image' in locals():
            transformed_image = bP.escalamiento_abs(bP.filtro_laplaciano(image=image))
            transformed_image2 = bP.sumar_imagenes2(image, transformed_image)
            if(bP.are_images_equal(transformed_image, image)):
                print("Son la misma imagen")
            bP.display_image(transformed_image, title="Filtro laplaciano puro despues del escalamiento abs")
            bP.display_results(image, transformed_image2, "Filtro laplaciano sumado a la imagen original")
        
        elif choice == '7' and 'image' in locals():
            transformed_image = bP.escalamiento_abs(bP.filtro_gradiente(image=image))
            transformed_image2 = bP.sumar_imagenes2(image, transformed_image)
            if(bP.are_images_equal(transformed_image, image)):
                print("Son la misma imagen")
            bP.display_image(transformed_image, title="Filtro del gradiente puro despues del escalamiento abs")
            bP.display_results(image, transformed_image2, "Filtro del gradiente")

        elif choice == '8' and 'image' in locals():
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
