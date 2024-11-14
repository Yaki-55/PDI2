import bibliotecaPDI as bP

def main_menu():
    while True:
        print("\n--- Menú de Procesamiento de Imagen ---")
        print("1. Cargar imagen")
        print("2. Ecualización de histograma global")
        print("3. Ecualización de histograma local")
        print("4. Calcular media y varianza global")
        print("5. Calcular media y varianza local")
        print("6. Realce de la imagen")
        print("7. Mostrar imagen cargada")
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
            transformed_image = bP.global_histogram_equalization(image)
            bP.display_results(image, transformed_image, "Ecualización de Histograma Global")
        
        elif choice == '3' and 'image' in locals():
            clip_limit= float(input('Limite de recorte (El valor predeterminado es 2.0): '))
            matrix = int(input('Tamaño de la matriz (ej. 3 para 3x3): '))
            transformed_image = bP.local_histogram_equalization(image, clip_limit=clip_limit, tile_grid_size=(matrix, matrix))
            bP.display_results(image, transformed_image, "Ecualización de Histograma Local")
        
        elif choice == '4' and 'image' in locals():
            mean, variance = bP.global_mean_and_variance(image)
            print(f"Media Global: {mean:.2f}, Varianza Global: {variance:.2f}")
        
        elif choice == '5' and 'image' in locals():
            kernel_size = int(input("Ingresa el tamaño de la vecindad (ej. 3 para 3x3): "))
            mean_local, variance_local = bP.local_mean_and_variance(image, (kernel_size, kernel_size))
            bP.display_results(image, mean_local, "Media Local")
            bP.display_results(image, variance_local, "Varianza Local")

        elif choice == '6' and 'image' in locals():
            E = float(input("Ingrese el valor de E: "))
            k0 = float(input("Ingrese el valor de k0: "))
            k1 = float(input("Ingrese el valor de k1: "))
            k2 = float(input("Ingrese el valor de k2: "))
            mask_size = int(input("Ingrese el tamaño de la máscara (ej. 3 para 3x3): "))
            transformed_image = bP.enhanced_contrast_filter(image, E, k0, k1, k2, (mask_size, mask_size))
            bP.display_results(image, transformed_image, "Realce de Contraste con Filtro Mejorado")
        
        elif choice == '7' and 'image' in locals():
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
