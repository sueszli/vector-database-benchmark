def tabla_multiplicacion(number: int) -> str:
    if False:
        i = 10
        return i + 15
    ' Funcion encargue de imprimir su tabla de multiplicar entre el 1 y el 10.\n    Args:\n        number (int): Numero del cual se quiere obtener la tabla de multiplicar.\n    Returns:\n        str: Tabla de multiplicar del numero ingresado.\n    Nota:\n        Debe visualizarse qué operación se realiza y su resultado.\n        Ej: 1 x 1 = 1\n            1 x 2 = 2\n            1 x 3 = 3\n    '
    return '\n'.join([f'{number} x {inicio} = {number * inicio}' for inicio in range(1, 11)])
if __name__ == '__main__':
    continuar = 'si'
    while continuar.lower() == 'si':
        valor = input('Ingrese un numero para obtener su tabla de multiplicar \n')
        valor = int(valor) if valor.isnumeric() else -1
        if valor >= 0:
            print(tabla_multiplicacion(int(valor)))
            continuar = input('¿Desea continuar? (si/no) \n')
            continue
        print('El numero ingresado debe ser positivo')