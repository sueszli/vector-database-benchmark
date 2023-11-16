"""
/*
 * Crea un programa que sea capaz de generar e imprimir todas las 
 * permutaciones disponibles formadas por las letras de una palabra.
 * - Las palabras generadas no tienen por qué existir.
 * - Deben usarse todas las letras en cada permutación.
 * - Ejemplo: sol, slo, ols, osl, los, lso 
 */
"""

def permutaciones(palabra, inicio=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Genera todas las permutaciones únicas de una palabra dada.\n\n    Esta función utiliza un enfoque recursivo para generar todas las permutaciones posibles de los caracteres\n    en la palabra de entrada. Las permutaciones se almacenan en un conjunto (set) para garantizar que solo se\n    incluyan permutaciones únicas.\n\n    Args:\n        palabra (str): La palabra de entrada para la cual se generarán las permutaciones.\n        inicio (int, opcional): El índice de inicio para la generación de permutaciones. No es necesario\n        especificarlo al llamar a la función.\n\n    Returns:\n        set: Un conjunto que contiene todas las permutaciones únicas de la palabra de entrada.\n\n    Ejemplo:\n        >>> resultado = permutaciones("abcd")\n\n    Notas:\n        - Esta función puede ser lenta para palabras muy largas debido a su enfoque recursivo.\n        - Se recomienda su uso con palabras de longitud razonable.\n\n    '
    if not isinstance(palabra, list):
        palabra = list(palabra)
    salida = set()
    if inicio == len(palabra) - 1:
        salida.add(''.join(palabra))
    else:
        for i in range(inicio, len(palabra)):
            (palabra[inicio], palabra[i]) = (palabra[i], palabra[inicio])
            salida.update(permutaciones(palabra, inicio + 1))
            (palabra[inicio], palabra[i]) = (palabra[i], palabra[inicio])
    return salida
if __name__ == '__main__':
    print(permutaciones('sol'))