"""/*
 * Crea una función que reciba una expresión matemática (String)
 * y compruebe si es correcta. Retornará true o false.
 * - Para que una expresión matemática sea correcta debe poseer
 *   un número, una operación y otro número separados por espacios.
 *   Tantos números y operaciones como queramos.
 * - Números positivos, negativos, enteros o decimales.
 * - Operaciones soportadas: + - * / %
 *
 * Ejemplos:
 * "5 + 6 / 7 - 4" -> true
 * "5 a 6" -> false
 */"""

def expresion(expresion: str) -> bool:
    if False:
        while True:
            i = 10
    'Revisa si la expresión es una función matematica del tipo:\n    numero + esp + operación + ...\n\n    Args:\n        expresion (str): expresion a analizar\n\n    Returns:\n        bool: Es o no una expresión matematica\n\n\n    python3 -m doctest -v expresion.py\n\n    >>> expresion("5 + 6 / 7 - 4")\n    True\n    >>> expresion("5 a 6")\n    False\n    '
    lista = expresion.split()
    salida_operacion = False
    salida_numero = True
    for i in range(len(lista)):
        if i % 2 == 0:
            try:
                valor = float(lista[i])
            except:
                salida_numero = False
        elif lista[i] in '+-*/%':
            salida_operacion = True
    if salida_operacion == True and salida_numero == True:
        return True
    else:
        return False
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    formula = input('¿Qué expresión matematica quieres analizar? ')
    print(f'La expresión: "{formula}" es: {expresion(formula)}')