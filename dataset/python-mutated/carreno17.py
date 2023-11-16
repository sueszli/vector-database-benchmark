"""
 * Crea una función que reciba un número decimal y lo trasforme a Octal
 * y Hexadecimal.
 * - No está permitido usar funciones propias del lenguaje de programación que
 * realicen esas operaciones directamente.

"""

def main(numero: int):
    if False:
        while True:
            i = 10
    print(decimal_a_octal(numero))
    print(decimal_a_hexadecimal(numero))

def decimal_a_octal(numero):
    if False:
        while True:
            i = 10
    octal = ''
    while numero > 0:
        residuo = numero % 8
        octal = str(residuo) + octal
        numero //= 8
    return f'EL numero octal es: {octal} '

def decimal_a_hexadecimal(numero: int):
    if False:
        for i in range(10):
            print('nop')
    hexadecimal = ''
    while numero > 0:
        residuo = numero % 16
        if residuo < 10:
            hexadecimal = str(residuo) + hexadecimal
        else:
            hexadecimal = chr(residuo + 55) + hexadecimal
        numero //= 16
    return f'EL numero octal es: {hexadecimal} '
if __name__ == 'main':
    main(255)