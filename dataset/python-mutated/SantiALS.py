"""
/*
 * Crea una función que sea capaz de leer el número representado por el ábaco.
 * - El ábaco se representa por un array con 7 elementos.
 * - Cada elemento tendrá 9 "O" (aunque habitualmente tiene 10 para realizar operaciones)
 *   para las cuentas y una secuencia de "---" para el alambre.
 * - El primer elemento del array representa los millones, y el último las unidades.
 * - El número en cada elemento se representa por las cuentas que están a la izquierda del alambre.
 *
 * Ejemplo de array y resultado:
 * ["O---OOOOOOOO",
 *  "OOO---OOOOOO",
 *  "---OOOOOOOOO",
 *  "OO---OOOOOOO",
 *  "OOOOOOO---OO",
 *  "OOOOOOOOO---",
 *  "---OOOOOOOOO"]
 *  
 *  Resultado: 1.302.790
 */
"""

def abaco(cadena: list) -> str:
    if False:
        i = 10
        return i + 15
    millon = ''
    miles = ''
    cientos = ''
    separador = 1
    for fila in cadena:
        if separador == 1:
            millon += str(fila.index('-'))
            separador += 1
        elif separador > 1 and separador <= 4:
            miles += str(fila.index('-'))
            separador += 1
        else:
            cientos += str(fila.index('-'))
    return (millon, miles, cientos)
cadena = ['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOOOOOO---OO', 'OOOOOOOOO---', '---OOOOOOOOO']
(millon, miles, cientos) = abaco(cadena)
print(f'Resultado: {millon}.{miles}.{cientos}')