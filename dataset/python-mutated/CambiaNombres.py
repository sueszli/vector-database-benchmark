"""
 * Los primeros dispositivos móviles tenían un teclado llamado T9
 * con el que se podía escribir texto utilizando únicamente su
 * teclado numérico (del 0 al 9).
"""

def t9(entrada: str) -> str:
    if False:
        while True:
            i = 10
    salida = ''
    teclas = [['.', ',', "'", '1'], ['a', 'b', 'c', '2'], ['d', 'e', 'f', '3'], ['g', 'h', 'i', '4'], ['j', 'k', 'l', '5'], ['m', 'n', 'o', '6'], ['p', 'q', 'r', 's', '7'], ['t', 'u', 'v', '8'], ['w', 'x', 'y', 'z', '9'], [' ', '0']]
    block = ''
    if entrada == '':
        return salida
    else:
        lista = entrada.split('-')
        for block in lista:
            added = False
            if block.isdigit():
                for number in block:
                    if len(block) == block.count(str(number)):
                        if not added:
                            salida += teclas[int(number) - 1][len(block) - 1]
                            added = True
                    else:
                        return ''
            else:
                return ''
        return salida
'\n * - Ejemplo:\n *     Entrada: 6-666-88-777-33-3-33-888\n *     Salida: MOUREDEV\n'
print(t9('6-666-88-777-33-3-33-888'))
print(t9('44-666-555-2'))