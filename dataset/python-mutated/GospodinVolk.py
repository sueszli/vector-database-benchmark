"""
 * Escribe un programa que muestre cómo transcurre un juego de tenis y quién lo ha ganado.
 * El programa recibirá una secuencia formada por "P1" (Player 1) o "P2" (Player 2), según quien
 * gane cada punto del juego.
 * 
 * - Las puntuaciones de un juego son "Love" (cero), 15, 30, 40, "Deuce" (empate), ventaja.
 * - Ante la secuencia [P1, P1, P2, P2, P1, P2, P1, P1], el programa mostraría lo siguiente:
 *   15 - Love
 *   30 - Love
 *   30 - 15
 *   30 - 30
 *   40 - 30
 *   Deuce
 *   Ventaja P1
 *   Ha ganado el P1
 * - Si quieres, puedes controlar errores en la entrada de datos.   
 * - Consulta las reglas del juego si tienes dudas sobre el sistema de puntos.   
"""

def puntuacion(p1, p2):
    if False:
        print('Hello World!')
    deuce = False
    if p1 > 2 and p2 > 2:
        deuce = True
    dict = {0: 'Love', 1: '15', 2: '30', 3: '40'}
    if p1 > 3 and p1 > p2 + 1:
        print('El jugador 1 gana el partido')
    elif p2 > 3 and p1 > p1 + 1:
        print('El jugador 2 gana el partido')
    elif deuce and p1 == p2:
        print('Deuce')
    elif deuce and p1 > p2:
        print('Ventaja para el jugador 1')
    elif deuce and p1 < p2:
        print('Ventaja para el jugador 2')
    else:
        print(dict[p1] + ' - ' + dict[p2])
puntos_1 = 0
puntos_2 = 0
cadena = input('Introduzca la secuencia: ')
for letra in cadena:
    if letra == '1':
        puntos_1 += 1
    elif letra == '2':
        puntos_2 += 1
    else:
        continue
    puntuacion(puntos_1, puntos_2)