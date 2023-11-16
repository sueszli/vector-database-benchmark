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
points_table = {0: 'Love', 1: '15', 2: '30', 3: '40'}

def score(puntos1, puntos2, adv):
    if False:
        return 10
    if adv == 2:
        print('El jugador 1 ha ganado')
        return True
    elif adv == -2:
        print('El jugador 2 ha ganado')
        return True
    if puntos1 == 3 and puntos2 == 3:
        if adv > 0:
            print('Ventaja P1')
        elif adv < 0:
            print('ventaja P2')
        else:
            print('Deuce')
    else:
        print(points_table[puntos1], '-', points_table[puntos2])
    return False

def juego_tenis(arr):
    if False:
        i = 10
        return i + 15
    jugador1: int = 0
    jugador2: int = 0
    ventaja: int = 0
    final: bool = False
    for point in arr:
        if not final:
            if point == 'P1' and jugador1 < 3:
                jugador1 += 1
            elif point == 'P2' and jugador2 < 3:
                jugador2 += 1
            elif jugador1 == 3 and jugador2 == 3:
                if point == 'P1':
                    ventaja += 1
                else:
                    ventaja -= 1
            elif point == 'P1':
                ventaja += 2
            elif point == 'P2':
                ventaja -= 2
            else:
                print('Valor ingresado no valido')
            final = score(jugador1, jugador2, ventaja)
        else:
            print('El juego ya ha terminado')
juego_tenis(['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P2', 'P2', 'P1', 'P1', 'P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P2'])