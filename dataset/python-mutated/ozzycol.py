"""
Created on Tue Jan 10 17:21:27 2023

@author: oardila
"""
'\n/*\n * Escribe un programa que muestre cómo transcurre un juego de tenis y quién lo ha ganado.\n * El programa recibirá una secuencia formada por "P1" (Player 1) o "P2" (Player 2), según quien\n * gane cada punto del juego.\n * \n * - Las puntuaciones de un juego son "Love" (cero), 15, 30, 40, "Deuce" (empate), ventaja.\n * - Ante la secuencia [P1, P1, P2, P2, P1, P2, P1, P1], el programa mostraría lo siguiente:\n *   15 - Love\n *   30 - Love\n *   30 - 15\n *   30 - 30\n *   40 - 30\n *   Deuce\n *   Ventaja P1\n *   Ha ganado el P1\n * - Si quieres, puedes controlar errores en la entrada de datos.   \n * - Consulta las reglas del juego si tienes dudas sobre el sistema de puntos.   \n */\n'
import random
players = ['P1', 'P2']
points = {0: 'Love', 1: '15', 2: '30', 3: '40'}
count_p1 = 0
count_p2 = 0
while True:
    if random.choice(players) == 'P1':
        count_p1 += 1
    else:
        count_p2 += 1
    if count_p1 == 4 and count_p2 < 3:
        print('Ha ganado el P1')
        break
    elif count_p2 == 4 and count_p1 < 3:
        print('Ha ganado el P2')
        break
    elif count_p1 == 3 and count_p2 == 3:
        print('Deuce')
        count = 0
        while count > -2 and count < 2:
            if random.choice(players) == 'P1':
                count += 1
            else:
                count -= 1
            if count == 0:
                print('Deuce')
            elif count == 1:
                print('Ventaja P1')
            elif count == -1:
                print('Ventaja P2')
            elif count == 2:
                print('Ha ganado el P1')
            elif count == -2:
                print('Ha ganado el P2')
        break
    else:
        print(f'{points[count_p1]} - {points[count_p2]}')
import random
players = ['P1', 'P2']

def convert_points(points):
    if False:
        while True:
            i = 10
    if points == 0:
        return 'Love'
    elif points == 1:
        return '15'
    elif points == 2:
        return '30'
    elif points == 3:
        return '40'
p1 = 0
p2 = 0
while True:
    p = random.choice(players)
    if p == 'P1':
        p1 += 1
    elif p == 'P2':
        p2 += 1
    else:
        print('El jugador introducido no es valido, el juego sigue: ')
    if p1 == 3 and p2 == 3:
        print('Deuce')
    elif p1 <= 3 and p2 <= 3:
        print(convert_points(p1) + ' - ' + convert_points(p2))
    elif p1 - 1 == p2:
        print('Ventaja para el jugador P1')
    elif p2 - 1 == p1:
        print('Ventaja para el jugador P2')
    elif p1 > p2:
        print('El jugador P1 ha ganado')
        break
    elif p2 > p1:
        print('El jugador p2 ha ganado')
        break
    else:
        print('Deuce')