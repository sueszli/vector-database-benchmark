"""/*
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
 */
 """
points = ['P2', 'P1', 'P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P2', 'P2']
tenisPoints = {0: 'Love', 1: 15, 2: 30, 3: 40, 4: '  Wins ', 5: 'Ventaja', 6: '       ', 7: '  Lose ', 8: 'Deuce'}
game = {'P1': 0, 'P2': 0}

def setPlayerPoint(player):
    if False:
        while True:
            i = 10
    otherPlayer = 'P2' if player == 'P1' else 'P1'
    if game[player] == 5:
        game[player] = 4
        game[otherPlayer] = 7
    elif game[player] == 6:
        game[player] = 8
        game[otherPlayer] = 8
    elif game[player] + 1 == 3 and game[otherPlayer] == 3:
        game[player] = 8
        game[otherPlayer] = 8
    elif game[player] == 8:
        game[player] = 5
        game[otherPlayer] = 6
    elif game[player] == 3 and game[otherPlayer] == 3:
        game[player] = 8
        game[otherPlayer] = 6
    else:
        game[player] += 1

def calculateGame(points):
    if False:
        return 10
    print('----------------------')
    print('      P1   :   P2       ')
    print('----------------------')
    for point in points:
        setPlayerPoint(point)
        if game['P1'] == 0:
            print('    ' + str(tenisPoints[game['P1']]) + '   :   ' + str(tenisPoints[game['P2']]) + '  ')
        elif game['P2'] == 0:
            print('    ' + str(tenisPoints[game['P1']]) + ' : ' + str(tenisPoints[game['P2']]) + '  ')
        elif game['P1'] == 8 or game['P2'] == 8:
            print('         Deuce   ')
        elif game['P1'] <= 3 and game['P2'] <= 3:
            print('     ' + str(tenisPoints[game['P1']]) + '    :   ' + str(tenisPoints[game['P2']]) + '  ')
        else:
            print('  ' + str(tenisPoints[game['P1']]) + '  :  ' + str(tenisPoints[game['P2']]))
calculateGame(points)
'\n----------------------\n      P1   :   P2       \n----------------------\n    Love   :   15  \n     15    :   15  \n     30    :   15  \n     40    :   15  \n     40    :   30  \n     40    :   40  \n         Deuce   \n         Deuce   \n           :  Ventaja\n    Lose   :    Wins \n'