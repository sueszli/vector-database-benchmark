"""
Created on Sun Jan 15 11:38:15 2023

@author: rafadataengineer
"""
'\n/*\n * Escribe un programa que muestre cómo transcurre un juego de tenis y quién lo ha ganado.\n * El programa recibirá una secuencia formada por "P1" (Player 1) o "P2" (Player 2), según quien\n * gane cada punto del juego.\n * \n * - Las puntuaciones de un juego son "Love" (cero), 15, 30, 40, "Deuce" (empate), ventaja.\n * - Ante la secuencia [P1, P1, P2, P2, P1, P2, P1, P1], el programa mostraría lo siguiente:\n *   15 - Love\n *   30 - Love\n *   30 - 15\n *   30 - 30\n *   40 - 30\n *   Deuce\n *   Ventaja P1\n *   Ha ganado el P1\n * - Si quieres, puedes controlar errores en la entrada de datos.   \n * - Consulta las reglas del juego si tienes dudas sobre el sistema de puntos.   \n */\n '
import os
secuencia = []
PLAYER1 = 1
PLAYER2 = 2

def main():
    if False:
        print('Hello World!')
    continuar = True
    while continuar:
        puntuaciones = {0: 'Love', 1: '15', 2: '30', 3: '40'}
        contador_p1 = 0
        contador_p2 = 0
        os.system('cls')
        print('                Partido de Tenis')
        print('~' * 50)
        print(f'\nPunto para el jugador:\n\n    {PLAYER1}) PLAYER 1\n    {PLAYER2}) PLAYER 2\n')
        print('~' * 50)
        try:
            punto = int(input('Seleccione una opcion: '))
        except:
            punto = -1
        if punto == PLAYER1:
            secuencia.append('P1')
        elif punto == PLAYER2:
            secuencia.append('P2')
        else:
            print('Opcion no valida')
        print('\nPuntuacion actual: ')
        print('-' * 22)
        ESPACIOS = 8
        print(f"{'P1': <{ESPACIOS}}{'-': <{ESPACIOS}}{'P2'}")
        print('-' * 22)
        if len(secuencia) == 0:
            print('No ha iniciado el partido')
        else:
            for i in secuencia:
                if i == 'P1':
                    contador_p1 += 1
                elif i == 'P2':
                    contador_p2 += 1
                if contador_p1 - contador_p2 >= 2 and contador_p1 > 3:
                    print('Ha ganado el P1')
                    continuar = False
                elif contador_p2 - contador_p1 >= 2 and contador_p2 > 3:
                    print('Ha ganado el P2')
                    continuar = False
                elif contador_p1 == contador_p2 and contador_p1 >= 3:
                    print('Deuce')
                elif contador_p1 - contador_p2 == 1 and contador_p1 > 3:
                    print('Ventaja P1')
                elif contador_p2 - contador_p1 == 1 and contador_p2 > 3:
                    print('Ventaja P2')
                else:
                    print(f"{puntuaciones[contador_p1]: <{ESPACIOS}}{'-': <{ESPACIOS}}{puntuaciones[contador_p2]}")
        print('\n')
        print('~' * 50)
        input('Presiona enter para continuar...')
    print('Ha finalizado el juego. Nos vemos')
if __name__ == '__main__':
    main()