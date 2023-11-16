juego = ['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1']
puntuacion = {0: 'Love', 1: '15', 2: '30', 3: '40'}

def juegoTennis(juego):
    if False:
        while True:
            i = 10
    resta = 0
    contadorp1 = 0
    contadorp2 = 0
    for i in juego:
        if i == 'P1':
            contadorp1 += 1
        if i == 'P2':
            contadorp2 += 1
        if contadorp1 < 3 or contadorp2 < 3:
            print(f'{puntuacion[contadorp1]} - {puntuacion[contadorp2]}')
        elif contadorp1 == 3 and contadorp2 == 3:
            print('Deuce')
        elif contadorp1 > 3 or contadorp2 > 3:
            resta = contadorp1 - contadorp2
            if resta == 0:
                print('Deuce')
            elif resta == 1:
                print('Ventaja P1')
            elif resta == 2:
                print('Ganador P1')
            elif resta == -1:
                print('Ventaja P2')
            elif resta == -2:
                print('Ganador P2')
juegoTennis(juego)
juego1 = ['P1', 'P2', 'P2', 'P1', 'P1', 'P2', 'P1', 'P2', 'P2', 'P2']
juegoTennis(juego1)