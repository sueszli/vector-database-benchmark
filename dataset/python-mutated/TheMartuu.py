def tenis_game(secuencia):
    if False:
        print('Hello World!')
    score_p1 = 0
    score_p2 = 0
    score_board = ['Love', '15', '30', '40']
    for i in secuencia:
        if i == 'P1':
            score_p1 += 1
        elif i == 'P2':
            score_p2 += 1
        else:
            print('Jugador incorrecto, intente nuevamente')
            return ()
        if score_p1 > 3 or score_p2 > 3:
            if score_p1 == score_p2:
                print('Deuce')
            elif score_p1 > score_p2:
                print('P1 es el ganador!')
                return ()
            elif score_p1 - 1 == score_p2:
                print('Ventaja para P1')
            elif score_p2 > score_p1:
                print('P2 es el ganador')
            elif score_p2 - 1 > score_p1:
                print('Ventaja para P2')
                return ()
            else:
                print('Resultado inválido')
        if score_p1 == 3 and score_p2 == 3:
            print('Deuce')
        else:
            print(f'{score_board[score_p1]} - {score_board[score_p2]}')
tenis_game(['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1'])