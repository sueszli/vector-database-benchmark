def func_punt_tenis(lp):
    if False:
        for i in range(10):
            print('nop')
    puntos = ['Love', '15', '30', '40']
    j1 = 0
    j2 = 0
    for p in lp:
        if p == 'P1':
            j1 += 1
        elif p == 'P2':
            j2 += 1
        else:
            print('El numero del player no es correcto, intentelo otra vez')
            return ()
        if j1 > j2 + 1 and j1 > 3:
            print('Ha ganado el P1')
            return ()
        elif j2 > j1 + 1 and j2 > 3:
            print('Ha ganado el P2')
            return ()
        elif j1 == j2 and j1 >= 3:
            print('Deuce')
        elif j1 > j2 and j1 > 3:
            print('Ventaja P1')
        elif j1 < j2 and j1 > 3:
            print('Ventaja P2')
        else:
            print(puntos[j1], '-', puntos[j2])
func_punt_tenis(['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P2', 'P2', 'P2'])