def point_for():
    if False:
        return 10
    point = input('Introduce quien anoto el punto: ')
    point = point.lower()
    if point == 'p1' or point == 'p2':
        return point
    else:
        print('Disculpa comando no valido vuelve a intentar ')
        return point_for()
p1 = 0
p2 = 0

def game():
    if False:
        while True:
            i = 10
    p1 = 0
    p2 = 0
    game = ['Love', '15', '30', '40']
    winner = False
    while winner != True:
        point = point_for()
        if point == 'p1':
            p1 += 1
        elif point == 'p2':
            p2 += 1
        if p1 >= 3 and p2 >= 3:
            if abs(p1 - p2) <= 1:
                print('Deuce' if p1 == p2 else 'Ventaja P1' if p1 > p2 else 'Ventaja P2')
            else:
                winner = True
                msj = 'Ha ganado el P1' if p1 > p2 else 'Ha Ganado el P2'
        elif p1 < 4 and p2 < 4:
            print(f'{game[p1]} - {game[p2]}')
        else:
            winner = True
            msj = 'Ha ganado el P1' if p1 > p2 else 'Ha Ganado el P2'
    return msj
print(game())