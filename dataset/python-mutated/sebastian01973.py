players = ('P1', 'P2')
points = ('Love', 15, 30, 40)
countP1 = 0
countP2 = 0

def result(value):
    if False:
        while True:
            i = 10
    return points[value]

def rules(player):
    if False:
        for i in range(10):
            print('nop')
    global countP1, countP2
    if player == players[0]:
        countP1 += 1
    if player == players[1]:
        countP2 += 1
    logic(countP1, countP2)

def logic(countP1, countP2):
    if False:
        i = 10
        return i + 15
    if countP1 <= 3 and countP2 <= 3:
        if countP2 == 3 or countP1 == 3:
            if countP2 == countP1:
                print('Deuce')
            else:
                print(f'{result(countP1)} - {result(countP2)}')
        else:
            print(f'{result(countP1)} - {result(countP2)}')
    elif countP1 > countP2:
        if countP1 > countP2 + 1:
            print(f'Ha ganado el {players[0]}')
        else:
            print(f'Ventaja {players[0]}')
    elif countP2 > countP1 + 1:
        print(f'Ha ganado el {players[1]}')
    else:
        print(f'Ventaja {players[1]}')

def game(sequence):
    if False:
        i = 10
        return i + 15
    for i in sequence:
        rules(i)
if __name__ == '__main__':
    sequence = ('P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1')
    sequence1 = ('P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P2', 'P2')
    sequence2 = ('P1', 'P1', 'P1', 'P1')
    game(sequence)