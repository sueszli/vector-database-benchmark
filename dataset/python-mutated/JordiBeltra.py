def DibujarEscalera(x: int) -> str:
    if False:
        return 10
    if x == 0:
        print('__')
    elif x < 0:
        print('_')
        for i in range(abs(x)):
            print(' ' * (i + 1) + ' ' * i + '|_')
    else:
        print(' ' * (x * 2 + 1) + '_')
        for i in range(x):
            print(' ' * (x - i - 1) + ' ' * (x - i) + '_|')