def validacion():
    if False:
        return 10
    evaf = True
    while evaf == True:
        leer = input('quien gano el punto P1/P2 :')
        if leer.upper() != 'P1' and leer.upper() != 'P2':
            print('opcion invalida, digite P1 o P2')
        else:
            return leer.upper()
p1 = 0
p2 = 0
puntos = ['love', '15', '30', '40']
eva = True
while eva == True:
    x = validacion()
    if x == 'P1':
        print('punto para P1')
        p1 = p1 + 1
    else:
        print('punto para p2')
        p2 = p2 + 1
    if p1 == p2 and p1 >= 3 and (p2 >= 3):
        print('deuce')
    elif p1 - p2 == 1 and p1 >= 4:
        print('ventaja P1')
    elif p2 - p1 == 1 and p2 >= 4:
        print('ventaja P2')
    elif p1 - p2 >= 2 and p1 > 4 or (p1 == 4 and p2 < 3):
        print('ha ganado P1')
        eva = False
    elif p2 - p1 >= 2 and p2 > 4 or (p2 == 4 and p1 < 3):
        print('ha ganado P2')
        eva = False
    elif p1 > 3 or p2 > 3:
        pass
    else:
        print(f'{puntos[p1]} - {puntos[p2]}')