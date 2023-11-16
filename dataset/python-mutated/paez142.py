def read_number(abaco: list()) -> list():
    if False:
        while True:
            i = 10
    acum = 0
    numero = ' '
    if valid_abaco(abaco):
        for number in abaco:
            for digit in number:
                if digit == 'O':
                    acum += 1
                else:
                    digito = acum
                    acum = 0
                    numero += str(digito)
                    break
        result = '{:,}'.format(int(numero))
        return result
    return 'Abaco No Valido'

def valid_abaco(abaco) -> bool:
    if False:
        return 10
    if len(abaco) != 7:
        return False
    for element in abaco:
        if len(element) != 12 or element.count('O') != 9:
            return False
    return True
print(read_number(['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOO---OOOOOO', 'OOOOOOOOO---', '---OOOOOOOOO']))