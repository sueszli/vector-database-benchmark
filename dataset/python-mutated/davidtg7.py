def grada(num):
    if False:
        i = 10
        return i + 15
    if num > 0:
        return '_|'
    return '|_'

def espacio():
    if False:
        print('Hello World!')
    return '  '

def gradas(num):
    if False:
        i = 10
        return i + 15
    if num == 0:
        print('__')
    if num > 0:
        aux = abs(num)
        for i in range(num + 1):
            if num == aux:
                print(espacio() * aux + '_')
                aux -= 1
                continue
            line = espacio() * aux + grada(num)
            print(line)
            aux -= 1
    if num < 0:
        aux = 0
        for i in range(abs(num) + 1):
            if num == aux:
                print('_')
                aux -= 1
                continue
            line = espacio() * aux + grada(num)
            print(line)
            aux += 1