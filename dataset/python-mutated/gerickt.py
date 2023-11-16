import time

def time_func(func):
    if False:
        print('Hello World!')

    def wrapper(*args, **kwargs):
        if False:
            return 10
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Tiempo transcurrido: {(end - start) * 1000:.3f} ms')
        return result
    return wrapper

@time_func
def read_abacus(abaco: list):
    if False:
        i = 10
        return i + 15
    numero = ''
    for cuenta in abaco:
        index = cuenta.find('-')
        numero += str(index)
    return 'Resultado: {:,}'.format(int(numero)).replace(',', '.')

@time_func
def read_abacus_comprehension(abaco: list):
    if False:
        for i in range(10):
            print('nop')
    lista = [str(cuenta.find('-')) for cuenta in abaco]
    numero = int(''.join(lista))
    return 'Resultado: {:,}'.format(numero).replace(',', '.')
if __name__ == '__main__':
    input = ['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOOOOOO---OO', 'OOOOOOOOO---', '---OOOOOOOOO']
    print(read_abacus(input))
    print(read_abacus_comprehension(input))