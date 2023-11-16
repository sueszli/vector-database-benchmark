def esPrimo(num):
    if False:
        print('Hello World!')
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

def esGemelo(num1, num2):
    if False:
        print('Hello World!')
    if num1 - num2 == 2 or num2 - num1 == 2:
        return True
    return False

def main():
    if False:
        while True:
            i = 10
    rango = int(input('Ingrese el rango: '))
    for i in range(2, rango):
        if esPrimo(i) and esPrimo(i + 2):
            if esGemelo(i, i + 2):
                print(f'({i}, {i + 2})')
if __name__ == '__main__':
    main()