def multiply(number):
    if False:
        for i in range(10):
            print('nop')
    number = int(number)
    for multiplier in range(1, 11):
        result = number * multiplier
        print(f'{number} x {multiplier} = {result}')
number = input('Introduce un número: ')
multiply(number)