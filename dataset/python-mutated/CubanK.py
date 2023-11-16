message_helper = {True: 'is', False: "isn't"}

def fibonacci(number: int):
    if False:
        return 10
    '\n    Fibonacci using loop while\n    '
    serie = []
    stop = False
    while number >= 0:
        x = serie[-2] if len(serie) > 1 else 0
        y = serie[-1] if len(serie) > 1 else 1 * len(serie)
        z = x + y
        serie.append(z)
        if z > number:
            break
    return serie

def is_prime(number: int) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Define if a number is prime\n    '
    if number < 2:
        return False
    for i in range(2, number):
        if number % i == 0:
            return False
    return True

def is_even(number: int) -> bool:
    if False:
        return 10
    '\n    Define if a number is even\n    '
    if number % 2 == 0:
        return True
    return False

def is_fibinacci(number: int, x: int=0, y: int=1) -> bool:
    if False:
        while True:
            i = 10
    '\n    Fibonacci using recursion\n    '
    z = x + y
    if number == z:
        return True
    elif number > z:
        return is_fibinacci(number=number, x=y, y=x + y)
    else:
        return False

def number_analize() -> str:
    if False:
        for i in range(10):
            print('nop')
    result = f''
    source = input('Enter a number:\n')
    try:
        number = int(source)
        result = f'{number} {message_helper[is_prime(number=number)]} prime, {message_helper[is_fibinacci(number=number)]} fibonacci and {message_helper[is_even(number=number)]} even.'
    except ValueError as e:
        result = f'Value error. {source} {message_helper[False]} integer.\nDescription: {e}'
    finally:
        print(result)
if __name__ == '__main__':
    number_analize()