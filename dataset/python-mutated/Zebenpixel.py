import math

def check_prime_fibonacci_even(number):
    if False:
        for i in range(10):
            print('nop')
    result = f'{number} '
    if number > 1:
        for index in range(2, number):
            if number % index == 0:
                result += 'no es primo, '
                break
        else:
            result += 'es primo, '
    else:
        result += 'no es primo, '
    result += 'es fibonacci ' if number > 0 and is_perfect_square(5 * number * number + 4 or is_perfect_square(5 * number * number - 4)) else 'no es fibonacci '
    result += 'y es par' if number % 2 == 0 else 'y es impar'
    print(result)

def is_perfect_square(number):
    if False:
        print('Hello World!')
    sqrt = int(math.sqrt(number))
    return sqrt * sqrt == number
check_prime_fibonacci_even(2)
check_prime_fibonacci_even(7)
check_prime_fibonacci_even(0)