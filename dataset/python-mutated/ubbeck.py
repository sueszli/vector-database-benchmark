"""
*
* Escribe un programa que, dado un número, compruebe y muestre si es primo, fibonacci y par.
* Ejemplos:
* - Con el número 2, nos dirá: "2 es primo, fibonacci y es par"
* - Con el número 7, nos dirá: "7 es primo, no es fibonacci y es impar"
*
"""
import math

def isEven(num: int) -> bool:
    if False:
        i = 10
        return i + 15
    return num % 2 == 0

def isPrime(num: int) -> bool:
    if False:
        print('Hello World!')
    if isEven(num) and num != 2 or num < 2:
        return False
    for i in range(3, int(math.sqrt(num) + 1), 2):
        if num % i == 0:
            return False
    return True

def isFibo(num: int) -> bool:
    if False:
        i = 10
        return i + 15
    if num == 1:
        return True
    fib = [1, 1]
    count = 1
    while fib[count] <= num:
        if fib[count] == num:
            return True
        fib.append(fib[count] + fib[count - 1])
        count += 1
    return False

def check(num: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    even = ('es par', 'es impar')
    prime = ('es primo', 'no es primo')
    fib = ('fibonacci', 'no es fibonacci')
    evenOp = 0 if isEven(num) else 1
    primeOp = 0 if isPrime(num) else 1
    fibOp = 0 if isFibo(num) else 1
    print(f'{num} {prime[primeOp]}, {fib[fibOp]} y {even[evenOp]}')
if __name__ == '__main__':
    for i in range(1, 25):
        check(i)