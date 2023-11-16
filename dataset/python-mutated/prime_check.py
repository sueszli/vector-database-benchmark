def prime_check(n):
    if False:
        i = 10
        return i + 15
    'Return True if n is a prime number\n    Else return False.\n    '
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    j = 5
    while j * j <= n:
        if n % j == 0 or n % (j + 2) == 0:
            return False
        j += 6
    return True