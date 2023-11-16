"""
Given an integer num_perfect_squares will return the minimum amount of perfect squares are required
to sum to the specified number. Lagrange's four-square theorem gives us that the answer will always
be between 1 and 4 (https://en.wikipedia.org/wiki/Lagrange%27s_four-square_theorem).

Some examples:
Number | Perfect Squares representation | Answer
-------|--------------------------------|--------
9      | 3^2                            | 1
10     | 3^2 + 1^2                      | 2
12     | 2^2 + 2^2 + 2^2                | 3
31     | 5^2 + 2^2 + 1^2 + 1^2          | 4
"""
import math

def num_perfect_squares(number):
    if False:
        i = 10
        return i + 15
    '\n    Returns the smallest number of perfect squares that sum to the specified number.\n    :return: int between 1 - 4\n    '
    if int(math.sqrt(number)) ** 2 == number:
        return 1
    while number > 0 and number % 4 == 0:
        number /= 4
    if number % 8 == 7:
        return 4
    for i in range(1, int(math.sqrt(number)) + 1):
        if int(math.sqrt(number - i ** 2)) ** 2 == number - i ** 2:
            return 2
    return 3