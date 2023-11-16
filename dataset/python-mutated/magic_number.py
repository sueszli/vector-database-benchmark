"""
Magic Number
A number is said to be a magic number,
if summing the digits of the number and then recursively repeating this process for the given sum
untill the number becomes a single digit number equal to 1.

Example:
    Number = 50113 => 5+0+1+1+3=10 => 1+0=1 [This is a Magic Number]
    Number = 1234 => 1+2+3+4=10 => 1+0=1 [This is a Magic Number]
    Number = 199 => 1+9+9=19 => 1+9=10 => 1+0=1 [This is a Magic Number]
    Number = 111 => 1+1+1=3 [This is NOT a Magic Number]

The following function checks for Magic numbers and returns a Boolean accordingly.
"""

def magic_number(n):
    if False:
        while True:
            i = 10
    ' Checks if n is a magic number '
    total_sum = 0
    while n > 0 or total_sum > 9:
        if n == 0:
            n = total_sum
            total_sum = 0
        total_sum += n % 10
        n //= 10
    return total_sum == 1