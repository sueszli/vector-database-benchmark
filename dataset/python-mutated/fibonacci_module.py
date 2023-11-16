"""Fibonacci numbers module.

@see: https://docs.python.org/3/tutorial/modules.html

A module is a file containing Python definitions and statements. The file name is the module name
with the suffix .py appended. Within a module, the module’s name (as a string) is available as the
value of the global variable __name__.
"""

def fibonacci_at_position(position):
    if False:
        while True:
            i = 10
    'Return Fibonacci number at specified position'
    current_position = 0
    (previous_number, current_number) = (0, 1)
    while current_position < position:
        current_position += 1
        (previous_number, current_number) = (current_number, previous_number + current_number)
    return previous_number

def fibonacci_smaller_than(limit):
    if False:
        for i in range(10):
            print('nop')
    'Return Fibonacci series up to limit'
    result = []
    (previous_number, current_number) = (0, 1)
    while previous_number < limit:
        result.append(previous_number)
        (previous_number, current_number) = (current_number, previous_number + current_number)
    return result
if __name__ == '__main__':
    import sys
    print(fibonacci_smaller_than(int(sys.argv[1])))