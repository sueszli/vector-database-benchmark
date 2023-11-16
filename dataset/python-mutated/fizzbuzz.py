"""
Write a function that returns an array containing the numbers from 1 to N, 
where N is the parametered value. N will never be less than 1.

Replace certain values however if any of the following conditions are met:

If the value is a multiple of 3: use the value 'Fizz' instead
If the value is a multiple of 5: use the value 'Buzz' instead
If the value is a multiple of 3 & 5: use the value 'FizzBuzz' instead
"""
"\nThere is no fancy algorithm to solve fizz buzz.\n\nIterate from 1 through n\nUse the mod operator to determine if the current iteration is divisible by:\n3 and 5 -> 'FizzBuzz'\n3 -> 'Fizz'\n5 -> 'Buzz'\nelse -> string of current iteration\nreturn the results\nComplexity:\n\nTime: O(n)\nSpace: O(n)\n"

def fizzbuzz(n):
    if False:
        print('Hello World!')
    if n < 1:
        raise ValueError('n cannot be less than one')
    if n is None:
        raise TypeError('n cannot be None')
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append('FizzBuzz')
        elif i % 3 == 0:
            result.append('Fizz')
        elif i % 5 == 0:
            result.append('Buzz')
        else:
            result.append(i)
    return result

def fizzbuzz_with_helper_func(n):
    if False:
        for i in range(10):
            print('nop')
    return [fb(m) for m in range(1, n + 1)]

def fb(m):
    if False:
        print('Hello World!')
    r = (m % 3 == 0) * 'Fizz' + (m % 5 == 0) * 'Buzz'
    return r if r != '' else m