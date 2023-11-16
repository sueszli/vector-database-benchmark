def bad_function():
    if False:
        print('Hello World!')
    'this docstring is not capitalized'

def good_function():
    if False:
        print('Hello World!')
    'This docstring is capitalized.'

def other_function():
    if False:
        return 10
    '\n    This docstring is capitalized.'

def another_function():
    if False:
        i = 10
        return i + 15
    ' This docstring is capitalized.'

def utf8_function():
    if False:
        print('Hello World!')
    'éste docstring is capitalized.'

def uppercase_char_not_possible():
    if False:
        print('Hello World!')
    "'args' is not capitalized."

def non_alphabetic():
    if False:
        for i in range(10):
            print('nop')
    'th!is is not capitalized.'

def non_ascii():
    if False:
        for i in range(10):
            print('nop')
    'th•s is not capitalized.'

def all_caps():
    if False:
        print('Hello World!')
    'th•s is not capitalized.'