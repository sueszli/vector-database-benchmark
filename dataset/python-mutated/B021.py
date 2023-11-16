f'\nShould emit:\nB021 - on lines 14, 22, 30, 38, 46, 54, 62, 70, 73\n'
VARIABLE = 'world'

def foo1():
    if False:
        print('Hello World!')
    'hello world!'

def foo2():
    if False:
        for i in range(10):
            print('nop')
    f'hello {VARIABLE}!'

class bar1:
    """hello world!"""

class bar2:
    f'hello {VARIABLE}!'

def foo1():
    if False:
        i = 10
        return i + 15
    'hello world!'

def foo2():
    if False:
        print('Hello World!')
    f'hello {VARIABLE}!'

class bar1:
    """hello world!"""

class bar2:
    f'hello {VARIABLE}!'

def foo1():
    if False:
        return 10
    'hello world!'

def foo2():
    if False:
        print('Hello World!')
    f'hello {VARIABLE}!'

class bar1:
    """hello world!"""

class bar2:
    f'hello {VARIABLE}!'

def foo1():
    if False:
        i = 10
        return i + 15
    'hello world!'

def foo2():
    if False:
        while True:
            i = 10
    f'hello {VARIABLE}!'

class bar1:
    """hello world!"""

class bar2:
    f'hello {VARIABLE}!'

def baz():
    if False:
        print('Hello World!')
    f"I'm probably a docstring: {VARIABLE}!"
    print(f"I'm a normal string")
    f"Don't detect me!"