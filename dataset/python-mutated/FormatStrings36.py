def f():
    if False:
        i = 10
        return i + 15
    a = '1'
    b = '2'
    return f'r:{a!r} and a:{a!a} and s:{a!s}'
print(f())

def f():
    if False:
        print('Hello World!')
    return f''
print(f())

def f():
    if False:
        return 10
    return f'some_text'
print(f())