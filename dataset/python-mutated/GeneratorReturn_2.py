def g():
    if False:
        while True:
            i = 10
    for a in range(3):
        yield a
    return 7
print('Yielder with return value', list(g()))