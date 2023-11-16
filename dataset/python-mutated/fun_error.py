try:
    [].append(x=1)
except TypeError:
    print('TypeError')
try:
    round()
except TypeError:
    print('TypeError')
try:
    round(1, 2, 3)
except TypeError:
    print('TypeError')
try:
    [].append(1, 2)
except TypeError:
    print('TypeError')
try:
    [].sort(1)
except TypeError:
    print('TypeError')
try:
    [].sort(noexist=1)
except TypeError:
    print('TypeError')
try:

    def f(x, y):
        if False:
            print('Hello World!')
        pass
    f(x=1)
except TypeError:
    print('TypeError')